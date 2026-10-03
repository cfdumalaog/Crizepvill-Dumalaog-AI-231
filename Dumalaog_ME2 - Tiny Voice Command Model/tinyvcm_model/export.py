"""Export trained TinyDSCNN-48 VCM to FP32 ONNX and calibrated static INT8 ONNX."""
from pathlib import Path
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from .config import CLASSES, FEATURE_SHAPE, LABELS, ROOT
from .model import TinyDSCNN


class ValCalibrationReader(CalibrationDataReader):
    def __init__(self, val_features, max_samples=256):
        self.samples = [val_features[i:i + 1] for i in range(min(max_samples, len(val_features)))]
        self.iter = iter(self.samples)

    def get_next(self):
        val = next(self.iter, None)
        if val is None:
            return None
        return {"input": val}


def export_models(run_dir, val_features, test_features, test_targets, labels=None, artifact_prefix='vcm_optionb'):
    labels = list(LABELS if labels is None else labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('labels must be a non-empty sequence of unique class names')
    if Path(artifact_prefix).name != artifact_prefix:
        raise ValueError('artifact_prefix must be a filename stem')
    run_dir = Path(run_dir)
    models_dir = run_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    weights_path = run_dir / 'best_model.pt'
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # 1. Load PyTorch Model
    model = TinyDSCNN(classes=len(labels), channels=48)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, *FEATURE_SHAPE)
    fp32_path = models_dir / f'{artifact_prefix}_fp32.onnx'
    int8_path = models_dir / f'{artifact_prefix}_int8.onnx'

    print(f"Exporting FP32 ONNX model to {fp32_path.name}...", flush=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    fp32_size = fp32_path.stat().st_size
    print(f"FP32 ONNX exported: {fp32_size:,} bytes.", flush=True)

    # 2. Static INT8 Quantization with Calibration Data
    print(f"Calibrating and quantizing to static INT8 ONNX...", flush=True)
    reader = ValCalibrationReader(val_features, max_samples=256)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True
    )
    onnx.checker.check_model(onnx.load(str(int8_path)))
    int8_size = int8_path.stat().st_size
    print(f"Static INT8 ONNX exported: {int8_size:,} bytes.", flush=True)

    # 3. Numerical Parity Verification on Test Set
    print("Verifying parity on held-out test split...", flush=True)
    session_fp32 = ort.InferenceSession(str(fp32_path), providers=['CPUExecutionProvider'])
    session_int8 = ort.InferenceSession(str(int8_path), providers=['CPUExecutionProvider'])

    # Test accuracy for both
    n = len(test_targets)
    preds_fp32 = []
    preds_int8 = []

    for i in range(n):
        inp = test_features[i:i + 1]
        out_fp32 = session_fp32.run(None, {'input': inp})[0]
        out_int8 = session_int8.run(None, {'input': inp})[0]
        preds_fp32.append(np.argmax(out_fp32))
        preds_int8.append(np.argmax(out_int8))

    acc_fp32 = float(np.mean(np.array(preds_fp32) == test_targets))
    acc_int8 = float(np.mean(np.array(preds_int8) == test_targets))
    parity_rate = float(np.mean(np.array(preds_fp32) == np.array(preds_int8)))

    print(f"Test Accuracy FP32 ONNX: {acc_fp32*100:.2f}%", flush=True)
    print(f"Test Accuracy INT8 ONNX: {acc_int8*100:.2f}% (Drop: {(acc_fp32 - acc_int8)*100:.2f}%)", flush=True)
    print(f"Prediction Parity (FP32 vs INT8): {parity_rate*100:.2f}%", flush=True)

    # 4. CPU Latency Benchmark
    t_records = []
    dummy = np.random.randn(1, *FEATURE_SHAPE).astype(np.float32)
    for _ in range(10):
        session_int8.run(None, {'input': dummy})  # Warmup

    for _ in range(200):
        t0 = time.perf_counter()
        session_int8.run(None, {'input': dummy})
        t_records.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(t_records, 50))
    p95 = float(np.percentile(t_records, 95))
    print(f"INT8 ONNX Single-thread CPU Latency: p50={p50:.3f} ms | p95={p95:.3f} ms", flush=True)

    export_summary = {
        "artifact_prefix": artifact_prefix,
        "classes_count": len(labels),
        "labels": labels,
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": int8_size,
        "compression_ratio": round(fp32_size / int8_size, 2),
        "test_accuracy_fp32_onnx": round(acc_fp32, 4),
        "test_accuracy_int8_onnx": round(acc_int8, 4),
        "prediction_parity": round(parity_rate, 4),
        "cpu_latency_p50_ms": round(p50, 3),
        "cpu_latency_p95_ms": round(p95, 3)
    }
    (models_dir / 'export_summary.json').write_text(json.dumps(export_summary, indent=2), encoding='utf-8')

    return export_summary
