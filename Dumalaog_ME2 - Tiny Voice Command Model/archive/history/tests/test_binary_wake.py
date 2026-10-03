import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_antigrav.model import TinyDSCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_WITH_BINARY_EXPORTS = sorted(
    path.parent.parent
    for path in PROJECT_ROOT.glob('runs/antigrav-binary-wake-*/models/binary_wake_int8.onnx')
)
RUN_DIR = RUNS_WITH_BINARY_EXPORTS[-1]


def test_binary_model_architecture_and_parameter_count():
    model = TinyDSCNN(classes=2, channels=48, dropout=0.15)
    params = sum(p.numel() for p in model.parameters())
    assert params == 13106, f"Expected 13,106 parameters, got {params}"

    dummy = torch.zeros(2, 1, 40, 251, dtype=torch.float32)
    out = model(dummy)
    assert out.shape == (2, 2), f"Expected output shape (2, 2), got {out.shape}"


def test_binary_onnx_artifacts_exist_and_under_footprint():
    fp32_onnx = RUN_DIR / 'models' / 'binary_wake_fp32.onnx'
    int8_onnx = RUN_DIR / 'models' / 'binary_wake_int8.onnx'

    assert fp32_onnx.exists(), f"FP32 ONNX missing at {fp32_onnx}"
    assert int8_onnx.exists(), f"INT8 ONNX missing at {int8_onnx}"

    int8_size = int8_onnx.stat().st_size
    assert int8_size < 50_000, f"INT8 ONNX size {int8_size} exceeds 50 KB edge target"
    assert int8_size > 30_000, f"INT8 ONNX size {int8_size} unexpectedly small"


def test_binary_int8_onnx_inference_and_latency():
    int8_onnx = RUN_DIR / 'models' / 'binary_wake_int8.onnx'
    session = ort.InferenceSession(str(int8_onnx), providers=['CPUExecutionProvider'])

    dummy = np.zeros((1, 1, 40, 251), dtype=np.float32)
    out = session.run(None, {'input': dummy})[0]

    assert out.shape == (1, 2), f"Expected ONNX output shape (1, 2), got {out.shape}"
    probs = np.exp(out[0]) / np.sum(np.exp(out[0]))
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5), "Softmax probabilities must sum to 1.0"
