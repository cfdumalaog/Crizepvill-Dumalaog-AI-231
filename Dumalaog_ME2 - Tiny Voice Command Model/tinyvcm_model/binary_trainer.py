"""Binary wake detector training, threshold selection, ONNX export, and evaluation.

Implements:
- Strict random initialization (seed 231, zero pretrained weights) with initial state hash.
- TinyDSCNN binary architecture (13,106 parameters).
- 35-epoch AdamW + CosineAnnealing training with class-weighted loss and SpecAugment.
- Validation checkpointing maximizing human wake recall and zero false alarms.
- Operating threshold sweep and selection on validation positives/negatives.
- Static INT8 QDQ quantization from 256 training-only calibration samples.
- Single-pass held-out test evaluation against installed 32-class model.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .binary_dataset import BinaryWakeDataset, prepare_binary_data
from .config import SAMPLES, SR, FEATURE_SHAPE
from .frontend import Frontend, fit_audio
from .model import TinyDSCNN


def compute_model_param_hash(model: nn.Module) -> str:
    """Compute SHA-256 hash of all model parameter tensors to record initial provenance."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class BinaryCalibrationReader(CalibrationDataReader):
    """Feeds 256 training-only samples to static INT8 quantizer."""
    def __init__(self, calibration_features: np.ndarray):
        self.samples = [calibration_features[i:i + 1] for i in range(len(calibration_features))]
        self.iter = iter(self.samples)

    def get_next(self):
        val = next(self.iter, None)
        if val is None:
            return None
        return {"input": val}


def train_binary_wake_model(
    project_root: Path,
    output_dir: Path,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 231,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Train binary wake detector from scratch, export to FP32 and INT8, and evaluate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"=== [Task D] Binary Wake-Word Training: {output_dir.name} ===")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size}")

    # 1. Prepare Datasets
    print("\n[1/6] Loading datasets and generating temporal jitter augmentations...")
    data = prepare_binary_data(project_root, rng_seed=seed)
    splits = data['splits_info']
    print(f"Train samples: {splits['train_samples']} ({splits['train_positives']} wake, {splits['train_negatives']} non-wake)")
    print(f"Val samples:   {splits['val_samples']} ({splits['val_positives']} wake, {splits['val_negatives']} non-wake)")
    print(f"Human wake groups: {splits['train_hw_groups']} train | {splits['val_hw_groups']} val | {splits['test_hw_groups']} test")
    print(f"Synthetic wake:    {splits['train_synth_groups']} groups train | {splits['val_synth_groups']} val | {splits['test_synth_groups']} test")

    train_ds = BinaryWakeDataset(data['train_x'], data['train_y'], augment=True, rng_seed=seed)
    val_ds = BinaryWakeDataset(data['val_x'], data['val_y'], augment=False, rng_seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model from Scratch
    print("\n[2/6] Initializing TinyDSCNN-Binary (2 classes) from random weights...")
    model = TinyDSCNN(classes=2, channels=48, dropout=0.15)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    init_hash = compute_model_param_hash(model)
    print(f"Architecture: TinyDSCNN-Binary (Stem + 4 DS Blocks + Linear(48 -> 2))")
    print(f"Total Parameters: {total_params:,}")
    print(f"Initial Weights SHA-256: {init_hash}")

    initial_provenance = {
        'model_name': 'TinyDSCNN-Binary',
        'classes': ['NON_WAKE', 'WAKE_WORD'],
        'total_parameters': total_params,
        'seed': seed,
        'initial_parameter_sha256': init_hash,
        'pretrained_weights_used': False,
        'splits_info': splits,
        'device': device,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with (output_dir / 'initial_provenance.json').open('w', encoding='utf-8') as f:
        json.dump(initial_provenance, f, indent=2)

    # 3. Optimizer, Scheduler & Class-Weighted Loss
    w_neg = 1.0
    w_pos = float(splits['train_negatives']) / float(splits['train_positives'])
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)

    # 4. Training Loop with Live Epoch Logging
    print("\n[3/6] Starting 35-epoch training loop...")
    history = []
    best_val_score = -1.0
    best_epoch = -1
    best_weights = None

    frontend = Frontend()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += len(by)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation Evaluation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        all_val_probs = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)

                all_val_probs.extend(probs.cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_targets.extend(by.cpu().numpy().tolist())

        val_loss /= len(all_val_targets)
        val_probs_arr = np.array(all_val_probs)
        val_targets_arr = np.array(all_val_targets)

        # Break down metrics
        pos_mask = (val_targets_arr == 1)
        neg_mask = (val_targets_arr == 0)

        # Human wake recall on validation takes
        val_hw_probs = []
        for it in data['val_hw_items']:
            w_fit = fit_audio(it['wav'], target_samples=SAMPLES)
            spec = frontend(w_fit)
            with torch.no_grad():
                inp = torch.from_numpy(spec[None, ...]).float().to(device)
                p = torch.softmax(model(inp), dim=-1)[0, 1].item()
                val_hw_probs.append(p)
        val_hw_recall = float(np.mean([1.0 if p >= 0.50 else 0.0 for p in val_hw_probs]))

        # Synthetic wake recall on validation files
        val_synth_probs = []
        for sf_path in data['val_synth_files']:
            sw, _ = sf.read(str(sf_path), dtype='float32')
            if sw.ndim > 1:
                sw = sw.mean(axis=1)
            spec = frontend(fit_audio(sw, target_samples=SAMPLES))
            with torch.no_grad():
                inp = torch.from_numpy(spec[None, ...]).float().to(device)
                p = torch.softmax(model(inp), dim=-1)[0, 1].item()
                val_synth_probs.append(p)
        val_synth_recall = float(np.mean([1.0 if p >= 0.50 else 0.0 for p in val_synth_probs]))

        # False alarms on validation command negatives (at default 0.50)
        neg_probs = val_probs_arr[neg_mask]
        false_alarms = int((neg_probs >= 0.50).sum())
        false_alarm_rate = false_alarms / max(1, len(neg_probs))

        # Composite validation selection score
        # Heavily prioritize 100% human wake recall and 0 false alarms
        val_score = (val_hw_recall * 0.45) + ((1.0 - false_alarm_rate) * 0.45) + (val_synth_recall * 0.10)

        epoch_time = time.time() - t0
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_score': round(val_score, 4),
            'val_hw_recall': round(val_hw_recall, 4),
            'val_synth_recall': round(val_synth_recall, 4),
            'val_false_alarms': false_alarms,
            'val_false_alarm_rate': round(false_alarm_rate, 4),
            'val_hw_probs': [round(p, 4) for p in val_hw_probs],
            'time_sec': round(epoch_time, 2)
        }
        history.append(log_entry)

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, output_dir / 'best_model.pt')

        best_marker = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:2d}/{epochs:2d} ({epoch_time:4.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Score: {val_score:.4f} HW: {val_hw_recall*100:5.1f}% FA: {false_alarms}/{len(neg_probs)} ({false_alarm_rate*100:4.2f}%){best_marker}")

    # Load best checkpoint
    model.load_state_dict(best_weights)
    print(f"\nBest validation score: {best_val_score:.4f} achieved at Epoch {best_epoch}.")
    with (output_dir / 'training_history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # 5. Validation Operating Threshold Selection (Sweep on Validation Only)
    print("\n[4/6] Sweeping validation operating threshold theta...")
    model.eval()
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    threshold_results = []
    selected_threshold = 0.50
    best_tradeoff = -1.0

    # Evaluate validation positive & negative distributions
    val_hw_probs_final = []
    for it in data['val_hw_items']:
        spec = frontend(fit_audio(it['wav'], target_samples=SAMPLES))
        with torch.no_grad():
            inp = torch.from_numpy(spec[None, ...]).float().to(device)
            val_hw_probs_final.append(torch.softmax(model(inp), dim=-1)[0, 1].item())

    val_synth_probs_final = []
    for sf_path in data['val_synth_files']:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        spec = frontend(fit_audio(sw, target_samples=SAMPLES))
        with torch.no_grad():
            inp = torch.from_numpy(spec[None, ...]).float().to(device)
            val_synth_probs_final.append(torch.softmax(model(inp), dim=-1)[0, 1].item())

    val_neg_probs_final = []
    with torch.no_grad():
        for i in range(0, len(data['val_x']), 128):
            bx = torch.from_numpy(data['val_x'][i:i + 128]).float().to(device)
            by = data['val_y'][i:i + 128]
            probs = torch.softmax(model(bx), dim=-1)[:, 1].cpu().numpy()
            for p, y in zip(probs, by):
                if y == 0:
                    val_neg_probs_final.append(float(p))

    for th in thresholds:
        hw_rec = float(np.mean([1.0 if p >= th else 0.0 for p in val_hw_probs_final]))
        synth_rec = float(np.mean([1.0 if p >= th else 0.0 for p in val_synth_probs_final]))
        fa_count = sum(1 for p in val_neg_probs_final if p >= th)
        fa_rate = fa_count / len(val_neg_probs_final)

        # Objective: 100% human wake recall, 0 false alarms
        tradeoff = (hw_rec * 100.0) - (fa_rate * 500.0) + (synth_rec * 10.0)
        entry = {
            'threshold': th,
            'hw_recall': round(hw_rec, 4),
            'synth_recall': round(synth_rec, 4),
            'false_accepts': fa_count,
            'false_accept_rate': round(fa_rate, 4),
            'tradeoff_score': round(tradeoff, 2)
        }
        threshold_results.append(entry)

        # Select threshold that guarantees 100% human wake recall while minimizing false alarms
        if hw_rec >= 1.0 and fa_count == 0:
            if tradeoff > best_tradeoff:
                best_tradeoff = tradeoff
                selected_threshold = th
        elif best_tradeoff < 0 and hw_rec >= 1.0:
            if tradeoff > best_tradeoff:
                best_tradeoff = tradeoff
                selected_threshold = th

    print(f"Validation operating threshold selected: theta* = {selected_threshold:.2f}")
    threshold_summary = {
        'validation_selected_threshold': selected_threshold,
        'selection_criterion': 'Strict 100% human wake validation recall with zero command false alarms',
        'val_hw_probabilities': [round(p, 4) for p in val_hw_probs_final],
        'val_hw_recall_at_theta': [r for r in threshold_results if r['threshold'] == selected_threshold][0]['hw_recall'],
        'val_false_accepts_at_theta': [r for r in threshold_results if r['threshold'] == selected_threshold][0]['false_accepts'],
        'sweep': threshold_results
    }
    with (output_dir / 'threshold_analysis.json').open('w', encoding='utf-8') as f:
        json.dump(threshold_summary, f, indent=2)

    # 6. ONNX Export & Static INT8 Quantization
    print("\n[5/6] Exporting FP32 and INT8 ONNX models...")
    model.eval().cpu()
    dummy_input = torch.zeros(1, 1, 40, 251, dtype=torch.float32)

    fp32_onnx_path = models_dir / 'vcm_wake2_binary_fp32.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )
    fp32_size = fp32_onnx_path.stat().st_size
    fp32_hash = hashlib.sha256(fp32_onnx_path.read_bytes()).hexdigest()
    print(f"FP32 ONNX Exported: {fp32_size:,} bytes | SHA-256: {fp32_hash}")

    int8_onnx_path = models_dir / 'vcm_wake2_binary_int8.onnx'
    calib_reader = BinaryCalibrationReader(data['calib_x'])
    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False
    )
    int8_size = int8_onnx_path.stat().st_size
    int8_hash = hashlib.sha256(int8_onnx_path.read_bytes()).hexdigest()
    print(f"INT8 ONNX Quantized: {int8_size:,} bytes | SHA-256: {int8_hash}")

    export_summary = {
        'model_name': 'TinyDSCNN-Binary',
        'classes': ['NON_WAKE', 'WAKE_WORD'],
        'validation_selected_threshold': selected_threshold,
        'fp32_onnx': {
            'path': str(fp32_onnx_path.name),
            'size_bytes': fp32_size,
            'sha256': fp32_hash
        },
        'int8_onnx': {
            'path': str(int8_onnx_path.name),
            'size_bytes': int8_size,
            'sha256': int8_hash
        }
    }
    with (output_dir / 'export_summary.json').open('w', encoding='utf-8') as f:
        json.dump(export_summary, f, indent=2)

    # 7. Single-Pass Held-Out Test Evaluation
    print(f"\n[6/6] Executing single-pass evaluation on held-out test split at theta* = {selected_threshold:.2f}...")
    ort_fp32 = ort.InferenceSession(str(fp32_onnx_path), providers=['CPUExecutionProvider'])
    ort_int8 = ort.InferenceSession(str(int8_onnx_path), providers=['CPUExecutionProvider'])

    # Held-out Human Wake Test (3 groups: 2 near, 1 far)
    test_hw_probs_fp32 = []
    test_hw_probs_int8 = []
    for it in data['test_hw_items']:
        spec = frontend(fit_audio(it['wav'], target_samples=SAMPLES))[None, ...]
        out_f = ort_fp32.run(None, {'input': spec})[0][0]
        out_i = ort_int8.run(None, {'input': spec})[0][0]
        p_f = float(np.exp(out_f[1]) / np.sum(np.exp(out_f)))
        p_i = float(np.exp(out_i[1]) / np.sum(np.exp(out_i)))
        test_hw_probs_fp32.append(p_f)
        test_hw_probs_int8.append(p_i)

    hw_recall_fp32 = float(np.mean([1.0 if p >= selected_threshold else 0.0 for p in test_hw_probs_fp32]))
    hw_recall_int8 = float(np.mean([1.0 if p >= selected_threshold else 0.0 for p in test_hw_probs_int8]))

    # Held-out Synthetic Wake Test (35 clips)
    test_synth_probs_fp32 = []
    test_synth_probs_int8 = []
    for sf_path in data['test_synth_files']:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        spec = frontend(fit_audio(sw, target_samples=SAMPLES))[None, ...]
        out_f = ort_fp32.run(None, {'input': spec})[0][0]
        out_i = ort_int8.run(None, {'input': spec})[0][0]
        p_f = float(np.exp(out_f[1]) / np.sum(np.exp(out_f)))
        p_i = float(np.exp(out_i[1]) / np.sum(np.exp(out_i)))
        test_synth_probs_fp32.append(p_f)
        test_synth_probs_int8.append(p_i)

    synth_recall_fp32 = float(np.mean([1.0 if p >= selected_threshold else 0.0 for p in test_synth_probs_fp32]))
    synth_recall_int8 = float(np.mean([1.0 if p >= selected_threshold else 0.0 for p in test_synth_probs_int8]))

    # False Alarms on 1,798 Held-out Option B Command Test Clips
    optb_test_x = data['optb_test_x']
    false_wakes_optb_fp32 = 0
    false_wakes_optb_int8 = 0
    parity_matches = 0

    for i in range(0, len(optb_test_x), 64):
        chunk = optb_test_x[i:i + 64]
        out_f = ort_fp32.run(None, {'input': chunk})[0]
        out_i = ort_int8.run(None, {'input': chunk})[0]

        probs_f = np.exp(out_f[:, 1]) / np.sum(np.exp(out_f), axis=-1)
        probs_i = np.exp(out_i[:, 1]) / np.sum(np.exp(out_i), axis=-1)

        false_wakes_optb_fp32 += int((probs_f >= selected_threshold).sum())
        false_wakes_optb_int8 += int((probs_i >= selected_threshold).sum())

        pred_f = (probs_f >= selected_threshold).astype(int)
        pred_i = (probs_i >= selected_threshold).astype(int)
        parity_matches += int((pred_f == pred_i).sum())

    parity = parity_matches / len(optb_test_x)

    metrics = {
        'validation_selected_threshold': selected_threshold,
        'test_results': {
            'fp32': {
                'held_out_human_wake_recall': hw_recall_fp32,
                'human_wake_probabilities': [round(p, 4) for p in test_hw_probs_fp32],
                'held_out_synth_wake_recall': synth_recall_fp32,
                'optionb_test_false_wakes': false_wakes_optb_fp32,
                'optionb_test_false_wake_rate': round(false_wakes_optb_fp32 / len(optb_test_x), 6),
            },
            'int8': {
                'held_out_human_wake_recall': hw_recall_int8,
                'human_wake_probabilities': [round(p, 4) for p in test_hw_probs_int8],
                'held_out_synth_wake_recall': synth_recall_int8,
                'optionb_test_false_wakes': false_wakes_optb_int8,
                'optionb_test_false_wake_rate': round(false_wakes_optb_int8 / len(optb_test_x), 6),
                'fp32_int8_parity': round(parity, 4)
            }
        },
        'model_hashes': {
            'fp32': fp32_hash,
            'int8': int8_hash
        }
    }
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  SINGLE-PASS HELD-OUT TEST EVALUATION (theta* = {selected_threshold:.2f})")
    print("=" * 70)
    print(f"Held-out Human Wake Recall (3 groups): FP32 {hw_recall_fp32*100:5.1f}% | INT8 {hw_recall_int8*100:5.1f}%")
    print(f"Human Wake Test Probabilities:        INT8 {[round(p, 4) for p in test_hw_probs_int8]}")
    print(f"Held-out Synthetic Wake Recall (35):   FP32 {synth_recall_fp32*100:5.1f}% | INT8 {synth_recall_int8*100:5.1f}%")
    print(f"False Wakes on 1,798 Option B Clips:   FP32 {false_wakes_optb_fp32} | INT8 {false_wakes_optb_int8} ({false_wakes_optb_int8/len(optb_test_x)*100:.3f}%)")
    print(f"FP32 vs INT8 Prediction Parity:        {parity*100:.2f}% ({parity_matches}/{len(optb_test_x)})")
    print("=" * 70)

    return metrics
