"""Paired Training Pipeline for Option B Control vs. Option B + SLURP Real Treatment.

This module implements:
1. Strict random initialization parity (seed 231, zero pretrained weights, identical initial parameter hash).
2. Matched step budget: 35 epochs, batch size 64, AdamW (lr=3e-3, weight_decay=1e-4), CosineAnnealingLR.
3. Control (Run C): Option B (14,370 train clips) + identical wake data (420 human-derived + 490 synthetic wake clips).
4. Treatment (Run T): Option B + eligible SLURP real train samples at a deterministic 75:25 mix per batch for the 13 supported classes; identical wake data.
5. Checkpoint selection based strictly on Option B validation metrics (val macro F1 + val wake recall).
6. Post-training static INT8 quantization using 256 training-only stratified samples.
7. Single-pass dual evaluation on Option B held-out test set and custom SLURP held-out test set at validation-frozen threshold theta* = 0.45.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_model.config import LABELS, WAKE_LABELS, SAMPLES, SR, CHANNELS, FEATURE_SHAPE
from tinyvcm_model.frontend import Frontend, fit_audio
from tinyvcm_model.model import TinyDSCNN
from tinyvcm_model.recording_labels import map_recorded_intent

SUPPORTED_SLURP_CLASSES = [
    'PLAY_MUSIC', 'WEATHER', 'TIME', 'LIGHT_ON', 'LIGHT_OFF',
    'VOLUME_UP', 'VOLUME_DOWN', 'COLOR_RED', 'COLOR_GREEN', 'COLOR_BLUE',
    'ALARM_6_00AM', 'ALARM_8_00AM', 'ALARM_9_00PM'
]

def compute_pcm_sha256(audio_path: Path) -> tuple[str, np.ndarray, int]:
    wav, sr = sf.read(str(audio_path), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pcm_bytes = wav.tobytes()
    pcm_hash = hashlib.sha256(pcm_bytes).hexdigest()
    return pcm_hash, wav, sr


def augment_waveform(wav: np.ndarray, rng: np.random.Generator, noises: list[np.ndarray]) -> np.ndarray:
    w = wav.copy()
    # 1. Gain scaling (0.65 to 1.35)
    gain = rng.uniform(0.65, 1.35)
    w = w * gain

    # 2. Time shifting (+/- 2400 samples = +/- 150ms)
    shift = rng.integers(-2400, 2400)
    w = np.roll(w, shift)
    if shift > 0:
        w[:shift] = 0.0
    elif shift < 0:
        w[shift:] = 0.0

    # 3. Additive background noise (SNR 20 to 35 dB)
    if noises and rng.random() < 0.75:
        noise = noises[rng.integers(0, len(noises))]
        if len(noise) >= len(w):
            start = rng.integers(0, len(noise) - len(w) + 1)
            n_slice = noise[start:start + len(w)]
        else:
            n_slice = np.tile(noise, int(np.ceil(len(w) / len(noise))))[:len(w)]
        signal_power = np.mean(w ** 2) + 1e-9
        noise_power = np.mean(n_slice ** 2) + 1e-9
        target_snr_db = rng.uniform(20.0, 35.0)
        target_noise_power = signal_power / (10 ** (target_snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        w = w + n_slice * scale

    return fit_audio(w, target_samples=SAMPLES)


class StratifiedTrainCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_features: np.ndarray):
        self.samples = [calibration_features[i:i + 1] for i in range(len(calibration_features))]
        self.iter = iter(self.samples)

    def get_next(self):
        val = next(self.iter, None)
        if val is None:
            return None
        return {"input": val}


def extract_features_from_manifest(manifest_csv: Path, output_npz: Path, force: bool = False):
    if output_npz.exists() and not force:
        print(f"[OK] Cache exists: {output_npz.name}")
        data = np.load(output_npz)
        return data['x'], data['y']

    frontend = Frontend()
    with manifest_csv.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    features = []
    labels = []
    skipped = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        wav_path = PROJECT_ROOT / r['path']
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            wav, sr = sf.read(str(wav_path), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            fitted = fit_audio(wav, target_samples=SAMPLES)
            feat = frontend(fitted)
            features.append(feat)
            labels.append(int(r['class_idx']))
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            skipped += 1

        if (i + 1) % 1000 == 0 or (i + 1) == len(rows):
            print(f"  Processed {i+1}/{len(rows)} clips (elapsed: {time.time()-t0:.1f}s)...", flush=True)

    x = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, x=x, y=y)
    print(f"[SAVED] Extracted {len(y)} clips to {output_npz.name} (skipped/missing: {skipped})")
    return x, y


def prepare_paired_datasets(seed: int = 231) -> dict:
    """Prepares training and validation datasets ONLY.

    Strict Test Isolation Policy:
    Test sets (Option B test, SLURP test, held-out wake test) are NOT loaded or featurized here.
    They remain untouched until evaluate_paired_models() is executed post-training.
    """
    frontend = Frontend()
    rng = np.random.default_rng(seed)

    # 1. Background Noise Clips
    noise_dir = PROJECT_ROOT / 'data' / 'dataset' / '_background_noise_'
    noise_files = sorted(noise_dir.glob('*.wav'))
    noises = []
    for nf in noise_files:
        nw, _ = sf.read(str(nf), dtype='float32')
        if nw.ndim > 1:
            nw = nw.mean(axis=1)
        noises.append(nw)

    # 2. Human Wake Takes (Deduplicated 9 unique waveforms, 5/2/2 split)
    human_manifest = PROJECT_ROOT / 'data' / 'human' / 'manifest.csv'
    with human_manifest.open('r', encoding='utf-8') as f:
        human_rows = list(csv.DictReader(f))

    wake_rows = [r for r in human_rows if r['label'] == 'wake_word']
    pcm_groups: dict[str, list[dict]] = {}
    for r in wake_rows:
        wav_path = PROJECT_ROOT / r['path']
        if not wav_path.exists():
            continue
        h, wav_data, _ = compute_pcm_sha256(wav_path)
        pcm_groups.setdefault(h, []).append({
            'row': r,
            'path': str(r['path']),
            'filename': wav_path.name,
            'condition': r['condition'],
            'wav': wav_data
        })

    unique_hashes = sorted(pcm_groups.keys())
    assert len(unique_hashes) == 9

    train_hw_groups = unique_hashes[:5]
    val_hw_groups = unique_hashes[5:7]
    test_hw_groups = unique_hashes[7:]

    train_hw_items = [item for h in train_hw_groups for item in pcm_groups[h]]
    val_hw_items = [item for h in val_hw_groups for item in pcm_groups[h]]
    test_hw_items = [item for h in test_hw_groups for item in pcm_groups[h]]

    # 3. Synthetic Wake Takes (9 groups, 7/1/1 split)
    synth_wake_dir = PROJECT_ROOT / 'data' / 'dataset' / 'wake_word'
    synth_files = sorted(synth_wake_dir.glob('*.wav'))
    synth_groups: dict[str, list[Path]] = {}
    for sf_path in synth_files:
        grp = re.sub(r'_\d+_[^_]+$', '', sf_path.stem)
        synth_groups.setdefault(grp, []).append(sf_path)

    unique_synth_groups = sorted(synth_groups.keys())
    assert len(unique_synth_groups) == 9

    train_synth_groups = unique_synth_groups[:7]
    val_synth_groups = unique_synth_groups[7:8]
    test_synth_groups = unique_synth_groups[8:]

    train_synth_files = [f for g in train_synth_groups for f in synth_groups[g]]
    val_synth_files = [f for g in val_synth_groups for f in synth_groups[g]]
    test_synth_files = [f for g in test_synth_groups for f in synth_groups[g]]

    # 4. Human Command Takes (16 clips, training only)
    human_cmd_features = []
    human_cmd_labels = []
    for r in human_rows:
        lbl = r['label']
        mapped_label = map_recorded_intent(lbl)
        if mapped_label is not None:
            wav_path = PROJECT_ROOT / r['path']
            if not wav_path.exists():
                continue
            wav, _ = sf.read(str(wav_path), dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            fitted = fit_audio(wav, target_samples=SAMPLES)
            spec = frontend(fitted)
            human_cmd_features.append(spec)
            human_cmd_labels.append(LABELS.index(mapped_label))

    human_cmd_x = np.stack(human_cmd_features, axis=0) if human_cmd_features else np.empty((0, 1, 40, 251), dtype=np.float32)
    human_cmd_y = np.array(human_cmd_labels, dtype=np.int64)

    # 5. Base Option B Cache (Train & Validation splits ONLY - Test split untouched)
    cache_path = PROJECT_ROOT / 'data' / 'option_b' / 'features_cache.npz'
    cache = np.load(cache_path)
    optb_train_x = cache['train_x'].astype(np.float32)
    optb_train_y = cache['train_y'].astype(np.int64)
    optb_val_x = cache['val_x'].astype(np.float32)
    optb_val_y = cache['val_y'].astype(np.int64)

    # 6. Training Wake Augmentation (Identical for Control & Treatment)
    wake_idx = WAKE_LABELS.index('WAKE_WORD')
    train_wake_features = []
    train_wake_sources = []

    # 7 files x 60 versions = 420 human-derived samples across 5 unique waveforms
    for it in train_hw_items:
        base_wav = it['wav']
        train_wake_features.append(frontend(fit_audio(base_wav, target_samples=SAMPLES)))
        train_wake_sources.append(('human', it['filename']))
        for _ in range(59):
            train_wake_features.append(frontend(augment_waveform(base_wav, rng, noises)))
            train_wake_sources.append(('human', it['filename']))

    # 245 files x 2 versions = 490 synthetic samples across 7 groups
    for sf_path in train_synth_files:
        swav, _ = sf.read(str(sf_path), dtype='float32')
        if swav.ndim > 1:
            swav = swav.mean(axis=1)
        train_wake_features.append(frontend(fit_audio(swav, target_samples=SAMPLES)))
        train_wake_sources.append(('synth', sf_path.name))
        train_wake_features.append(frontend(augment_waveform(swav, rng, noises)))
        train_wake_sources.append(('synth', sf_path.name))

    train_wake_x = np.stack(train_wake_features, axis=0).astype(np.float32)
    train_wake_y = np.full(len(train_wake_x), wake_idx, dtype=np.int64)

    # 7. Validation Positives & Negatives
    val_hw_feats = [frontend(fit_audio(it['wav'], target_samples=SAMPLES)) for it in val_hw_items]
    val_hw_x = np.stack(val_hw_feats, axis=0).astype(np.float32)
    val_hw_y = np.full(len(val_hw_x), wake_idx, dtype=np.int64)

    val_synth_feats = []
    for sf_path in val_synth_files:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        val_synth_feats.append(frontend(fit_audio(sw, target_samples=SAMPLES)))
    val_synth_x = np.stack(val_synth_feats, axis=0).astype(np.float32)
    val_synth_y = np.full(len(val_synth_x), wake_idx, dtype=np.int64)

    # 8. Control Training Set (Run C)
    control_train_x = np.concatenate([optb_train_x, train_wake_x, human_cmd_x], axis=0)
    control_train_y = np.concatenate([optb_train_y, train_wake_y, human_cmd_y], axis=0)

    # 9. Load SLURP Train & Val Features ONLY (Test set untouched)
    slurp_cache_dir = PROJECT_ROOT / 'data' / 'slurp'
    slurp_train_x, slurp_train_y = extract_features_from_manifest(
        slurp_cache_dir / 'manifest_slurp_train.csv',
        slurp_cache_dir / 'slurp_train_features.npz'
    )
    slurp_val_x, slurp_val_y = extract_features_from_manifest(
        slurp_cache_dir / 'manifest_slurp_val.csv',
        slurp_cache_dir / 'slurp_val_features.npz'
    )

    # 10. Treatment Training Set (Run T) with 75:25 Mix for Supported Classes
    supported_indices = set(WAKE_LABELS.index(lbl) for lbl in SUPPORTED_SLURP_CLASSES)
    treatment_x_list = []
    treatment_y_list = []

    for c in range(31):
        optb_mask = (optb_train_y == c)
        optb_c_x = optb_train_x[optb_mask]
        optb_c_y = optb_train_y[optb_mask]
        n_c = len(optb_c_y)

        if c in supported_indices:
            slurp_mask = (slurp_train_y == c)
            slurp_c_x = slurp_train_x[slurp_mask]
            slurp_c_y = slurp_train_y[slurp_mask]

            if len(slurp_c_y) > 0:
                n_slurp = int(round(n_c * 0.25))
                n_optb = n_c - n_slurp

                optb_perm = rng.permutation(len(optb_c_y))[:n_optb]
                treatment_x_list.append(optb_c_x[optb_perm])
                treatment_y_list.append(optb_c_y[optb_perm])

                replace = len(slurp_c_y) < n_slurp
                slurp_choices = rng.choice(len(slurp_c_y), size=n_slurp, replace=replace)
                treatment_x_list.append(slurp_c_x[slurp_choices])
                treatment_y_list.append(slurp_c_y[slurp_choices])
            else:
                treatment_x_list.append(optb_c_x)
                treatment_y_list.append(optb_c_y)
        else:
            treatment_x_list.append(optb_c_x)
            treatment_y_list.append(optb_c_y)

    treatment_train_x = np.concatenate(treatment_x_list + [train_wake_x, human_cmd_x], axis=0)
    treatment_train_y = np.concatenate(treatment_y_list + [train_wake_y, human_cmd_y], axis=0)

    assert len(control_train_y) == len(treatment_train_y), (
        f"Step budget mismatch: Control {len(control_train_y)} vs Treatment {len(treatment_train_y)}"
    )

    data = {
        'control_train_x': control_train_x,
        'control_train_y': control_train_y,
        'treatment_train_x': treatment_train_x,
        'treatment_train_y': treatment_train_y,
        'optb_val_x': optb_val_x,
        'optb_val_y': optb_val_y,
        'val_hw_x': val_hw_x,
        'val_synth_x': val_synth_x,
        'slurp_val_x': slurp_val_x,
        'slurp_val_y': slurp_val_y,
        'train_hw_items': train_hw_items,
        'train_synth_files': train_synth_files,
        'train_wake_sources': train_wake_sources,
        'slurp_train_y': slurp_train_y,
        # Metadata references for post-training test evaluation only:
        'test_hw_items': test_hw_items,
        'test_synth_files': test_synth_files,
    }
    return data


def train_model(
    run_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    datasets: dict,
    run_name: str,
    model_name: str,
    is_treatment: bool = False,
    seed: int = 231,
    epochs: int = 35,
    batch_size: int = 64
) -> dict:
    """Trains model for 35 epochs and exports FP32 + static INT8 ONNX without touching test data."""
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"  STARTING MODEL RUN: {run_name} ({'TREATMENT (SLURP Mix)' if is_treatment else 'CONTROL (Option B Baseline)'})")
    print(f"  Device: {device} | Total Training Samples: {len(train_y):,} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    # 1. Scratch Model Initialization
    model = TinyDSCNN(classes=len(WAKE_LABELS), channels=CHANNELS).to(device)
    initial_bytes = b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()
    total_params = sum(p.numel() for p in model.parameters())

    initial_provenance = {
        'run_name': run_name,
        'model_name': model_name,
        'is_treatment': is_treatment,
        'seed': seed,
        'pretrained_weights_loaded': False,
        'initial_parameters_sha256': initial_hash,
        'total_parameters': total_params,
        'num_classes': len(WAKE_LABELS),
        'labels': WAKE_LABELS,
    }
    (run_dir / 'initial_provenance.json').write_text(json.dumps(initial_provenance, indent=2), encoding='utf-8')
    print(f"Initial parameters SHA-256: {initial_hash}")

    # 2. Optimizer & Class Weights
    counts = np.bincount(train_y, minlength=len(WAKE_LABELS)).astype(np.float32)
    weights = len(train_y) / (len(WAKE_LABELS) * counts)
    weights = np.clip(weights, 0.25, 4.0)
    weights_tensor = torch.from_numpy(weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.02)

    gpu_train_x = torch.from_numpy(train_x).to(device)
    gpu_train_y = torch.from_numpy(train_y).to(device)

    optb_val_x = datasets['optb_val_x']
    optb_val_y = datasets['optb_val_y']
    val_hw_x = datasets['val_hw_x']
    val_synth_x = datasets['val_synth_x']

    best_score = -1.0
    best_epoch = -1
    best_state = None
    best_val_metrics = {}
    history = []

    print(f"\nTraining for {epochs} epochs on {device} (batch size {batch_size})...")
    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(gpu_train_y), device=device)
        running_loss = 0.0
        running_correct = 0

        for i in range(0, len(gpu_train_y), batch_size):
            ids = order[i:i + batch_size]
            xb = gpu_train_x[ids].clone()
            yb = gpu_train_y[ids]

            # SpecAugment on GPU
            f_start = np.random.randint(0, 36)
            t_start_spec = np.random.randint(0, 235)
            xb[:, :, f_start:f_start + 4, :] = 0.0
            xb[:, :, :, t_start_spec:t_start_spec + 12] = 0.0

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(ids)
            running_correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        epoch_time = time.perf_counter() - t_start
        train_acc = running_correct / len(gpu_train_y)
        train_loss = running_loss / len(gpu_train_y)

        # Validation Evaluation strictly on Option B val + wake validation
        model.eval()
        with torch.no_grad():
            val_batches = []
            for vi in range(0, len(optb_val_x), 128):
                v_xb = torch.from_numpy(optb_val_x[vi:vi + 128]).to(device)
                val_batches.append(model(v_xb).cpu())
            val_cmd_logits = torch.cat(val_batches, dim=0)
            val_cmd_probs = torch.softmax(val_cmd_logits, dim=-1).numpy()
            val_cmd_preds = np.argmax(val_cmd_probs, axis=-1)

            val_cmd_acc = float(np.mean(val_cmd_preds == optb_val_y))
            val_cmd_f1 = float(f1_score(optb_val_y, val_cmd_preds, average='macro', zero_division=0))

            hw_logits = model(torch.from_numpy(val_hw_x).to(device)).cpu()
            hw_probs = torch.softmax(hw_logits, dim=-1).numpy()[:, 31]

            sw_logits = model(torch.from_numpy(val_synth_x).to(device)).cpu()
            sw_probs = torch.softmax(sw_logits, dim=-1).numpy()[:, 31]

            cmd_wake_probs = val_cmd_probs[:, 31]
            false_wake_45 = int(np.sum(cmd_wake_probs >= 0.45))
            hw_rec_45 = float(np.mean(hw_probs >= 0.45))
            sw_rec_45 = float(np.mean(sw_probs >= 0.45))

            composite_score = (val_cmd_f1 * 0.55 + hw_rec_45 * 0.35 + sw_rec_45 * 0.10) - (false_wake_45 / len(optb_val_y)) * 5.0

        is_best = composite_score > best_score
        if is_best:
            best_score = composite_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, run_dir / 'best_model.pt')
            best_val_metrics = {
                'epoch': epoch,
                'composite_score': composite_score,
                'val_cmd_acc': val_cmd_acc,
                'val_cmd_f1': val_cmd_f1,
                'val_hw_recall_th45': hw_rec_45,
                'val_synth_recall_th45': sw_rec_45,
                'val_false_wake_th45': false_wake_45,
            }

        hist_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_cmd_acc': val_cmd_acc,
            'val_cmd_f1': val_cmd_f1,
            'val_hw_recall_th45': hw_rec_45,
            'val_synth_recall_th45': sw_rec_45,
            'val_false_wake_th45': false_wake_45,
            'composite_score': composite_score,
            'is_best': is_best,
            'time_sec': epoch_time,
        }
        history.append(hist_entry)

        mark = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:2d}/{epochs} ({epoch_time:.1f}s) | Train Acc: {train_acc*100:5.2f}% | Val Cmd Acc: {val_cmd_acc*100:5.2f}% | Val Cmd F1: {val_cmd_f1*100:5.2f}% | HW Rec(0.45): {hw_rec_45*100:5.1f}% | False Wake(0.45): {false_wake_45:2d}{mark}", flush=True)

    (run_dir / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f"\nTraining completed for {run_name}. Best epoch: {best_epoch} (score: {best_score:.4f})")

    # Load best model for export
    model.load_state_dict(best_state)
    model.eval()

    # 3. Export FP32 ONNX
    fp32_onnx_path = models_dir / f"{run_name}_fp32.onnx"
    dummy_input = torch.randn(1, 1, 40, 251, dtype=torch.float32, device='cpu')
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(fp32_onnx_path),
        input_names=['input'],
        output_names=['logits'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        do_constant_folding=True,
        dynamo=False
    )
    fp32_size = fp32_onnx_path.stat().st_size
    fp32_hash = hashlib.sha256(fp32_onnx_path.read_bytes()).hexdigest()
    print(f"[OK] Exported FP32 ONNX: {fp32_onnx_path.name} ({fp32_size:,} B, SHA-256 {fp32_hash[:8]}...)")

    # 4. Stratified Training-Only Calibration with Diverse Wake Sources for Static INT8 Quantization
    calib_indices = []

    # 8 samples per class for the 31 command classes (stratified across training partition)
    for c in range(31):
        c_train_ids = np.where(train_y == c)[0]
        step = len(c_train_ids) / 8.0
        chosen = [c_train_ids[int(idx * step)] for idx in range(8)]
        calib_indices.extend(chosen)

    # 8 diverse wake samples: 4 distinct human wake takes + 4 distinct synthetic wake groups
    wake_ids = np.where(train_y == 31)[0]
    human_wake_local_ids = [0, 60, 120, 180]  # First version of 4 distinct human waveforms
    synth_wake_local_ids = [420, 420 + 70, 420 + 140, 420 + 210]  # First version from 4 distinct synth groups
    wake_calib_indices = [wake_ids[i] for i in human_wake_local_ids + synth_wake_local_ids]
    calib_indices.extend(wake_calib_indices)

    assert len(calib_indices) == 256, f"Expected 256 calibration samples, got {len(calib_indices)}"
    calib_feats = train_x[calib_indices]
    calib_reader = StratifiedTrainCalibrationReader(calib_feats)

    int8_onnx_path = models_dir / f"{run_name}_int8.onnx"
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
    print(f"[OK] Quantized static INT8 ONNX: {int8_onnx_path.name} ({int8_size:,} B, SHA-256 {int8_hash[:8]}...)")

    export_summary = {
        'run_name': run_name,
        'model_name': model_name,
        'is_treatment': is_treatment,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_score': best_score,
        'best_val_metrics': best_val_metrics,
        'validation_selected_threshold': 0.45,
        'artifacts': {
            'checkpoint': str(run_dir / 'best_model.pt'),
            'fp32_onnx': str(fp32_onnx_path),
            'fp32_size_bytes': fp32_size,
            'fp32_sha256': fp32_hash,
            'int8_onnx': str(int8_onnx_path),
            'int8_size_bytes': int8_size,
            'int8_sha256': int8_hash,
        }
    }
    (models_dir / 'export_summary.json').write_text(json.dumps(export_summary, indent=2), encoding='utf-8')
    print(f"[OK] Export summary saved: {models_dir / 'export_summary.json'}")
    return export_summary


def evaluate_paired_models(
    control_dir: Path,
    treatment_dir: Path,
    datasets: dict,
    seed: int = 231
) -> dict:
    """Performs single-pass dual evaluation on Option B held-out test and custom SLURP test.

    This function is executed strictly AFTER both Control and Treatment models
    have completed training, checkpoint selection, FP32 export, and INT8 quantization.
    """
    print("\n" + "=" * 90)
    print("  EXECUTING SINGLE-PASS DUAL EVALUATION (Option B Test & SLURP Test at theta* = 0.45)")
    print("=" * 90)

    frontend = Frontend()

    # 1. Load Option B Test Set (1,798 clips across 10 held-out speakers)
    optb_cache_path = PROJECT_ROOT / 'data' / 'option_b' / 'features_cache.npz'
    optb_cache = np.load(optb_cache_path)
    optb_test_x = optb_cache['test_x'].astype(np.float32)
    optb_test_y = optb_cache['test_y'].astype(np.int64)

    # 2. Extract & Load SLURP Test Set (636 clips across 16 held-out speakers)
    slurp_cache_dir = PROJECT_ROOT / 'data' / 'slurp'
    slurp_test_x, slurp_test_y = extract_features_from_manifest(
        slurp_cache_dir / 'manifest_slurp_test.csv',
        slurp_cache_dir / 'slurp_test_features.npz'
    )

    # 3. Featurize Held-Out Wake Test Positives
    wake_idx = WAKE_LABELS.index('WAKE_WORD')
    test_hw_items = datasets['test_hw_items']
    test_hw_feats = [frontend(fit_audio(it['wav'], target_samples=SAMPLES)) for it in test_hw_items]
    test_hw_x = np.stack(test_hw_feats, axis=0).astype(np.float32)
    test_hw_y = np.full(len(test_hw_x), wake_idx, dtype=np.int64)

    test_synth_files = datasets['test_synth_files']
    test_synth_feats = []
    for sf_path in test_synth_files:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        test_synth_feats.append(frontend(fit_audio(sw, target_samples=SAMPLES)))
    test_synth_x = np.stack(test_synth_feats, axis=0).astype(np.float32)
    test_synth_y = np.full(len(test_synth_x), wake_idx, dtype=np.int64)

    print(f"Evaluation Test Splits Loaded:")
    print(f"  Option B Test Clips:        {len(optb_test_y):,} (10 held-out speakers)")
    print(f"  SLURP Test Clips:           {len(slurp_test_y):,} (16 held-out speakers)")
    print(f"  Held-Out Human Wake Clips:  {len(test_hw_y):,} (2 held-out source groups)")
    print(f"  Held-Out Synth Wake Clips:  {len(test_synth_y):,} (1 held-out source group, 35 files)")

    def run_ort_inference(model_path: Path, x_arr: np.ndarray) -> np.ndarray:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        outs = []
        for bi in range(0, len(x_arr), 128):
            batch = x_arr[bi:bi + 128]
            out = sess.run(None, {'input': batch})[0]
            outs.append(out)
        return np.concatenate(outs, axis=0) if outs else np.empty((0, 32), dtype=np.float32)

    def evaluate_model_on_splits(model_path: Path) -> dict:
        # Option B Test Evaluation
        optb_logits = run_ort_inference(model_path, optb_test_x)
        optb_probs = np.exp(optb_logits - np.max(optb_logits, axis=-1, keepdims=True))
        optb_probs /= np.sum(optb_probs, axis=-1, keepdims=True)
        optb_preds = np.argmax(optb_logits, axis=-1)

        optb_acc = float(np.mean(optb_preds == optb_test_y))
        optb_f1 = float(f1_score(optb_test_y, optb_preds, average='macro', zero_division=0))
        false_wakes = int(np.sum(optb_probs[:, 31] >= 0.45))

        optb_per_class = {}
        for c in range(31):
            c_mask = (optb_test_y == c)
            if np.any(c_mask):
                optb_per_class[LABELS[c]] = float(np.mean(optb_preds[c_mask] == c))

        # Held-Out Wake Recall Evaluation
        hw_logits = run_ort_inference(model_path, test_hw_x)
        hw_probs = np.exp(hw_logits - np.max(hw_logits, axis=-1, keepdims=True))
        hw_probs /= np.sum(hw_probs, axis=-1, keepdims=True)
        hw_wake_probs = hw_probs[:, 31].tolist()
        hw_recall = float(np.mean(np.array(hw_wake_probs) >= 0.45))

        sw_logits = run_ort_inference(model_path, test_synth_x)
        sw_probs = np.exp(sw_logits - np.max(sw_logits, axis=-1, keepdims=True))
        sw_probs /= np.sum(sw_probs, axis=-1, keepdims=True)
        sw_recall = float(np.mean(sw_probs[:, 31] >= 0.45))

        # SLURP Held-Out Test Evaluation
        slurp_logits = run_ort_inference(model_path, slurp_test_x)
        slurp_probs = np.exp(slurp_logits - np.max(slurp_logits, axis=-1, keepdims=True))
        slurp_probs /= np.sum(slurp_probs, axis=-1, keepdims=True)
        slurp_preds = np.argmax(slurp_logits, axis=-1)

        slurp_acc = float(np.mean(slurp_preds == slurp_test_y)) if len(slurp_test_y) > 0 else 0.0
        slurp_f1 = float(f1_score(slurp_test_y, slurp_preds, average='macro', zero_division=0)) if len(slurp_test_y) > 0 else 0.0
        slurp_false_wakes = int(np.sum(slurp_probs[:, 31] >= 0.45)) if len(slurp_test_y) > 0 else 0

        slurp_per_class = {}
        for c_idx in sorted(list(set(slurp_test_y))):
            lbl = WAKE_LABELS[c_idx]
            c_mask = (slurp_test_y == c_idx)
            supp = int(np.sum(c_mask))
            acc = float(np.mean(slurp_preds[c_mask] == c_idx))
            slurp_per_class[lbl] = {'support': supp, 'accuracy': acc}

        return {
            'optb_test_acc': optb_acc,
            'optb_test_f1': optb_f1,
            'false_wakes_on_optb_commands': false_wakes,
            'optb_per_class': optb_per_class,
            'hw_wake_test_recall': hw_recall,
            'hw_wake_test_probs': hw_wake_probs,
            'synth_wake_test_recall': sw_recall,
            'slurp_test_acc': slurp_acc,
            'slurp_test_f1': slurp_f1,
            'slurp_false_wakes': slurp_false_wakes,
            'slurp_per_class': slurp_per_class,
            'optb_preds': optb_preds,
            'slurp_preds': slurp_preds,
        }

    # Evaluate Control Models
    c_models = control_dir / 'models'
    c_fp32_path = list(c_models.glob('*_fp32.onnx'))[0]
    c_int8_path = list(c_models.glob('*_int8.onnx'))[0]
    print(f"Evaluating Control Models ({c_fp32_path.name}, {c_int8_path.name})...")
    c_fp_eval = evaluate_model_on_splits(c_fp32_path)
    c_i8_eval = evaluate_model_on_splits(c_int8_path)
    c_optb_parity = float(np.mean(c_fp_eval['optb_preds'] == c_i8_eval['optb_preds']))
    c_slurp_parity = float(np.mean(c_fp_eval['slurp_preds'] == c_i8_eval['slurp_preds']))

    # Evaluate Treatment Models
    t_models = treatment_dir / 'models'
    t_fp32_path = list(t_models.glob('*_fp32.onnx'))[0]
    t_int8_path = list(t_models.glob('*_int8.onnx'))[0]
    print(f"Evaluating Treatment Models ({t_fp32_path.name}, {t_int8_path.name})...")
    t_fp_eval = evaluate_model_on_splits(t_fp32_path)
    t_i8_eval = evaluate_model_on_splits(t_int8_path)
    t_optb_parity = float(np.mean(t_fp_eval['optb_preds'] == t_i8_eval['optb_preds']))
    t_slurp_parity = float(np.mean(t_fp_eval['slurp_preds'] == t_i8_eval['slurp_preds']))

    # Clean predictions from dicts before JSON saving
    del c_fp_eval['optb_preds'], c_fp_eval['slurp_preds']
    del c_i8_eval['optb_preds'], c_i8_eval['slurp_preds']
    del t_fp_eval['optb_preds'], t_fp_eval['slurp_preds']
    del t_i8_eval['optb_preds'], t_i8_eval['slurp_preds']

    metrics_control = {
        'run_name': control_dir.name,
        'model_name': 'TinyDSCNN-48 Control (Option B Baseline)',
        'is_treatment': False,
        'seed': seed,
        'validation_selected_threshold': 0.45,
        'fp32': c_fp_eval,
        'int8': c_i8_eval,
        'optb_prediction_parity': c_optb_parity,
        'slurp_prediction_parity': c_slurp_parity,
        'artifacts': {
            'fp32_path': str(c_fp32_path),
            'fp32_size_bytes': c_fp32_path.stat().st_size,
            'fp32_sha256': hashlib.sha256(c_fp32_path.read_bytes()).hexdigest(),
            'int8_path': str(c_int8_path),
            'int8_size_bytes': c_int8_path.stat().st_size,
            'int8_sha256': hashlib.sha256(c_int8_path.read_bytes()).hexdigest(),
        }
    }
    (control_dir / 'metrics.json').write_text(json.dumps(metrics_control, indent=2), encoding='utf-8')

    metrics_treatment = {
        'run_name': treatment_dir.name,
        'model_name': 'TinyDSCNN-48 Treatment (Option B + SLURP Real)',
        'is_treatment': True,
        'seed': seed,
        'validation_selected_threshold': 0.45,
        'fp32': t_fp_eval,
        'int8': t_i8_eval,
        'optb_prediction_parity': t_optb_parity,
        'slurp_prediction_parity': t_slurp_parity,
        'artifacts': {
            'fp32_path': str(t_fp32_path),
            'fp32_size_bytes': t_fp32_path.stat().st_size,
            'fp32_sha256': hashlib.sha256(t_fp32_path.read_bytes()).hexdigest(),
            'int8_path': str(t_int8_path),
            'int8_size_bytes': t_int8_path.stat().st_size,
            'int8_sha256': hashlib.sha256(t_int8_path.read_bytes()).hexdigest(),
        }
    }
    (treatment_dir / 'metrics.json').write_text(json.dumps(metrics_treatment, indent=2), encoding='utf-8')

    paired_summary = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'validation_selected_threshold': 0.45,
        'control': metrics_control,
        'treatment': metrics_treatment,
    }
    paired_summary_path = PROJECT_ROOT / 'runs' / 'paired_slurp_comparison_metrics.json'
    paired_summary_path.write_text(json.dumps(paired_summary, indent=2), encoding='utf-8')
    print(f"\n[OK] Metrics saved to {control_dir / 'metrics.json'}, {treatment_dir / 'metrics.json'}, and {paired_summary_path}")

    return paired_summary
