"""Train a separate scratch-initialized two-class personalized wake detector.

The command model and deployed artifacts are never modified. Real wake clips are
deduplicated by decoded PCM hash and held out by recording before augmentation.
The wake threshold is selected using validation data only.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT.parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_antigrav.config import CHANNELS, FEATURE_SHAPE, LABELS, SR, SAMPLES
from tinyvcm_antigrav.frontend import Frontend, fit_audio
from tinyvcm_antigrav.model import TinyDSCNN

SEED = 231
EPOCHS = 25
SAMPLES_PER_CLASS_PER_EPOCH = 768
BATCH_SIZE = 64
MAX_VALIDATION_FALSE_RATE = 0.001


def pcm_sha256(audio: np.ndarray) -> str:
    pcm = np.asarray(audio, dtype="<f4").reshape(-1)
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def load_unique_human_wakes():
    manifest = PROJECT_ROOT / "data" / "human" / "manifest.csv"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    groups = {}
    for row in rows:
        if row.get("label") != "wake_word":
            continue
        path = PROJECT_ROOT / row["path"]
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz wake clips; got {sr} Hz")
        digest = pcm_sha256(audio)
        groups.setdefault(digest, {"hash": digest, "audio": audio, "speaker": row["speaker"]})
    return rows, list(groups.values()), hashlib.sha256(manifest.read_bytes()).hexdigest()


def split_wake_groups(groups, seed=SEED):
    """Fixed, deterministic 10/3/remainder recording split; never split an utterance."""
    order = np.random.default_rng(seed).permutation(len(groups)).tolist()
    shuffled = [groups[i] for i in order]
    if len(shuffled) < 12:
        raise ValueError(f"Need at least 12 unique real wake recordings; found {len(shuffled)}")
    train_end = max(1, int(round(len(shuffled) * 0.625)))
    val_end = train_end + max(1, int(round(len(shuffled) * 0.1875)))
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def place_and_augment(audio, rng, noises):
    """Create a rolling 2.5-second context window, with optional training-only noise."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > SAMPLES:
        audio = fit_audio(audio)
    out = np.zeros(SAMPLES, dtype=np.float32)
    start = int(rng.integers(0, SAMPLES - len(audio) + 1)) if len(audio) < SAMPLES else 0
    out[start:start + len(audio)] = audio
    out *= float(rng.uniform(0.7, 1.3))
    if noises and rng.random() < 0.8:
        noise = noises[int(rng.integers(0, len(noises)))]
        if len(noise) < SAMPLES:
            noise = np.tile(noise, int(np.ceil(SAMPLES / max(len(noise), 1))))
        offset = int(rng.integers(0, len(noise) - SAMPLES + 1))
        noise = noise[offset:offset + SAMPLES]
        signal_power = float(np.mean(out ** 2)) + 1e-9
        noise_power = float(np.mean(noise ** 2)) + 1e-9
        snr = float(rng.uniform(15.0, 32.0))
        out += noise * np.sqrt(signal_power / (10 ** (snr / 10.0) * noise_power))
    return np.clip(out, -1.0, 1.0)


def fixed_context_windows(audio):
    """Five possible phrase placements, used to score one recording as one event."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > SAMPLES:
        audio = fit_audio(audio)
    max_start = max(0, SAMPLES - len(audio))
    starts = np.linspace(0, max_start, 5, dtype=int)
    result = []
    for start in starts:
        window = np.zeros(SAMPLES, dtype=np.float32)
        window[start:start + len(audio)] = audio
        result.append(window)
    return result


def features_for_audio(frontend, audio):
    return np.stack([frontend(w) for w in audio]).astype(np.float32)


def build_positive_bank(frontend, train_groups, noises, seed=SEED):
    rng = np.random.default_rng(seed + 100)
    features = []
    for item in train_groups:
        # One clean and 31 random rolling-context augmentations per unique take.
        for j in range(32):
            window = fit_audio(item["audio"]) if j == 0 else place_and_augment(item["audio"], rng, noises)
            features.append(frontend(window))

    # TTS wake variants are training-only support; real-speaker results are reported separately.
    synth_dir = PROJECT_ROOT / "data" / "dataset" / "wake_word"
    synth_groups = {}
    for path in sorted(synth_dir.glob("*.wav")):
        group = re.sub(r"_\d+_[^_]+$", "", path.stem)
        synth_groups.setdefault(group, []).append(path)
    ordered_groups = sorted(synth_groups)
    rng.shuffle(ordered_groups)
    n_train = max(1, int(len(ordered_groups) * 0.75))
    selected_groups = ordered_groups[:n_train]
    synth_train_files = [p for g in selected_groups for p in synth_groups[g]]
    for path in synth_train_files:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz synthetic wake clips; got {sr} Hz")
        features.append(frontend(place_and_augment(audio, rng, noises)))
    return np.stack(features).astype(np.float32), {
        "human_augmented_windows": len(train_groups) * 32,
        "synthetic_training_files": len(synth_train_files),
        "synthetic_training_groups": selected_groups,
        "synthetic_total_groups": len(ordered_groups),
    }


def probabilities(session, features, batch=256):
    inp_name = session.get_inputs()[0].name
    chunks = []
    for i in range(0, len(features), batch):
        logits = session.run(None, {inp_name: features[i:i + batch]})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        chunks.append(probs / probs.sum(axis=1, keepdims=True))
    return np.concatenate(chunks, axis=0)


def validation_threshold(wake_max_probs, nonwake_probs, max_false_rate=MAX_VALIDATION_FALSE_RATE):
    """Maximize real-wake recall, then minimize false accepts, under a val FA cap."""
    false_budget = max(1, int(np.floor(len(nonwake_probs) * max_false_rate)))
    candidates = np.unique(np.concatenate([wake_max_probs, nonwake_probs, np.array([1.0])]))
    feasible = []
    for threshold in candidates:
        wake_recall = float(np.mean(wake_max_probs >= threshold))
        false_count = int(np.sum(nonwake_probs >= threshold))
        if false_count <= false_budget:
            feasible.append((wake_recall, -false_count, float(threshold), false_count))
    if not feasible:
        return 1.01, {"wake_recall": 0.0, "false_count": len(nonwake_probs), "false_budget": false_budget}
    best = max(feasible)
    return best[2], {"wake_recall": best[0], "false_count": best[3], "false_budget": false_budget}


def evaluate_logits(model, features, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch):
            x = torch.from_numpy(features[i:i + batch]).to(device)
            out.append(model(x).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


class CalibrationReader(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = [{"input": x[None].astype(np.float32)} for x in samples]
        self.reset()

    def reset(self):
        self._iter = iter(self.samples)

    def get_next(self):
        return next(self._iter, None)


def run_training():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"antigrav-binary-wake-{timestamp}"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_dir.name} | device={device} | pretrained weights: none", flush=True)

    human_rows, wake_groups, manifest_hash = load_unique_human_wakes()
    train_wakes, val_wakes, test_wakes = split_wake_groups(wake_groups)
    print(f"Human wake groups: {len(wake_groups)} unique; split {len(train_wakes)}/{len(val_wakes)}/{len(test_wakes)} train/val/test; speakers={sorted({g['speaker'] for g in wake_groups})}", flush=True)

    frontend = Frontend()
    noise_dir = PROJECT_ROOT / "data" / "dataset" / "_background_noise_"
    noises = []
    for path in sorted(noise_dir.glob("*.wav")):
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr == SR and len(audio):
            noises.append(audio)
    print(f"Noise clips available for training augmentation: {len(noises)}", flush=True)

    cache_path = PROJECT_ROOT / "data" / "option_b" / "features_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        train_neg = cache["train_x"].astype(np.float32)
        val_neg = cache["val_x"].astype(np.float32)
    print(f"Negative command windows: {len(train_neg)} training; {len(val_neg)} validation. Test split is excluded from training and threshold selection.", flush=True)

    pos_train, train_source_counts = build_positive_bank(frontend, train_wakes, noises)
    val_windows, val_groups = [], []
    for i, item in enumerate(val_wakes):
        for window in fixed_context_windows(item["audio"]):
            val_windows.append(window)
            val_groups.append(i)
    val_pos_features = features_for_audio(frontend, val_windows)
    print(f"Positive training windows: {len(pos_train)} ({train_source_counts['human_augmented_windows']} human-derived, {train_source_counts['synthetic_training_files']} synthetic). Validation real recordings: {len(val_wakes)}.", flush=True)

    model = TinyDSCNN(classes=2, channels=CHANNELS).to(device)
    initial_blob = b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
    initial_hash = hashlib.sha256(initial_blob).hexdigest()
    params = sum(p.numel() for p in model.parameters())
    print(f"Scratch model initialized; parameters={params}; initial state SHA-256={initial_hash}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=3e-5)
    criterion = nn.CrossEntropyLoss()
    pos_bank = torch.from_numpy(pos_train)
    neg_bank = torch.from_numpy(train_neg)
    n_per_class = SAMPLES_PER_CLASS_PER_EPOCH
    best_key = None
    best_path = run_dir / "best_model.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pos_idx = rng.integers(0, len(pos_bank), size=n_per_class)
        neg_idx = rng.choice(len(neg_bank), size=n_per_class, replace=False)
        x = torch.cat([pos_bank[pos_idx], neg_bank[neg_idx]], dim=0).numpy()
        y = np.concatenate([np.ones(n_per_class, dtype=np.int64), np.zeros(n_per_class, dtype=np.int64)])
        order = rng.permutation(len(y))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_pos_probs = torch.softmax(torch.from_numpy(evaluate_logits(model, val_pos_features, device)), dim=1).numpy()[:, 1]
        val_max_per_clip = np.array([val_pos_probs[np.asarray(val_groups) == j].max() for j in range(len(val_wakes))])
        val_neg_probs = torch.softmax(torch.from_numpy(evaluate_logits(model, val_neg, device)), dim=1).numpy()[:, 1]
        threshold, val_metrics = validation_threshold(val_max_per_clip, val_neg_probs)
        mean_loss = float(np.mean(losses))
        key = (val_metrics["wake_recall"], -val_metrics["false_count"], threshold)
        row = {"epoch": epoch, "train_loss": mean_loss, "val_wake_recall": val_metrics["wake_recall"], "val_false_wakes": val_metrics["false_count"], "val_false_budget": val_metrics["false_budget"], "threshold": threshold}
        history.append(row)
        mark = " *" if best_key is None or key > best_key else ""
        if best_key is None or key > best_key:
            best_key = key
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={mean_loss:.4f} | val wake={val_metrics['wake_recall']:.3f} ({int(val_metrics['wake_recall']*len(val_wakes))}/{len(val_wakes)}) | val false={val_metrics['false_count']}/{len(val_neg)} cap={val_metrics['false_budget']} | threshold={threshold:.4f}{mark}", flush=True)

    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    checkpoint_val_pos_probs = torch.softmax(torch.from_numpy(evaluate_logits(model, val_pos_features, device)), dim=1).numpy()[:, 1]
    checkpoint_val_max_per_clip = np.array([checkpoint_val_pos_probs[np.asarray(val_groups) == j].max() for j in range(len(val_wakes))])
    checkpoint_val_neg_probs = torch.softmax(torch.from_numpy(evaluate_logits(model, val_neg, device)), dim=1).numpy()[:, 1]
    checkpoint_threshold, checkpoint_val_metrics = validation_threshold(checkpoint_val_max_per_clip, checkpoint_val_neg_probs)
    print(f"Best-checkpoint FP32 validation: wake recall={checkpoint_val_metrics['wake_recall']:.3f}; command false accepts={checkpoint_val_metrics['false_count']}/{len(val_neg)}", flush=True)

    fp32_path = model_dir / "binary_wake_fp32.onnx"
    int8_path = model_dir / "binary_wake_int8.onnx"
    model.cpu()
    torch.onnx.export(model, torch.randn(1, *FEATURE_SHAPE), str(fp32_path), input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    calib_n = min(64, len(pos_train), len(train_neg))
    calib = np.concatenate([pos_train[:calib_n], train_neg[:calib_n]], axis=0)
    quantize_static(model_input=str(fp32_path), model_output=str(int8_path), calibration_data_reader=CalibrationReader(calib), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    onnx.checker.check_model(onnx.load(str(int8_path)))
    print(f"Exported FP32 {fp32_path.stat().st_size:,} B and INT8 {int8_path.stat().st_size:,} B ONNX", flush=True)

    # Select the operating threshold on the deployed-format INT8 model using
    # validation only. This avoids calibrating a threshold on FP32 logits and
    # then applying it to a numerically different quantized model.
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    val_fp32_pos = probabilities(sess_fp32, val_pos_features)[:, 1]
    val_int8_pos = probabilities(sess_int8, val_pos_features)[:, 1]
    val_fp32_neg = probabilities(sess_fp32, val_neg)[:, 1]
    val_int8_neg = probabilities(sess_int8, val_neg)[:, 1]
    val_groups_array = np.asarray(val_groups)
    val_fp32_max_per_clip = np.array([val_fp32_pos[val_groups_array == j].max() for j in range(len(val_wakes))])
    val_int8_max_per_clip = np.array([val_int8_pos[val_groups_array == j].max() for j in range(len(val_wakes))])
    threshold, val_int8_metrics = validation_threshold(val_int8_max_per_clip, val_int8_neg)
    val_fp32_metrics = {
        "wake_recall": float(np.mean(val_fp32_max_per_clip >= threshold)),
        "false_count": int(np.sum(val_fp32_neg >= threshold)),
    }
    val_metrics = {
        "threshold_selected_on": "INT8 ONNX validation scores",
        "wake_clips": len(val_wakes),
        "command_windows": len(val_neg),
        "int8": val_int8_metrics,
        "fp32_at_same_int8_selected_threshold": val_fp32_metrics,
        "fp32_checkpoint_selection_threshold": checkpoint_threshold,
        "fp32_checkpoint_selection_metrics": checkpoint_val_metrics,
    }
    export_summary = {
        "model_name": "TinyDSCNN-Binary-Wake",
        "classes": ["NON_WAKE", "WAKE_WORD"],
        "validation_selected_threshold": threshold,
        "threshold_selected_on": "INT8 ONNX validation scores",
        "fp32_onnx": {
            "path": fp32_path.name,
            "size_bytes": fp32_path.stat().st_size,
            "sha256": hashlib.sha256(fp32_path.read_bytes()).hexdigest(),
        },
        "int8_onnx": {
            "path": int8_path.name,
            "size_bytes": int8_path.stat().st_size,
            "sha256": hashlib.sha256(int8_path.read_bytes()).hexdigest(),
        },
        "calibration_samples": int(len(calib)),
        "calibration_sources": "training-only wake and Option B command features",
    }
    (run_dir / "export_summary.json").write_text(json.dumps(export_summary, indent=2), encoding="utf-8")
    print(f"Frozen INT8 validation threshold={threshold:.4f}; wake recall={val_int8_metrics['wake_recall']:.3f}; command false accepts={val_int8_metrics['false_count']}/{len(val_neg)}", flush=True)

    # Final one-pass development comparison only: prior project work has accessed
    # this Option B holdout, so it is explicitly not claimed as a pristine test.
    with np.load(cache_path, allow_pickle=False) as cache:
        test_neg = cache["test_x"].astype(np.float32)
    test_pos_features, test_pos_groups = [], []
    for i, item in enumerate(test_wakes):
        for window in fixed_context_windows(item["audio"]):
            test_pos_features.append(window)
            test_pos_groups.append(i)
    test_pos_features = features_for_audio(frontend, test_pos_features)

    p_fp32_neg = probabilities(sess_fp32, test_neg)[:, 1]
    p_int8_neg = probabilities(sess_int8, test_neg)[:, 1]
    p_fp32_pos = probabilities(sess_fp32, test_pos_features)[:, 1]
    p_int8_pos = probabilities(sess_int8, test_pos_features)[:, 1]
    max_fp32_pos = np.array([p_fp32_pos[np.asarray(test_pos_groups) == j].max() for j in range(len(test_wakes))])
    max_int8_pos = np.array([p_int8_pos[np.asarray(test_pos_groups) == j].max() for j in range(len(test_wakes))])

    old_path = PROJECT_ROOT / "models" / "antigrav_wake32_int8.onnx"
    old_sess = ort.InferenceSession(str(old_path), providers=["CPUExecutionProvider"])
    old_neg_probs = probabilities(old_sess, test_neg)
    old_pos_probs = probabilities(old_sess, test_pos_features)
    old_neg_fa = int(np.sum((old_neg_probs[:, 31] >= 0.60) & (old_neg_probs.argmax(axis=1) == 31)))
    old_pos_scores = np.array([old_pos_probs[np.asarray(test_pos_groups) == j, 31].max() for j in range(len(test_wakes))])
    old_pos_recall = int(np.sum((old_pos_scores >= 0.60)))

    fp32_neg_fa = int(np.sum(p_fp32_neg >= threshold))
    int8_neg_fa = int(np.sum(p_int8_neg >= threshold))
    fp32_pos_hits = int(np.sum(max_fp32_pos >= threshold))
    int8_pos_hits = int(np.sum(max_int8_pos >= threshold))
    parity = float(np.mean(probabilities(sess_fp32, test_neg).argmax(axis=1) == probabilities(sess_int8, test_neg).argmax(axis=1)))
    metrics = {
        "run_name": run_dir.name,
        "objective": "personalized two-class wake detector; NON_WAKE=0, WAKE_WORD=1",
        "scratch_initialized": True,
        "pretrained_weights_used": False,
        "seed": SEED,
        "initial_state_sha256": initial_hash,
        "parameter_count": params,
        "human_wake_manifest_rows": sum(r.get("label") == "wake_word" for r in human_rows),
        "human_wake_unique_pcm": len(wake_groups),
        "human_speakers": sorted({g["speaker"] for g in wake_groups}),
        "split_hashes": {s: [g["hash"] for g in gs] for s, gs in [("train", train_wakes), ("validation", val_wakes), ("test", test_wakes)]},
        "human_manifest_sha256": manifest_hash,
        "validation_threshold": threshold,
        "validation": val_metrics,
        "development_holdout_comparison": {
            "note": "One pass after training; Option B holdout artifacts have prior project evaluations and are not a pristine test. During planning their shapes were inspected before training but no test samples or labels were used in fit/threshold selection. Threshold and checkpoint were frozen using validation only; treat this as exploratory.",
            "human_wake_clips": len(test_wakes),
            "old_32_class_model_wake_hits_at_0.60": old_pos_recall,
            "binary_fp32_wake_hits_at_frozen_threshold": fp32_pos_hits,
            "binary_int8_wake_hits_at_frozen_threshold": int8_pos_hits,
            "option_b_command_windows": len(test_neg),
            "old_32_class_false_wakes_at_0.60": old_neg_fa,
            "binary_fp32_false_wakes_at_frozen_threshold": fp32_neg_fa,
            "binary_int8_false_wakes_at_frozen_threshold": int8_neg_fa,
            "fp32_int8_command_prediction_parity": parity,
            "human_wake_max_scores_int8": max_int8_pos.tolist(),
        },
        "artifacts": {"fp32_sha256": hashlib.sha256(fp32_path.read_bytes()).hexdigest(), "fp32_size_bytes": fp32_path.stat().st_size, "int8_sha256": hashlib.sha256(int8_path.read_bytes()).hexdigest(), "int8_size_bytes": int8_path.stat().st_size},
        "limitations": ["Only one real wake speaker is available; results are personalized, not multi-speaker robust.", "The held-out human wake count is small.", "Command-window false accepts are not continuous-audio false activations per hour.", "Option B development holdout has prior project use; this result is exploratory, not a pristine test.", "This candidate is not deployed; Pi frontend-plus-model latency and live wake performance are not verified."],
        "deployed_artifacts_modified": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "split_provenance.json").write_text(json.dumps({"human_wakes": {"rows": len([r for r in human_rows if r.get('label') == 'wake_word']), "unique_pcm": len(wake_groups), "train": len(train_wakes), "validation": len(val_wakes), "test": len(test_wakes)}, "training_sources": train_source_counts, "human_manifest_sha256": manifest_hash}, indent=2), encoding="utf-8")
    print(f"Holdout wake recall: INT8 {int8_pos_hits}/{len(test_wakes)}; false wake windows: INT8 {int8_neg_fa}/{len(test_neg)}, old 32-class {old_neg_fa}/{len(test_neg)}", flush=True)
    print(f"INT8 SHA-256: {metrics['artifacts']['int8_sha256']} | size={metrics['artifacts']['int8_size_bytes']:,} B", flush=True)
    print("This is an experimental candidate only; deployed model files were not changed.", flush=True)
    return metrics


if __name__ == "__main__":
    run_training()
