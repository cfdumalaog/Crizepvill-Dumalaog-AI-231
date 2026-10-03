"""Train and compare an isolated scratch TinyDSCNN candidate on the ME2 Spoken Command Dataset.

The script has two explicit phases. ``fit`` reads only the frozen train shard,
carves a deterministic source-speaker-disjoint validation split, trains and
freezes a candidate plus validation-selected reject gates. ``score-test`` is a
one-shot operation that writes a lock before opening the published test audio.
It never writes to the active deployment.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import scipy.io.wavfile as wavfile
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

PROJECT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT / "data" / "ai231-me2-voice-commands-hf-a90b8d10"
REVISION = "a90b8d106349b02c5570a1a258503386043f63b2"
RUN_NAME = f"classagreementvcm-20261002-a90b8d10-r3"
RUN = PROJECT / "runs" / RUN_NAME
ACTIVE_INTENT = PROJECT / "deployment" / "current_vcm" / "models" / "intent_int8.onnx"
ACTIVE_WAKE = PROJECT / "deployment" / "current_vcm" / "models" / "binary_wake_int8.onnx"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tinyvcm_model.config import (  # noqa: E402
    FEATURE_SHAPE, INTENT_CONFIDENCE_THRESHOLD as ACTIVE_CONFIDENCE,
    INTENT_MARGIN_THRESHOLD as ACTIVE_MARGIN, LABELS, MELS, SAMPLES, SR,
)
from tinyvcm_model.frontend import Frontend, fit_audio  # noqa: E402
from tinyvcm_model.model import TinyDSCNN  # noqa: E402

SEED = 231
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.003
PATIENCE = 18
VALIDATION_FRACTION = 0.20
FALSE_ACTION_LIMIT = 0.01
REJECT_GRID_CONFIDENCE = np.round(np.arange(0.40, 0.991, 0.01), 2)
REJECT_GRID_MARGIN = np.round(np.arange(0.00, 0.951, 0.025), 3)
COMMAND_VALUES = {
    "ALARM": {"6:00 AM": "ALARM_6_00AM", "8:00 AM": "ALARM_8_00AM", "9:00 PM": "ALARM_9_00PM"},
    "BRIGHTNESS": {"20 percent": "BRIGHTNESS_20", "60 percent": "BRIGHTNESS_60", "100 percent": "BRIGHTNESS_100"},
    "COLOR": {"Blue": "COLOR_BLUE", "Green": "COLOR_GREEN", "Red": "COLOR_RED"},
    "CREATE_REMINDER": {"Drink water": "CREATE_REMINDER_DRINK_WATER", "Exercise": "CREATE_REMINDER_EXERCISE", "Study": "CREATE_REMINDER_STUDY"},
    "TEMPERATURE": {"18 degrees": "TEMPERATURE_18", "22 degrees": "TEMPERATURE_22", "26 degrees": "TEMPERATURE_26"},
    "TIMER": {"10 seconds": "TIMER_10s", "1 minute": "TIMER_1m", "30 seconds": "TIMER_30s"},
}
FIXED_COMMANDS = {
    "CALL", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT",
    "PAUSE", "PLAY_MUSIC", "STOP", "TIME", "VOLUME_DOWN", "VOLUME_UP", "WEATHER",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_leaf(command: str, slot_value: str, variation: str) -> str | None:
    if command in FIXED_COMMANDS and variation:
        return command
    return COMMAND_VALUES.get(command, {}).get(slot_value)


def read_rows(split: str, include_audio: bool = False) -> list[dict[str, Any]]:
    paths = sorted((SNAPSHOT / "data").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} Parquet shard under {SNAPSHOT}")
    columns = ["file", "command", "variation", "slot_value", "out_of_scope", "bucket", "speaker_id", "source", "is_synthetic", "accent_group", "transcript_source", "duration_s"]
    if include_audio:
        columns.insert(0, "audio")
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=48, columns=columns):
            rows.extend(batch.to_pylist())
    return rows


def classify_row(row: dict[str, Any]) -> tuple[str, str | None]:
    if int(row.get("out_of_scope") or 0) == 1 or row.get("command") == "OUT_OF_SCOPE":
        return "true_oos", None
    leaf = derive_leaf(str(row.get("command") or ""), str(row.get("slot_value") or ""), str(row.get("variation") or ""))
    if leaf not in LABELS:
        return "unsupported_slot", None
    return "supported", leaf


def group_id(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "unknown")
    speaker = str(row.get("speaker_id") or "unknown")
    return sha256_bytes(f"{source}\0{speaker}".encode("utf-8"))


def select_validation_groups(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    groups = np.asarray([row["group_id"] for row in rows], dtype=object)
    y = np.asarray([row["target"] if row["target"] is not None else -1 for row in rows], dtype=np.int64)
    categories = np.asarray([row["category"] for row in rows], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1600, test_size=VALIDATION_FRACTION, random_state=SEED)
    best: tuple[float, set[str], dict[str, Any]] | None = None
    for train_ids, val_ids in splitter.split(np.zeros(len(rows)), y, groups):
        val_groups = set(groups[val_ids].tolist())
        train_y = y[train_ids]
        val_y = y[val_ids]
        train_support = np.bincount(train_y[train_y >= 0], minlength=len(LABELS))
        val_support = np.bincount(val_y[val_y >= 0], minlength=len(LABELS))
        if np.any(train_support < 10) or np.any(val_support < 4):
            continue
        val_true_oos = int(np.sum(categories[val_ids] == "true_oos"))
        val_unsupported = int(np.sum(categories[val_ids] == "unsupported_slot"))
        if val_true_oos < 10 or val_unsupported < 5:
            continue
        train_true_oos = int(np.sum(categories[train_ids] == "true_oos"))
        train_unsupported = int(np.sum(categories[train_ids] == "unsupported_slot"))
        if train_true_oos < 10 or train_unsupported < 5:
            continue
        fraction = len(val_ids) / len(rows)
        leaf_share = val_support / np.maximum(train_support + val_support, 1)
        score = abs(fraction - VALIDATION_FRACTION) + float(np.mean(np.abs(leaf_share - VALIDATION_FRACTION)))
        summary = {
            "seed": SEED,
            "builder": "GroupShuffleSplit search, source-speaker groups, 1600 deterministic candidates",
            "validation_fraction_target": VALIDATION_FRACTION,
            "validation_fraction_rows": fraction,
            "train_groups": len(set(groups[train_ids].tolist())),
            "validation_groups": len(val_groups),
            "train_rows_by_category": dict(Counter(categories[train_ids].tolist())),
            "validation_rows_by_category": dict(Counter(categories[val_ids].tolist())),
            "minimum_train_supported_per_leaf": int(train_support.min()),
            "minimum_validation_supported_per_leaf": int(val_support.min()),
            "validation_supported_per_leaf": {LABELS[i]: int(val_support[i]) for i in range(len(LABELS))},
            "score": score,
        }
        if best is None or score < best[0]:
            best = (score, val_groups, summary)
    if best is None:
        raise RuntimeError("No deterministic group-disjoint validation split met class/OOS coverage constraints")
    return best[1], best[2]


def build_split_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows("train")
    if len(rows) != 10682:
        raise ValueError(f"Unexpected train rows: {len(rows)}")
    seen_files: set[str] = set()
    clean: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("file") or "")
        p = Path(name.replace("\\", "/"))
        if not name or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Unsafe or missing manifest path: {name!r}")
        if name in seen_files:
            raise ValueError(f"Duplicate source filename in training partition: {name}")
        seen_files.add(name)
        category, leaf = classify_row(row)
        row = dict(row)
        row["category"] = category
        row["target"] = LABELS.index(leaf) if leaf in LABELS else None
        row["group_id"] = group_id(row)
        row["row_id_sha256"] = sha256_bytes(name.encode("utf-8"))
        clean.append(row)
    val_groups, summary = select_validation_groups(clean)
    for row in clean:
        row["partition"] = "validation" if row["group_id"] in val_groups else "fit"
    fit_group_set = {row["group_id"] for row in clean if row["partition"] == "fit"}
    val_group_set = {row["group_id"] for row in clean if row["partition"] == "validation"}
    if fit_group_set & val_group_set:
        raise AssertionError("Source-speaker leakage in derived split")
    split_path = RUN / "train_validation_manifest.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id_sha256", "file", "category", "target_label", "partition", "group_id_sha256", "source", "is_synthetic", "accent_group", "duration_s", "bucket"]
    with split_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(clean, key=lambda r: r["file"]):
            writer.writerow({
                "row_id_sha256": row["row_id_sha256"], "file": row["file"], "category": row["category"],
                "target_label": LABELS[row["target"]] if row["target"] is not None else "",
                "partition": row["partition"], "group_id_sha256": row["group_id"], "source": row["source"],
                "is_synthetic": row["is_synthetic"], "accent_group": row["accent_group"],
                "duration_s": row["duration_s"], "bucket": row["bucket"],
            })
    summary.update({
        "total_train_partition_rows": len(clean),
        "source_snapshot_revision": REVISION,
        "fit_manifest_sha256": sha256_file(split_path),
        "validation_group_sha256": sha256_bytes("\n".join(sorted(val_groups)).encode("ascii")),
        "split_disjoint": True,
    })
    (RUN / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return clean, summary


def decode_audio(row: dict[str, Any]) -> tuple[np.ndarray, str]:
    audio_field = row.get("audio")
    wav_bytes = (audio_field or {}).get("bytes") if isinstance(audio_field, dict) else None
    if not wav_bytes:
        raise ValueError(f"No embedded WAV bytes: {row.get('file')}")
    rate, signal = wavfile.read(io.BytesIO(wav_bytes))
    if int(rate) != SR:
        raise ValueError(f"Sample-rate mismatch at {row.get('file')}: {rate}")
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Non-mono WAV at {row.get('file')}: {signal.shape}")
    if signal.dtype != np.int16 or not signal.size:
        raise ValueError(f"Expected non-empty PCM16 WAV at {row.get('file')}: {signal.dtype}, {signal.shape}")
    pcm_hash = sha256_bytes(signal.astype("<i2", copy=False).tobytes())
    waveform = signal.astype(np.float32) / 32768.0
    return fit_audio(waveform), pcm_hash


def materialize_features(rows: list[dict[str, Any]], split: str, frontend: Frontend, pcm_seen: dict[str, tuple[str, str, str]] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    selected = {
        row["file"]: row for row in rows
        if row["partition"] == split and (split == "validation" or row["target"] is not None)
    }
    feature_rows: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    pcm_seen = pcm_seen if pcm_seen is not None else {}
    duplicate_files: list[str] = []
    failures: list[dict[str, str]] = []
    scanned = 0
    for path in sorted((SNAPSHOT / "data").glob("train-*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32, columns=["audio", "file"]):
            for item in batch.to_pylist():
                name = str(item["file"])
                row = selected.get(name)
                if row is None:
                    continue
                scanned += 1
                try:
                    audio, pcm_hash = decode_audio(item | {"file": name})
                    label = row["target_label"] if "target_label" in row else (LABELS[row["target"]] if row["target"] is not None else None)
                    duplicate = pcm_seen.get(pcm_hash)
                    if duplicate:
                        prior_label, prior_file, prior_split = duplicate
                        current_label = label or row["category"]
                        if prior_label != current_label:
                            raise ValueError(f"Cross-label duplicate PCM: {prior_file} / {name}")
                        if prior_split != split:
                            raise ValueError(f"Derived fit/validation PCM leakage: {prior_file} / {name}")
                        duplicate_files.append(name)
                        continue
                    pcm_seen[pcm_hash] = (label or row["category"], name, split)
                    feature_rows.append(frontend(audio).astype(np.float16, copy=False))
                    targets.append(row["target"] if row["target"] is not None else -1)
                    metadata.append(row)
                except Exception as exc:
                    failures.append({"file_sha256": row["row_id_sha256"], "error": f"{type(exc).__name__}: {exc}"})
    if scanned != len(selected):
        raise ValueError(f"Matched only {scanned}/{len(selected)} rows for {split} feature extraction")
    if failures:
        raise ValueError(f"Audio extraction failures ({len(failures)}): {failures[:3]}")
    if not feature_rows:
        raise ValueError(f"No feature vectors materialized for {split}")
    x = np.stack(feature_rows).astype(np.float16, copy=False)
    y = np.asarray(targets, dtype=np.int64)
    save_name = "fit_features.npy" if split == "fit" else "validation_features.npy"
    save_y = "fit_targets.npy" if split == "fit" else "validation_targets.npy"
    np.save(RUN / save_name, x)
    np.save(RUN / save_y, y)
    stats = {"requested_rows": len(selected), "decoded_rows": scanned, "unique_pcm_rows": len(x), "duplicates_dropped": len(duplicate_files), "feature_shape": list(x.shape), "features_dtype": str(x.dtype)}
    (RUN / f"{split}_feature_audit.json").write_text(json.dumps(stats | {"duplicate_file_row_hashes": [sha256_bytes(n.encode()) for n in duplicate_files]}, indent=2) + "\n", encoding="utf-8")
    return x, y, metadata, stats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def initial_hash(model: nn.Module) -> str:
    return sha256_bytes(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))


@torch.inference_mode()
def logits_torch(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
        outputs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_model(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, run_dir: Path) -> dict[str, Any]:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDSCNN(classes=len(LABELS), channels=48, dropout=0.15).to(device)
    start_hash = initial_hash(model)
    counts = np.bincount(y, minlength=len(LABELS))
    if len(counts) != len(LABELS) or np.any(counts == 0):
        raise ValueError(f"Missing fit classes: {[LABELS[i] for i in np.flatnonzero(counts == 0)]}")
    weights = np.sqrt(len(y) / counts.astype(np.float64))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=3e-5)
    best_f1 = -1.0
    best_weights = None
    best_epoch = 0
    patience_count = 0
    history: list[dict[str, Any]] = []
    t_total = time.perf_counter()
    print(f"Initialized TinyDSCNN-48 from random weights on {device}; {len(LABELS)} command leaves; {len(y):,} fitting clips; {int((vy < 0).sum()):,} validation reject clips.", flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = np.random.permutation(len(y))
        total_loss = 0.0
        correct = 0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            xb_np = np.asarray(x[ids], dtype=np.float32).copy()
            yb = torch.from_numpy(y[ids]).to(device)
            xb = torch.from_numpy(xb_np).to(device)
            # Time/frequency masking and a short temporal roll mirror the established recipe.
            f_start = random.randrange(0, 36)
            t_start = random.randrange(0, 235)
            xb[:, :, f_start:f_start + 4, :] = 0.0
            xb[:, :, :, t_start:t_start + 15] = 0.0
            xb = torch.roll(xb, random.randint(-10, 10), dims=-1)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(ids)
            correct += int((logits.argmax(1) == yb).sum().item())
        scheduler.step()
        val_known = vy >= 0
        val_logits = logits_torch(model, vx[val_known], device)
        val_pred = val_logits.argmax(axis=1)
        val_acc = float(np.mean(val_pred == vy[val_known]))
        val_f1 = float(f1_score(vy[val_known], val_pred, labels=np.arange(len(LABELS)), average="macro", zero_division=0))
        is_best = val_f1 > best_f1 + 1e-7
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, run_dir / "best_model.pt")
            patience_count = 0
        else:
            patience_count += 1
        epoch_s = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch, "fit_loss": total_loss / len(y), "fit_accuracy": correct / len(y),
            "validation_supported_accuracy": val_acc, "validation_supported_macro_f1": val_f1,
            "seconds": epoch_s, "lr": float(optimizer.param_groups[0]["lr"]), "best": is_best,
        }
        history.append(record)
        print(f"Epoch {epoch:02d}/{EPOCHS} | fit loss {record['fit_loss']:.4f} acc {record['fit_accuracy']:.3%} | val acc {val_acc:.3%} macro-F1 {val_f1:.3%} | {epoch_s:.1f}s" + (" [BEST]" if is_best else ""), flush=True)
        if patience_count >= PATIENCE:
            print(f"Early stop after {PATIENCE} epochs without macro-F1 improvement.", flush=True)
            break
    if best_weights is None:
        raise RuntimeError("Training produced no selected checkpoint")
    model.load_state_dict(best_weights)
    model.eval()
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "training_summary.json").write_text(json.dumps({
        "model": "TinyDSCNN-48", "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": SEED, "initial_state_sha256": start_hash, "pretrained_weights_used": False,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1, "training_wall_seconds": time.perf_counter() - t_total,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "class_weight_formula": "sqrt(n_fit / class_count)", "label_smoothing": 0.03,
        "augmentation": "feature frequency mask 4 bands, time mask 15 frames, roll +/-10 frames",
        "fit_samples": len(y), "fit_counts": {LABELS[i]: int(counts[i]) for i in range(len(LABELS))},
    }, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "device": device, "best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "initial_state_sha256": start_hash, "epochs_completed": len(history)}


class CalibrationReader(CalibrationDataReader):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, count: int = 256):
        rng = np.random.default_rng(SEED)
        per_class = max(1, count // len(LABELS))
        selected: list[int] = []
        for label in range(len(LABELS)):
            ids = np.flatnonzero(train_y == label)
            selected.extend(rng.choice(ids, min(per_class, len(ids)), replace=False).tolist())
        if len(selected) < count:
            remaining = np.setdiff1d(np.arange(len(train_y)), np.asarray(selected, dtype=np.int64))
            selected.extend(rng.choice(remaining, min(count - len(selected), len(remaining)), replace=False).tolist())
        self.features = np.asarray(train_x[np.asarray(selected[:count])], dtype=np.float32)
        self.offset = 0

    def get_next(self):
        if self.offset >= len(self.features):
            return None
        batch = self.features[self.offset:self.offset + 1]
        self.offset += 1
        return {"input": batch}


def export_models(model: nn.Module, train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    models = RUN / "models"
    models.mkdir(parents=True, exist_ok=True)
    fp32 = models / "intent_gold_fp32.onnx"
    int8 = models / "intent_gold_int8.onnx"
    model = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(model, torch.zeros((1, *FEATURE_SHAPE), dtype=torch.float32), str(fp32),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))
    reader = CalibrationReader(train_x, train_y)
    quantize_static(str(fp32), str(int8), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    onnx.checker.check_model(onnx.load(str(int8)))
    if fp32.stat().st_size >= 500_000 or int8.stat().st_size >= 500_000:
        raise ValueError("Intent model export exceeds the 500 KB budget")
    return {
        "fp32_path": str(fp32), "int8_path": str(int8),
        "fp32_bytes": fp32.stat().st_size, "int8_bytes": int8.stat().st_size,
        "fp32_sha256": sha256_file(fp32), "int8_sha256": sha256_file(int8),
        "calibration_count": len(reader.features), "calibration_source": "gold train fitting rows only",
        "classes": LABELS, "output_shape": ["batch", len(LABELS)],
    }


def ort_logits(model_path: Path, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPUExecutionProvider not available")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    expected_classes = 2 if Path(model_path).resolve() == ACTIVE_WAKE.resolve() else len(LABELS)
    if tuple(output_meta.shape[1:]) != (expected_classes,):
        raise ValueError(f"Bad output shape for {model_path}: {output_meta.shape}")
    chunks = []
    for start in range(0, len(features), batch_size):
        x = np.asarray(features[start:start + batch_size], dtype=np.float32)
        result = session.run(None, {input_meta.name: x})[0]
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite logits from {model_path}")
        chunks.append(result)
    return np.concatenate(chunks, axis=0)


def probabilities(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def score_gate(probs: np.ndarray, targets: np.ndarray, confidence: float, margin: float) -> dict[str, Any]:
    order = np.argsort(probs, axis=1)[:, ::-1]
    pred = order[:, 0]
    conf = probs[np.arange(len(probs)), pred]
    diff = conf - probs[np.arange(len(probs)), order[:, 1]]
    accept = (conf >= confidence) & (diff >= margin)
    known = targets >= 0
    unknown = ~known
    correct = accept & known & (pred == targets)
    wrong_known = accept & known & (pred != targets)
    false_action = accept & unknown
    return {
        "confidence_threshold": float(confidence), "margin_threshold": float(margin),
        "known_support": int(known.sum()), "known_correct_actions": int(correct.sum()),
        "known_wrong_actions": int(wrong_known.sum()), "known_rejected": int((known & ~accept).sum()),
        "known_correct_action_coverage": float(correct.sum() / max(int(known.sum()), 1)),
        "known_action_accuracy": float(correct.sum() / max(int((correct | wrong_known).sum()), 1)),
        "unknown_support": int(unknown.sum()), "unknown_false_actions": int(false_action.sum()),
        "unknown_false_action_rate": float(false_action.sum() / max(int(unknown.sum()), 1)),
        "unknown_rejected": int((unknown & ~accept).sum()),
    }


def choose_gate(probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    unknown_count = int(np.sum(targets < 0))
    if unknown_count < 10:
        raise ValueError(f"Too few validation reject examples to calibrate safe action gate: {unknown_count}")
    max_false = math.floor(unknown_count * FALSE_ACTION_LIMIT + 1e-9)
    options = []
    for confidence in REJECT_GRID_CONFIDENCE:
        for margin in REJECT_GRID_MARGIN:
            metric = score_gate(probs, targets, float(confidence), float(margin))
            if metric["unknown_false_actions"] <= max_false:
                options.append(metric)
    if not options:
        raise RuntimeError("No candidate gate satisfies the predeclared validation OOS false-action bound")
    # Maximize correct in-scope actions; ties avoid wrong actions, then prefer
    # lower supported rejection, then the stricter OOS rate, then a lower gate.
    options.sort(key=lambda x: (
        x["known_correct_actions"], -x["known_wrong_actions"], -x["unknown_false_actions"],
        -x["known_rejected"], -x["confidence_threshold"], -x["margin_threshold"],
    ), reverse=True)
    best = options[0]
    best["selection_rule"] = "maximize correctly accepted supported validation commands with <=1% validation false actions on true OOS plus unsupported-slot speech; ties minimize wrong actions then maximize known acceptance"
    best["validation_false_action_limit"] = FALSE_ACTION_LIMIT
    return best


def lock_candidate(candidate_export: dict[str, Any], split_summary: dict[str, Any], gate: dict[str, Any], baseline_gate: dict[str, Any], val_scores: dict[str, Any]) -> dict[str, Any]:
    import tinyvcm_model.frontend as frontend_module
    import tinyvcm_model.model as model_module
    data_manifest = RUN / "train_validation_manifest.csv"
    lock = {
        "status": "frozen_before_gold_test_access",
        "created_at_utc": utc_now(), "dataset_name": "ME2 Spoken Command Dataset", "dataset_revision": REVISION,
        "run_name": RUN_NAME, "candidate_intent_sha256": candidate_export["int8_sha256"],
        "baseline_intent_sha256": sha256_file(ACTIVE_INTENT), "wake_sha256": sha256_file(ACTIVE_WAKE),
        "frontend_sha256": sha256_file(Path(frontend_module.__file__)), "model_definition_sha256": sha256_file(Path(model_module.__file__)),
        "training_script_sha256": sha256_file(Path(__file__)), "split_manifest_sha256": sha256_file(data_manifest),
        "class_order": LABELS, "frontend": {"sample_rate_hz": SR, "window_samples": SAMPLES, "feature_shape": FEATURE_SHAPE, "mels": MELS},
        "candidate_gate": {"confidence": gate["confidence_threshold"], "margin": gate["margin_threshold"]},
        "baseline_validation_gate": {"confidence": baseline_gate["confidence_threshold"], "margin": baseline_gate["margin_threshold"]},
        "baseline_deployed_gate": {"confidence": ACTIVE_CONFIDENCE, "margin": ACTIVE_MARGIN},
        "wake_threshold": 0.95,
        "wake_training": "not possible from this command dataset because it has no wake positives; exact active binary model retained",
        "quantization_calibration": candidate_export["calibration_source"],
        "validation_scores": val_scores,
        "test_accessed": False,
        "threshold_policy": "frozen one-shot comparison; confidence+margin gates calibrated on group-held-out gold train validation only",
    }
    path = RUN / "evaluation_lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def fit_candidate() -> dict[str, Any]:
    if RUN.exists() and any(RUN.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a prior run: {RUN}")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "DATA_USE_NOTE.md").write_text(
        "# Data-use note\n\nThis local candidate uses the ME2 Spoken Command Dataset at pinned revision " + REVISION + ". It is retained for the owner’s private coursework/deployment evaluation only. Source terms and participant permissions remain unresolved in the dataset audit, so do not publish the raw dataset or redistribute this model/package until those rights are checked. No audio is included in the Raspberry Pi package.\n",
        encoding="utf-8",
    )
    rows, split_summary = build_split_manifest()
    frontend = Frontend()
    split_pcm_hashes: dict[str, tuple[str, str, str]] = {}
    fit_x, fit_y, fit_meta, fit_stats = materialize_features(rows, "fit", frontend, split_pcm_hashes)
    val_x, val_y, val_meta, val_stats = materialize_features(rows, "validation", frontend, split_pcm_hashes)
    fit_keep = fit_y >= 0
    if not np.all(fit_keep):
        raise AssertionError("Fitting feature materialization must contain supported commands only")
    (RUN / "feature_materialization.json").write_text(json.dumps({"fit": fit_stats, "validation": val_stats}, indent=2) + "\n", encoding="utf-8")
    train_result = train_model(fit_x, fit_y, val_x, val_y, RUN)
    export_info = export_models(train_result["model"], fit_x, fit_y)
    (RUN / "export_summary.json").write_text(json.dumps(export_info, indent=2) + "\n", encoding="utf-8")
    candidate_logits = ort_logits(Path(export_info["int8_path"]), val_x)
    active_logits = ort_logits(ACTIVE_INTENT, val_x)
    candidate_probs = probabilities(candidate_logits)
    baseline_probs = probabilities(active_logits)
    candidate_gate = choose_gate(candidate_probs, val_y)
    baseline_gate = choose_gate(baseline_probs, val_y)
    known = val_y >= 0
    val_scores = {
        "candidate_raw_supported_accuracy": float(np.mean(candidate_probs[known].argmax(1) == val_y[known])),
        "candidate_raw_supported_macro_f1": float(f1_score(val_y[known], candidate_probs[known].argmax(1), labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
        "candidate_gate": candidate_gate,
        "baseline_raw_supported_accuracy": float(np.mean(baseline_probs[known].argmax(1) == val_y[known])),
        "baseline_raw_supported_macro_f1": float(f1_score(val_y[known], baseline_probs[known].argmax(1), labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
        "baseline_validation_calibrated_gate": baseline_gate,
        "baseline_deployed_gate": score_gate(baseline_probs, val_y, ACTIVE_CONFIDENCE, ACTIVE_MARGIN),
        "validation_rows": int(len(val_y)), "supported_rows": int(known.sum()),
        "true_oos_rows": int(sum(row["category"] == "true_oos" for row in val_meta)),
        "unsupported_slot_rows": int(sum(row["category"] == "unsupported_slot" for row in val_meta)),
    }
    lock = lock_candidate(export_info, split_summary, candidate_gate, baseline_gate, val_scores)
    package_meta = {
        "dataset_name": "ME2 Spoken Command Dataset", "dataset_revision": REVISION,
        "source_run": RUN_NAME, "status": "frozen_candidate_not_promoted",
        "pretrained_weights_used": False, "wake_model_training": "unchanged active binary wake weights, no wake positives in source dataset",
        "intent_architecture": "TinyDSCNN-48, 31 supported leaves, initialized randomly",
        "intent_classes": LABELS, "reject_policy": "no-response gate from group-held-out validation; true OOS and unsupported slot tracked as distinct categories",
        "intent_confidence_threshold": candidate_gate["confidence_threshold"],
        "intent_margin_threshold": candidate_gate["margin_threshold"],
        "validation": val_scores, "intent_model": export_info,
        "wake_model": {"path": str(ACTIVE_WAKE), "sha256": lock["wake_sha256"], "threshold": 0.95, "training": "retained byte-identical"},
        "cautions": ["Published gold test was not used for fitting or gate selection before evaluation lock.",
                     "Gold numerals split is excluded.", "Participant/source permissions need follow-up before redistribution.",
                     "The test is a single comparison and must not be reused for model selection."],
    }
    (RUN / "candidate_metadata.json").write_text(json.dumps(package_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate frozen before test access: {RUN}", flush=True)
    print(f"Selected validation epoch: {train_result['best_epoch']}; candidate gate confidence={candidate_gate['confidence_threshold']:.2f}, margin={candidate_gate['margin_threshold']:.3f}", flush=True)
    print(f"Baseline validation gate confidence={baseline_gate['confidence_threshold']:.2f}, margin={baseline_gate['margin_threshold']:.3f}", flush=True)
    print("Gold test has not been read.", flush=True)
    return {"run": str(RUN), "training": train_result, "export": export_info, "validation": val_scores, "lock": lock}


def gates_from_probs(probs: np.ndarray, confidence: float, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probs, axis=1)[:, ::-1]
    pred = order[:, 0]
    conf = probs[np.arange(len(probs)), pred]
    diff = conf - probs[np.arange(len(probs)), order[:, 1]]
    accept = (conf >= confidence) & (diff >= margin)
    return pred, conf, diff


def basic_scores(targets: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(np.mean(targets == pred)),
        "macro_f1": float(f1_score(targets, pred, labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
        "per_label": classification_report(targets, pred, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, pred, labels=np.arange(len(LABELS))).tolist(),
    }


def top_level_label(label: str) -> str:
    for prefix, intent in (
        ("ALARM_", "ALARM"), ("BRIGHTNESS_", "BRIGHTNESS"),
        ("COLOR_", "COLOR"), ("CREATE_REMINDER_", "CREATE_REMINDER"),
        ("LIGHT_", "LIGHT"), ("TEMPERATURE_", "TEMPERATURE"),
        ("TIMER_", "TIMER"),
    ):
        if label.startswith(prefix):
            return intent
    return label


def cluster_bootstrap(targets: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, speaker_groups: list[str], reps: int = 500) -> dict[str, Any]:
    unique = sorted(set(speaker_groups))
    grouped = {g: np.flatnonzero(np.asarray(speaker_groups, dtype=object) == g) for g in unique}
    rng = np.random.default_rng(231)
    values = {name: {"accuracy": [], "macro_f1": []} for name in ("baseline", "candidate")}
    for _ in range(reps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([grouped[g] for g in chosen])
        for name, pred in (("baseline", baseline), ("candidate", candidate)):
            values[name]["accuracy"].append(float(np.mean(targets[ix] == pred[ix])))
            values[name]["macro_f1"].append(float(f1_score(targets[ix], pred[ix], labels=np.arange(len(LABELS)), average="macro", zero_division=0)))
    return {name: {metric: {"lower_95": float(np.percentile(vals, 2.5)), "upper_95": float(np.percentile(vals, 97.5)), "cluster_count": len(unique), "replicates": reps, "unit": "source-speaker cluster"} for metric, vals in by_metric.items()} for name, by_metric in values.items()}


def materialize_test_audio() -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = read_rows("test", include_audio=True)
    if len(rows) != 4418:
        raise ValueError(f"Frozen test row count changed: {len(rows)}")
    frontend = Frontend()
    feats = []
    details = []
    seen = {}
    for row in rows:
        category, leaf = classify_row(row)
        audio, pcm_hash = decode_audio(row)
        prior = seen.get(pcm_hash)
        identity = leaf if leaf else category
        if prior and prior != identity:
            raise ValueError("Frozen test contains a cross-label duplicate PCM not covered by the prior audit")
        seen[pcm_hash] = identity
        feats.append(frontend(audio).astype(np.float32, copy=False))
        details.append({
            "file": row["file"], "category": category, "label": leaf,
            "target": LABELS.index(leaf) if leaf in LABELS else -1,
            "speaker_group": group_id(row), "source": row["source"],
            "is_synthetic": int(row.get("is_synthetic") or 0), "accent_group": row.get("accent_group") or "",
            "variation": row.get("variation") or "", "transcript_source": row.get("transcript_source") or "",
        })
    return np.stack(feats), details


def score_test_once() -> dict[str, Any]:
    lock_path = RUN / "evaluation_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Candidate evaluation lock missing; fit and freeze the candidate first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen_before_gold_test_access" or lock.get("test_accessed"):
        raise RuntimeError("Candidate lock is not in a scoreable pre-test state")
    if lock.get("test_harness_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Test harness changed after the frozen paired-evaluation lock")
    if sha256_file(RUN / "models" / "intent_gold_int8.onnx") != lock["candidate_intent_sha256"]:
        raise RuntimeError("Candidate model changed after threshold lock")
    if sha256_file(ACTIVE_INTENT) != lock["baseline_intent_sha256"] or sha256_file(ACTIVE_WAKE) != lock["wake_sha256"]:
        raise RuntimeError("Baseline/wake model changed since lock")
    access_path = RUN / "gold_test_access_started.json"
    if access_path.exists():
        raise RuntimeError("Gold test was already opened. Do not rerun or tune on it.")
    access_path.write_text(json.dumps({"test_access_started_at_utc": utc_now(), "reason": "one-time paired final comparison", "run": RUN_NAME, "lock_sha256": sha256_file(lock_path)}, indent=2) + "\n", encoding="utf-8")
    print("One-shot lock written. Reading frozen gold test now; this split will not be reused for tuning.", flush=True)
    x, details = materialize_test_audio()
    y_all = np.asarray([item["target"] for item in details], dtype=np.int64)
    known = y_all >= 0
    if len(x) != 4418 or int(known.sum()) != 4371:
        raise ValueError(f"Unexpected published test composition: rows={len(x)}, supported={int(known.sum())}")
    candidate_path = RUN / "models" / "intent_gold_int8.onnx"
    base_logits = ort_logits(ACTIVE_INTENT, x)
    candidate_logits = ort_logits(candidate_path, x)
    base_probs = probabilities(base_logits)
    candidate_probs = probabilities(candidate_logits)
    base_pred = base_probs.argmax(1)
    candidate_pred = candidate_probs.argmax(1)
    base_gate = lock["baseline_validation_gate"]
    candidate_gate = lock["candidate_gate"]
    base_pred_current, base_conf_current, base_margin_current = gates_from_probs(base_probs, ACTIVE_CONFIDENCE, ACTIVE_MARGIN)
    base_accept_current = (base_conf_current >= ACTIVE_CONFIDENCE) & (base_margin_current >= ACTIVE_MARGIN)
    base_pred_cal, base_conf_cal, base_margin_cal = gates_from_probs(base_probs, base_gate["confidence"], base_gate["margin"])
    base_accept_cal = (base_conf_cal >= base_gate["confidence"]) & (base_margin_cal >= base_gate["margin"])
    cand_pred_cal, cand_conf_cal, cand_margin_cal = gates_from_probs(candidate_probs, candidate_gate["confidence"], candidate_gate["margin"])
    cand_accept_cal = (cand_conf_cal >= candidate_gate["confidence"]) & (cand_margin_cal >= candidate_gate["margin"])
    supported_indices = np.flatnonzero(known)
    base_raw_scores = basic_scores(y_all[known], base_pred[known])
    cand_raw_scores = basic_scores(y_all[known], candidate_pred[known])
    base_top = np.asarray([top_level_label(LABELS[i]) for i in base_pred[known]])
    cand_top = np.asarray([top_level_label(LABELS[i]) for i in candidate_pred[known]])
    target_top = np.asarray([top_level_label(LABELS[i]) for i in y_all[known]])
    top_labels = sorted(set(target_top.tolist()))
    top_idx = {label: i for i, label in enumerate(top_labels)}
    top_y = np.asarray([top_idx[v] for v in target_top])
    base_top_y = np.asarray([top_idx[v] for v in base_top])
    cand_top_y = np.asarray([top_idx[v] for v in cand_top])
    test_noncommand = ~known

    def operational(pred: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
        correct = accepted & known & (pred == y_all)
        wrong_known = accepted & known & (pred != y_all)
        false_oos = accepted & test_noncommand
        return {
            "supported_support": int(known.sum()), "correct_actions": int(correct.sum()),
            "wrong_supported_actions": int(wrong_known.sum()), "supported_rejects": int((known & ~accepted).sum()),
            "correct_action_coverage": float(correct.sum() / known.sum()),
            "accepted_supported_accuracy": float(correct.sum() / max(int((correct | wrong_known).sum()), 1)),
            "true_oos_support": int(sum(x["category"] == "true_oos" for x in details)),
            "unsupported_slot_support": int(sum(x["category"] == "unsupported_slot" for x in details)),
            "noncommand_false_actions": int(false_oos.sum()),
            "noncommand_false_action_rate": float(false_oos.sum() / max(int(test_noncommand.sum()), 1)),
            "noncommand_rejected": int((test_noncommand & ~accepted).sum()),
        }

    wake_probs = ort_logits(ACTIVE_WAKE, x)
    wake_prob = probabilities(wake_probs)[:, 1]
    wake_false = wake_prob >= 0.95
    per_class = {}
    for idx, label in enumerate(LABELS):
        mask = known & (y_all == idx)
        per_class[label] = {
            "support": int(mask.sum()),
            "baseline_f1": base_raw_scores["per_label"][label]["f1-score"],
            "candidate_f1": cand_raw_scores["per_label"][label]["f1-score"],
            "baseline_recall": base_raw_scores["per_label"][label]["recall"],
            "candidate_recall": cand_raw_scores["per_label"][label]["recall"],
            "baseline_candidate_f1_delta_pp": 100.0 * (cand_raw_scores["per_label"][label]["f1-score"] - base_raw_scores["per_label"][label]["f1-score"]),
        }
    source_slices = {}
    for source in sorted({str(item["source"]) for item in details}):
        ids = np.asarray([i for i, item in enumerate(details) if item["source"] == source and item["target"] >= 0], dtype=np.int64)
        if len(ids):
            source_slices[source] = {
                "support": len(ids), "baseline_accuracy": float(np.mean(base_pred[ids] == y_all[ids])),
                "candidate_accuracy": float(np.mean(candidate_pred[ids] == y_all[ids])),
            }
    group_list = [details[i]["speaker_group"] for i in supported_indices]
    result = {
        "dataset_name": "ME2 Spoken Command Dataset", "dataset_revision": REVISION,
        "run_name": RUN_NAME, "created_at_utc": utc_now(), "evaluation_lock_sha256": sha256_file(lock_path),
        "test_rows": len(details), "supported_test_rows": int(known.sum()),
        "true_oos_test_rows": int(sum(x["category"] == "true_oos" for x in details)),
        "unsupported_slot_test_rows": int(sum(x["category"] == "unsupported_slot" for x in details)),
        "wake": {"model_sha256": lock["wake_sha256"], "threshold": 0.95,
                 "positives_in_dataset_test": 0, "negative_support": len(details),
                 "false_accepts": int(wake_false.sum()), "false_accept_rate": float(wake_false.mean()),
                 "positive_recall": None, "limitation": "No wake-positive examples are present in this dataset test."},
        "baseline": {
            "model_sha256": lock["baseline_intent_sha256"], "raw_supported": {"accuracy": base_raw_scores["accuracy"], "macro_f1": base_raw_scores["macro_f1"], "per_label": base_raw_scores["per_label"], "confusion_matrix": base_raw_scores["confusion_matrix"]},
            "top_level_19": {"labels": top_labels, "accuracy": float(np.mean(base_top_y == top_y)), "macro_f1": float(f1_score(top_y, base_top_y, labels=np.arange(len(top_labels)), average="macro", zero_division=0))},
            "deployed_gate": {"confidence": ACTIVE_CONFIDENCE, "margin": ACTIVE_MARGIN, "results": operational(base_pred_current, base_accept_current)},
            "validation_calibrated_gate": {"confidence": base_gate["confidence"], "margin": base_gate["margin"], "results": operational(base_pred_cal, base_accept_cal)},
        },
        "candidate": {
            "model_sha256": lock["candidate_intent_sha256"], "raw_supported": {"accuracy": cand_raw_scores["accuracy"], "macro_f1": cand_raw_scores["macro_f1"], "per_label": cand_raw_scores["per_label"], "confusion_matrix": cand_raw_scores["confusion_matrix"]},
            "top_level_19": {"labels": top_labels, "accuracy": float(np.mean(cand_top_y == top_y)), "macro_f1": float(f1_score(top_y, cand_top_y, labels=np.arange(len(top_labels)), average="macro", zero_division=0))},
            "validation_calibrated_gate": {"confidence": candidate_gate["confidence"], "margin": candidate_gate["margin"], "results": operational(cand_pred_cal, cand_accept_cal)},
        },
        "paired_deltas_percentage_points": {
            "candidate_minus_baseline_accuracy": (cand_raw_scores["accuracy"] - base_raw_scores["accuracy"]) * 100,
            "candidate_minus_baseline_macro_f1": (cand_raw_scores["macro_f1"] - base_raw_scores["macro_f1"]) * 100,
            "candidate_minus_baseline_top_level_macro_f1": (float(f1_score(top_y, cand_top_y, labels=np.arange(len(top_labels)), average="macro", zero_division=0)) - float(f1_score(top_y, base_top_y, labels=np.arange(len(top_labels)), average="macro", zero_division=0))) * 100,
        },
        "per_class": per_class,
        "source_slices": source_slices,
        "source_speaker_cluster_bootstrap_95_ci": cluster_bootstrap(y_all[known], base_pred[known], candidate_pred[known], group_list),
        "limits": ["Single frozen test comparison; do not tune or rerun against this split.",
                   "Group bootstrap intervals do not resolve source licensing or participant consent.",
                   "This offline replay has no live microphone timing or command-action side effects."],
    }
    out_json = RUN / "frozen_test_comparison.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (RUN / "test_predictions_private.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["row_id_sha256", "category", "true_label", "baseline_prediction", "baseline_confidence", "baseline_margin", "candidate_prediction", "candidate_confidence", "candidate_margin", "wake_probability", "source", "is_synthetic", "accent_group"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(details):
            writer.writerow({
                "row_id_sha256": sha256_bytes(item["file"].encode()), "category": item["category"], "true_label": item["label"] or "",
                "baseline_prediction": LABELS[int(base_pred[i])], "baseline_confidence": float(np.max(base_probs[i])),
                "baseline_margin": float(np.max(base_probs[i]) - np.partition(base_probs[i], -2)[-2]),
                "candidate_prediction": LABELS[int(candidate_pred[i])], "candidate_confidence": float(np.max(candidate_probs[i])),
                "candidate_margin": float(np.max(candidate_probs[i]) - np.partition(candidate_probs[i], -2)[-2]),
                "wake_probability": float(wake_prob[i]), "source": item["source"],
                "is_synthetic": item["is_synthetic"], "accent_group": item["accent_group"],
            })
    report_path = RUN / "COMPARISON_REPORT.md"
    report_path.write_text(render_report(result, lock), encoding="utf-8")
    access_path.write_text(json.dumps({"test_access_started_at_utc": json.loads(access_path.read_text(encoding="utf-8"))["test_access_started_at_utc"], "status": "completed_once", "completed_at_utc": utc_now(), "rows_scored": len(details), "result_sha256": sha256_file(out_json), "note": "Do not rerun or tune using this split."}, indent=2) + "\n", encoding="utf-8")
    lock["status"] = "test_accessed_single_pass"
    lock["test_accessed"] = True
    lock["test_access_completed_at_utc"] = utc_now()
    lock["test_result_sha256"] = sha256_file(out_json)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"), flush=True)
    return result


def render_report(result: dict[str, Any], lock: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"
    base = result["baseline"]
    cand = result["candidate"]
    lines = [
        "# Scratch Candidate vs Current VCM",
        "",
        f"**Evaluation:** one paired inference pass on {result['test_rows']:,} published test WAVs from {REVISION}. The test was opened only after the model and gates were frozen.",
        "",
        "## Why this dataset was not used before",
        "",
        "The earlier action stopped at source and split auditing because the dataset combined several source licenses, participant recordings and generated voices whose usage terms had not been fully verified. This run was explicitly requested for private coursework evaluation. The raw audio stays local and is not packaged to the Pi. Rights/consent still need confirmation before redistribution.",
        "",
        "## Method",
        "",
        f"- Dataset: **ME2 Spoken Command Dataset**, revision `{REVISION}`.",
        "- Fit only the 31 supported command/value labels from the official train partition. No ASR, pretrained weights or cloud service.",
        "- Validation: deterministic source/speaker-group split carved from train; true OOS and unsupported-slot speech kept as distinct reject cases for gate calibration.",
        "- Numeral-only partition excluded. Published holdout untouched.",
        "- Runtime candidate reuses the active binary wake model byte-identically. The dataset has no wake positives, so it cannot retrain wake.",
        "- Gates selected on validation by maximizing correct accepted commands while allowing at most 1% validation false actions on reject speech.",
        "- INT8 quantization calibrated from fitting rows only.",
        "",
        "## Paired intent results on supported commands",
        "",
        "| Model | Accuracy | Macro F1 | Top-level intent accuracy | Top-level macro F1 | Gate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current VCM, raw argmax | {pct(base['raw_supported']['accuracy'])} | {pct(base['raw_supported']['macro_f1'])} | {pct(base['top_level_19']['accuracy'])} | {pct(base['top_level_19']['macro_f1'])} | n/a |",
        f"| Dataset candidate, raw argmax | {pct(cand['raw_supported']['accuracy'])} | {pct(cand['raw_supported']['macro_f1'])} | {pct(cand['top_level_19']['accuracy'])} | {pct(cand['top_level_19']['macro_f1'])} | n/a |",
        f"| Current VCM, deployed gate | - | - | - | - | conf {ACTIVE_CONFIDENCE:.2f}, margin {ACTIVE_MARGIN:.2f} |",
        f"| Dataset candidate, validation gate | - | - | - | - | conf {lock['candidate_gate']['confidence']:.2f}, margin {lock['candidate_gate']['margin']:.3f} |",
        "",
        f"Candidate minus current: accuracy **{result['paired_deltas_percentage_points']['candidate_minus_baseline_accuracy']:+.2f} pp**, macro F1 **{result['paired_deltas_percentage_points']['candidate_minus_baseline_macro_f1']:+.2f} pp**.",
        "",
        "## Rejection and safety",
        "",
        f"- Gold test support: {result['supported_test_rows']:,} supported commands, {result['true_oos_test_rows']} true out-of-scope clips, {result['unsupported_slot_test_rows']} unsupported-slot clips.",
        f"- Current deployed gate false actions on OOS/unsupported speech: {base['deployed_gate']['results']['noncommand_false_actions']}/{result['test_rows'] - result['supported_test_rows']}.",
        f"- Candidate validation-selected gate false actions on OOS/unsupported speech: {cand['validation_calibrated_gate']['results']['noncommand_false_actions']}/{result['test_rows'] - result['supported_test_rows']}.",
        f"- Candidate gate correct-action coverage: {pct(cand['validation_calibrated_gate']['results']['correct_action_coverage'])}; accepted-command accuracy: {pct(cand['validation_calibrated_gate']['results']['accepted_supported_accuracy'])}.",
        "- Thresholds are operating gates, not statements that the model is 95% accurate.",
        "",
        "## Wake and edge package",
        "",
        f"- Wake threshold retained at 0.95. On {result['wake']['negative_support']:,} gold-test command/OOS clips, false wake accepts: {result['wake']['false_accepts']} ({pct(result['wake']['false_accept_rate'])}). No wake-positive test examples exist, so wake recall cannot be scored here.",
        f"- Candidate INT8 intent: {result['candidate']['model_sha256']}.",
        f"- Candidate ONNX intent size: {json.loads((RUN / 'export_summary.json').read_text(encoding='utf-8'))['int8_bytes']:,} bytes; paired with unchanged wake model. See export/ARM64 benchmark for total.",
        "- Pi installation uses a sibling folder and a separate loopback port; current `/home/dalmacio/Desktop/dandan` remains untouched.",
        "",
        "## Per-class comparison",
        "",
        "| Label | n | Current F1 | Candidate F1 | Delta pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        item = result["per_class"][label]
        lines.append(f"| `{label}` | {item['support']} | {item['baseline_f1']:.3%} | {item['candidate_f1']:.3%} | {item['baseline_candidate_f1_delta_pp']:+.2f} |")
    lines.extend(["", "## Decision notes", "", "A candidate is a measured improvement only if it improves the paired supported-command metrics without a material increase in OOS false actions. Per-class scores, reject coverage and user-specific audio still matter. If the candidate wins offline, its separate Pi launch is for hands-on evaluation; it has not replaced the current release.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score-test"))
    args = parser.parse_args()
    if not SNAPSHOT.is_dir():
        raise FileNotFoundError(f"Pinned dataset snapshot is missing: {SNAPSHOT}")
    if args.phase == "fit":
        fit_candidate()
    else:
        score_test_once()


if __name__ == "__main__":
    main()
