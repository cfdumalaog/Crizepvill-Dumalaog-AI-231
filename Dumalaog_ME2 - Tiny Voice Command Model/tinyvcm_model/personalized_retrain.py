"""Personalized wake/intent experiments using the locally recorded ME2 speech.

All models initialize from random weights. Raw recordings are deduplicated by
decoded PCM hash before splitting; no recording variant can cross a split.
The ME2 Spoken Command Dataset speaker-disjoint test split is opened only after training/export.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn

from .config import CHANNELS, FEATURE_SHAPE, LABELS, MELS, SAMPLES, SR, TIME_STEPS
from .frontend import Frontend, fit_audio
from .model import TinyDSCNN
from .recording_labels import LEGACY_INTENT_MAP, NON_INTENT_LABELS, map_recorded_intent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
INTENT_MAP = {label: label for label in LABELS}
INTENT_MAP.update(LEGACY_INTENT_MAP)
SEED = 231


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audio_digest(audio: np.ndarray) -> str:
    return _sha256(np.asarray(audio, dtype="<f4").reshape(-1).tobytes())


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_counts(n: int) -> tuple[int, int, int]:
    """Per-class 70/15/15 split; ensure all three splits where n >= 3."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    n_train = max(1, int(math.floor(n * 0.70)))
    n_val = max(1, int(math.floor(n * 0.15)))
    while n_train + n_val >= n:
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
    return n_train, n_val, n - n_train - n_val


def _split_items(items: list[dict[str, Any]], seed: int = SEED) -> None:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[item["label"]].append(item)
    for label, group in sorted(by_label.items()):
        # Hash ordering makes splits deterministic without depending on file-system order.
        group.sort(key=lambda x: _sha256(f"{seed}|{label}|{x['pcm_sha256']}".encode()))
        n_train, n_val, _ = _split_counts(len(group))
        for i, item in enumerate(group):
            item["split"] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")


def load_personal_records(project_root: Path = PROJECT_ROOT, split_output_path: Path | None = None, manifest_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate WAVs, globally deduplicate PCM, exclude cross-label collisions, split, featurize."""
    project_root = Path(project_root).resolve()
    manifest = Path(manifest_path) if manifest_path is not None else project_root / "data" / "human" / "manifest.csv"
    rows = _read_manifest(manifest)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        wav_path = (project_root / row["path"]).resolve()
        if not wav_path.is_relative_to(project_root) or not wav_path.is_file():
            raise ValueError(f"Invalid/missing human recording path in manifest: {row['path']}")
        audio, rate = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SR or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio format in {row['path']}: rate={rate}, samples={audio.size}")
        if len(audio) > SAMPLES:
            raise ValueError(f"Audio is longer than the model window: {row['path']}")
        by_hash[_audio_digest(audio)].append({**row, "full_path": str(wav_path), "audio": audio})

    collisions = []
    groups = []
    duplicate_rows = 0
    for digest, same_audio in by_hash.items():
        labels = sorted({r["label"] for r in same_audio})
        if len(labels) != 1:
            collisions.append({
                "pcm_sha256": digest,
                "labels": labels,
                "paths": [r["path"] for r in same_audio],
                "action": "excluded_all_conflicting_rows_from_model_splits; original_audio_and_manifest_preserved",
            })
            continue
        duplicate_rows += len(same_audio) - 1
        chosen = sorted(same_audio, key=lambda r: (r["condition"], r["path"]))[0]
        groups.append({
            "label": chosen["label"],
            "intent": map_recorded_intent(chosen["label"]),
            "speaker": chosen["speaker"],
            "condition": chosen["condition"],
            "path": chosen["path"],
            "source_id": chosen["source_id"],
            "duplicate_count": len(same_audio),
            "duplicate_conditions": sorted({r["condition"] for r in same_audio}),
            "pcm_sha256": digest,
            "full_path": chosen["full_path"],
            "audio": chosen["audio"],
        })
    _split_items(groups)
    frontend = Frontend()
    for item in groups:
        item["feature"] = frontend(fit_audio(item["audio"]))

    columns = ["path", "speaker", "condition", "label", "mapped_intent", "source_id", "pcm_sha256", "duplicate_count", "split"]
    if split_output_path is not None:
        split_output_path = Path(split_output_path)
        split_output_path.parent.mkdir(parents=True, exist_ok=True)
        with split_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for item in sorted(groups, key=lambda x: (x["label"], x["pcm_sha256"])):
                writer.writerow({
                    "path": item["path"], "speaker": item["speaker"], "condition": item["condition"],
                    "label": item["label"], "mapped_intent": item["intent"] or "", "source_id": item["source_id"],
                    "pcm_sha256": item["pcm_sha256"], "duplicate_count": item["duplicate_count"], "split": item["split"],
                })
    audit = {
        "raw_manifest_rows": len(rows), "unique_pcm_groups": len(groups), "exact_duplicate_rows_collapsed": duplicate_rows,
        "cross_label_hash_collisions": collisions, "cross_label_collision_groups_excluded": len(collisions),
        "speakers": sorted({x["speaker"] for x in groups}),
        "condition_groups": dict(sorted(Counter(x["condition"] for x in groups).items())),
        "raw_label_rows": dict(sorted(Counter(x["label"] for x in rows).items())),
        "unique_label_groups": dict(sorted(Counter(x["label"] for x in groups).items())),
        "unique_split_counts": dict(sorted(Counter(x["split"] for x in groups).items())),
        "intent_mapping": INTENT_MAP,
        "unique_mapped_intents": dict(sorted(Counter(x["intent"] for x in groups if x["intent"]).items())),
        "unmapped_labels": dict(sorted(Counter(x["label"] for x in groups if x["intent"] is None and x["label"] not in NON_INTENT_LABELS).items())),
        "manifest_sha256": _sha256(manifest.read_bytes()),
        "split_manifest_path": str(split_output_path) if split_output_path is not None else None,
        "split_policy": "content-deduplicated, deterministic per-label 70/15/15; all three partitions only when n>=3; speakers may occur in multiple partitions, so clip holdout is not speaker holdout",
    }
    return groups, audit


def load_optionb_train_val(project_root: Path = PROJECT_ROOT) -> dict[str, np.ndarray]:
    """Load only train/validation feature arrays; never opens the reference test arrays."""
    project_root = Path(project_root)
    manifest = _read_manifest(project_root / "data" / "option_b" / "manifest.csv")
    expected = Counter(r["split"] for r in manifest)
    cache_path = project_root / "data" / "option_b" / "features_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        data = {k: cache[k] for k in ("train_x", "train_y", "val_x", "val_y")}
    data["train_y"] = data["train_y"].astype(np.int64, copy=False)
    data["val_y"] = data["val_y"].astype(np.int64, copy=False)
    if data["train_x"].shape[0] != expected["train"] or data["val_x"].shape[0] != expected["val"]:
        raise ValueError(f"Reference feature cache does not match manifest split counts: {expected}")
    if data["train_x"].shape[1:] != FEATURE_SHAPE or data["val_x"].shape[1:] != FEATURE_SHAPE:
        raise ValueError("Unexpected reference feature shape")
    data["train_x"] = data["train_x"].astype(np.float32, copy=False)
    data["val_x"] = data["val_x"].astype(np.float32, copy=False)
    data["manifest_sha256"] = np.asarray(_sha256((project_root / "data" / "option_b" / "manifest.csv").read_bytes()))
    return data


def load_optionb_test(project_root: Path = PROJECT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Open the reused reference test partition only in the post-training evaluation stage."""
    cache_path = Path(project_root) / "data" / "option_b" / "features_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        x = cache["test_x"].astype(np.float32, copy=False)
        y = cache["test_y"].astype(np.int64, copy=False)
    return x, y


def _feature_matrix(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.empty((0, *FEATURE_SHAPE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack([i["feature"] for i in items]).astype(np.float32), np.asarray([LABELS.index(i["intent"]) for i in items], dtype=np.int64)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def _intent_metrics(y: np.ndarray, pred: np.ndarray, label_names: list[str] = LABELS) -> dict[str, Any]:
    present = np.unique(y)
    return {
        "count": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_supported_classes": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "classification_report": classification_report(y, pred, labels=present, target_names=[label_names[i] for i in present], output_dict=True, zero_division=0),
        "per_class_support": {label_names[i]: int(np.sum(y == i)) for i in present},
    }


def _eval_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    with torch.inference_mode():
        for i in range(0, len(y), batch_size):
            logits.append(model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu().numpy())
    raw = np.concatenate(logits, axis=0) if logits else np.empty((0, len(LABELS)), dtype=np.float32)
    return raw.argmax(axis=1), raw


def train_intent_candidate(data: dict[str, Any], output_dir: Path, epochs: int = 35, batch_size: int = 64, personal_per_batch: int = 3, seed: int = SEED, recipe: str = 'standard') -> dict[str, Any]:
    """Train the 31-class intent model from scratch with a modest personal-audio mix."""
    import torch.nn.functional as F

    output_dir = Path(output_dir)
    if recipe not in ('standard', 'refined'):
        raise ValueError(f'Unknown intent training recipe: {recipe}')
    output_dir.mkdir(parents=True, exist_ok=False)
    base_x, base_y = data["train_x"], data["train_y"]
    intent_train = [i for i in data["human"] if i["split"] == "train" and i["intent"]]
    intent_val = [i for i in data["human"] if i["split"] == "validation" and i["intent"]]
    intent_test = [i for i in data["human"] if i["split"] == "test" and i["intent"]]
    user_x, user_y = _feature_matrix(intent_train)
    val_user_x, val_user_y = _feature_matrix(intent_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = TinyDSCNN(classes=len(LABELS), channels=CHANNELS, dropout=0.10 if recipe == 'refined' else 0.15).to(device)
    init_hash = _sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()))
    base_counts = np.bincount(base_y, minlength=len(LABELS))
    class_weights = torch.tensor(np.sqrt(len(base_y) / np.maximum(base_counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0 if recipe == 'refined' else 0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    steps = math.ceil(len(base_y) / batch_size)
    rng = np.random.default_rng(seed)
    print(f"Intent: scratch TinyDSCNN-48, 31 classes, {sum(p.numel() for p in model.parameters()):,} parameters, device={device}", flush=True)
    print(f"Training pool: ME2 Spoken Command Dataset {len(base_y):,} + personal {len(user_y)} unique clips; {personal_per_batch}/{batch_size} personal draws per treatment batch; no test samples opened", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(base_y))
        cursor = 0
        epoch_loss = 0.0
        epoch_correct = 0
        seen = 0
        for _ in range(steps):
            n_base = batch_size - (personal_per_batch if len(user_y) else 0)
            if cursor + n_base > len(order):
                remaining = order[cursor:]
                order = rng.permutation(len(base_y))
                need = n_base - len(remaining)
                ids = np.concatenate((remaining, order[:need]))
                cursor = need
            else:
                ids = order[cursor:cursor + n_base]
                cursor += n_base
            bx = base_x[ids]
            by = base_y[ids]
            if len(user_y):
                ui = rng.integers(0, len(user_y), size=personal_per_batch)
                bx = np.concatenate((bx, user_x[ui]), axis=0)
                by = np.concatenate((by, user_y[ui]), axis=0)
            perm = rng.permutation(len(by))
            xb = torch.from_numpy(bx[perm].copy()).to(device)
            yb = torch.from_numpy(by[perm].copy()).to(device)
            # Match the existing training recipe: SpecAugment and small temporal jitter.
            for row in range(len(xb)):
                fw = int(torch.randint(0, 4 if recipe == 'refined' else 5, (1,), device=device).item())
                tw = int(torch.randint(0, 10 if recipe == 'refined' else 16, (1,), device=device).item())
                if fw:
                    fs = int(torch.randint(0, MELS - fw + 1, (1,), device=device).item())
                    xb[row, :, fs:fs + fw, :] = 0
                if tw:
                    ts = int(torch.randint(0, TIME_STEPS - tw + 1, (1,), device=device).item())
                    xb[row, :, :, ts:ts + tw] = 0
            shift_limit = 6 if recipe == 'refined' else 10
            shift = int(torch.randint(-shift_limit, shift_limit + 1, (1,), device=device).item())
            xb = torch.roll(xb, shifts=shift, dims=-1)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(yb)
            epoch_correct += int((logits.argmax(1) == yb).sum().item())
            seen += len(yb)
        scheduler.step()

        val_base_pred, _ = _eval_torch(model, data["val_x"], data["val_y"], device)
        val_base_f1 = float(f1_score(data["val_y"], val_base_pred, labels=np.arange(len(LABELS)), average="macro", zero_division=0))
        val_class_f1 = f1_score(data["val_y"], val_base_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
        val_min_f1 = float(np.min(val_class_f1))
        if len(val_user_y):
            val_user_pred, _ = _eval_torch(model, val_user_x, val_user_y, device)
            val_user_f1 = _intent_metrics(val_user_y, val_user_pred)["macro_f1_supported_classes"]
        else:
            val_user_f1 = val_base_f1
        score = (0.7 * val_base_f1 + 0.2 * val_min_f1 + 0.1 * val_user_f1) if recipe == 'refined' else (0.8 * val_base_f1 + 0.2 * val_user_f1)
        entry = {
            "epoch": epoch, "train_loss": epoch_loss / max(1, seen), "train_accuracy": epoch_correct / max(1, seen),
            "val_optionb_macro_f1": val_base_f1, "val_optionb_min_class_f1": val_min_f1,
            "val_personal_macro_f1": val_user_f1,
            "checkpoint_score": score, "epoch_seconds": time.perf_counter() - t0,
        }
        history.append(entry)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pt")
        _save_json(output_dir / "history.json", {"history": history})
        print(f"Intent epoch {epoch:02d}/{epochs} loss={entry['train_loss']:.4f} train_acc={entry['train_accuracy']:.3f} val_reference_macroF1={val_base_f1:.3f} val_personal_macroF1={val_user_f1:.3f} score={score:.3f}{' [BEST]' if best_epoch == epoch else ''}", flush=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    run = {
        "objective": "31-class ME2 Spoken Command Dataset intent model, scratch initialization plus recorded-personal training subset",
        "recipe": recipe,
        "labels": LABELS, "pretrained_weights_used": False, "seed": seed, "initial_state_sha256": init_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device),
        "epochs": epochs, "best_epoch": best_epoch, "best_validation_score": best_score,
        "base_training_clips_per_epoch": len(base_y), "personal_unique_train_clips": len(user_y),
        "personal_draws_per_batch": personal_per_batch, "optimizer_steps_per_epoch": steps,
        "personal_split_counts": dict(Counter(i["split"] for i in data["human"] if i["intent"])),
        "intent_mapping": INTENT_MAP, "optionb_manifest_sha256": str(data["manifest_sha256"].item()),
        "personal_manifest_sha256": data["human_audit"]["manifest_sha256"],
        "personal_speaker_ids": data["human_audit"]["speakers"],
        "personal_split_policy": data["human_audit"]["split_policy"],
        "limit": f"Personal recordings cover {len(data['human_audit']['speakers'])} speaker IDs ({', '.join(data['human_audit']['speakers'])}); the per-label clip split may include each speaker in train, validation, and test, so no unseen-speaker claim is supported.",
    }
    _save_json(output_dir / "run_info.json", run)
    return {"output_dir": output_dir, "model": model, "run_info": run, "history": history,
            "personal_train_x": user_x, "personal_train_y": user_y,
            "personal_val_x": val_user_x, "personal_val_y": val_user_y,
            "personal_test_items": intent_test}


def _context_windows(audio: np.ndarray, frontend: Frontend, count: int = 5, rng: np.random.Generator | None = None, noises: list[np.ndarray] | None = None, augment: bool = False) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > SAMPLES:
        audio = fit_audio(audio)
    max_start = max(0, SAMPLES - len(audio))
    windows = []
    for j in range(count):
        if augment and rng is not None:
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
        else:
            start = int(np.linspace(0, max_start, count, dtype=int)[j])
        window = np.zeros(SAMPLES, dtype=np.float32)
        window[start:start + len(audio)] = audio
        if augment and rng is not None:
            window *= float(rng.uniform(0.75, 1.25))
            if noises and rng.random() < 0.8:
                noise = noises[int(rng.integers(0, len(noises)))].reshape(-1)
                if len(noise) < SAMPLES:
                    noise = np.tile(noise, int(np.ceil(SAMPLES / max(len(noise), 1))))
                offset = int(rng.integers(0, len(noise) - SAMPLES + 1))
                noise = noise[offset:offset + SAMPLES]
                sig_power = float(np.mean(window ** 2)) + 1e-9
                noise_power = float(np.mean(noise ** 2)) + 1e-9
                snr = float(rng.uniform(18, 32))
                window += noise * np.sqrt(sig_power / (noise_power * 10 ** (snr / 10)))
        windows.append(frontend(np.clip(window, -1, 1)))
    return np.stack(windows).astype(np.float32)


def _predict_torch_positive(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[i:i + batch_size]).to(device))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _threshold_for_zero_false(wake_scores: np.ndarray, nonwake_scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.concatenate([wake_scores, nonwake_scores, np.asarray([1.0], dtype=np.float32)]))
    best = (-1.0, -1, 1.0, 0)
    for threshold in candidates:
        recall = float(np.mean(wake_scores >= threshold)) if len(wake_scores) else 0.0
        false_count = int(np.sum(nonwake_scores >= threshold))
        if false_count == 0 and (recall, float(threshold)) > (best[0], best[2]):
            best = (recall, -false_count, float(threshold), false_count)
    return best[2], {"wake_recall": best[0], "false_accepts": best[3], "wake_support": int(len(wake_scores)), "nonwake_support": int(len(nonwake_scores)), "selection_constraint": "zero false accepts on validation windows"}


def _predict_onnx(path: Path, x: np.ndarray, binary: bool = False, batch_size: int = 256) -> np.ndarray:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = []
    for i in range(0, len(x), batch_size):
        out.append(session.run(None, {name: x[i:i + batch_size].astype(np.float32, copy=False)})[0])
    logits = np.concatenate(out) if out else np.empty((0, 2 if binary else len(LABELS)), dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _export_intent_candidate(run: dict[str, Any], data: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from torch.onnx import export as torch_onnx_export

    out = Path(run["output_dir"])
    models = out / "models"
    models.mkdir(exist_ok=True)
    fp32 = models / "intent_personalized_fp32.onnx"
    int8 = models / "intent_personalized_int8.onnx"
    model = run["model"].cpu().eval()
    torch_onnx_export(model, torch.randn(1, *FEATURE_SHAPE), str(fp32), input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))

    # Static quantization uses training-only calibration, never validation/test samples.
    rng = np.random.default_rng(seed)
    chosen = []
    per_class = max(1, 192 // len(LABELS))
    for label in range(len(LABELS)):
        ids = np.flatnonzero(data["train_y"] == label)
        take = min(per_class, len(ids))
        chosen.extend(data["train_x"][rng.choice(ids, take, replace=False)])
    need_personal = 256 - len(chosen)
    if need_personal > 0 and len(run["personal_train_y"]):
        ids = rng.integers(0, len(run["personal_train_y"]), size=need_personal)
        chosen.extend(run["personal_train_x"][ids])
    if len(chosen) < 256:
        ids = rng.integers(0, len(data["train_y"]), size=256 - len(chosen))
        chosen.extend(data["train_x"][ids])
    calib = np.stack(chosen[:256]).astype(np.float32)

    class Reader(CalibrationDataReader):
        def __init__(self, samples: np.ndarray):
            self.samples = iter([{"input": x[None]} for x in samples])
        def get_next(self):
            return next(self.samples, None)
    quantize_static(str(fp32), str(int8), Reader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    onnx.checker.check_model(onnx.load(str(int8)))
    summary = {
        "fp32_path": str(fp32), "int8_path": str(int8), "fp32_size_bytes": fp32.stat().st_size,
        "int8_size_bytes": int8.stat().st_size, "calibration_count": len(calib),
        "calibration_policy": "256 representative features selected only from ME2 Spoken Command Dataset train and mapped personal training clips",
        "fp32_sha256": _sha256(fp32.read_bytes()), "int8_sha256": _sha256(int8.read_bytes()),
        "classes": LABELS, "parameters": sum(p.numel() for p in model.parameters()),
    }
    _save_json(models / "export_summary.json", summary)
    return summary


def _load_external_noise(project_root: Path) -> list[np.ndarray]:
    import scipy.signal
    result = []
    for path in sorted((project_root / "data" / "dataset" / "_background_noise_").glob("*.wav")):
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SR:
            gcd = math.gcd(SR, int(rate))
            audio = scipy.signal.resample_poly(audio, SR // gcd, rate // gcd).astype(np.float32)
        if len(audio):
            result.append(audio)
    return result


def _wake_context_item(item: dict[str, Any], frontend: Frontend, rng: np.random.Generator, noises: list[np.ndarray] | None, augment: bool) -> np.ndarray:
    return _context_windows(item["audio"], frontend, count=1 if augment else 5, rng=rng, noises=noises, augment=augment)


def _evaluate_binary_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(y), batch):
            chunks.append(torch.softmax(model(torch.from_numpy(x[i:i + batch]).to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def train_wake_model(data: dict[str, Any], output_dir: Path, include_personal_hard_negatives: bool, epochs: int = 30, seed: int = SEED) -> dict[str, Any]:
    """Train one binary wake variant; hard-negative variant adds this user's non-wake speech."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from torch.onnx import export as torch_onnx_export

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    rng = np.random.default_rng(seed)
    frontend = Frontend()
    noises = _load_external_noise(data["project_root"])
    wake_train = [i for i in data["human"] if i["label"] == "wake_word" and i["split"] == "train"]
    wake_val = [i for i in data["human"] if i["label"] == "wake_word" and i["split"] == "validation"]
    wake_test = [i for i in data["human"] if i["label"] == "wake_word" and i["split"] == "test"]
    if not wake_train or not wake_val or not wake_test:
        raise ValueError(f"Wake data need non-empty train/validation/test after dedup: {len(wake_train)}/{len(wake_val)}/{len(wake_test)}")

    # Both conditions use the same human wake positives, Option B command pool, silence/noise,
    # validation examples and test examples. Variant B adds same-speaker spoken hard negatives.
    positive = np.concatenate([_context_windows(i["audio"], frontend, 12, rng, noises, True) for i in wake_train])
    # Every command in the training partition is a NON_WAKE example. Keep
    # validation and speaker-held-out test commands out of this pool.
    negative_parts = [data["train_x"]]
    personal_noise_train = [i for i in data["human"] if i["split"] == "train" and i["label"] in ("_background_noise_", "_silence_")]
    if personal_noise_train:
        negative_parts.append(np.stack([i["feature"] for i in personal_noise_train]).astype(np.float32))
    # Include a few real ambient-noise windows in both variants; use training files only.
    if noises:
        noise_features = []
        for _ in range(48):
            n = noises[int(rng.integers(0, len(noises)))]
            if len(n) < SAMPLES:
                n = np.tile(n, int(np.ceil(SAMPLES / max(1, len(n)))))
            offset = int(rng.integers(0, len(n) - SAMPLES + 1))
            noise_features.append(frontend(n[offset:offset + SAMPLES]))
        negative_parts.append(np.stack(noise_features))
    if include_personal_hard_negatives:
        hard_items = [i for i in data["human"] if i["split"] == "train" and i["label"] not in ("wake_word", "_background_noise_", "_silence_")]
        hard_features = []
        for item in hard_items:
            # Keep all takes as one source group, then use one randomized context per feature.
            hard_features.append(_context_windows(item["audio"], frontend, 1, rng, noises, True)[0])
        if hard_features:
            negative_parts.append(np.stack(hard_features))
    negative = np.concatenate(negative_parts, axis=0).astype(np.float32)
    train_x = np.concatenate((negative, positive)).astype(np.float32)
    train_y = np.concatenate((np.zeros(len(negative), dtype=np.int64), np.ones(len(positive), dtype=np.int64)))
    perm = rng.permutation(len(train_y))
    train_x, train_y = train_x[perm], train_y[perm]

    val_positive_windows = np.concatenate([_context_windows(i["audio"], frontend, 5) for i in wake_val])
    val_positive_groups = np.repeat(np.arange(len(wake_val)), 5)
    val_hardneg = [i for i in data["human"] if i["split"] == "validation" and i["label"] != "wake_word"]
    val_negative_parts = [data["val_x"]]
    val_negative_groups = [("optionb", j) for j in range(len(data["val_y"]))]
    if val_hardneg:
        hard_windows = np.concatenate([_context_windows(i["audio"], frontend, 5) for i in val_hardneg])
        val_negative_parts.append(hard_windows)
        val_negative_groups += [("human", i) for i, _ in enumerate(val_hardneg) for _ in range(5)]
    val_negative_x = np.concatenate(val_negative_parts).astype(np.float32)
    val_negative_group_ids = np.concatenate((np.arange(len(data["val_y"])), np.repeat(np.arange(len(val_hardneg)) + len(data["val_y"]), 5)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = TinyDSCNN(classes=2, channels=CHANNELS).to(device)
    initial_hash = _sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()))
    class_counts = np.bincount(train_y, minlength=2)
    weights = torch.tensor(np.sqrt(len(train_y) / np.maximum(class_counts, 1)), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    steps = math.ceil(len(train_y) / 64)
    best_recall, best_epoch, best_threshold = -1.0, 0, 1.000001
    best_state, history = None, []
    print(f"Wake variant={'hard-negative' if include_personal_hard_negatives else 'broad-negatives-only'} | wake unique train={len(wake_train)}, val={len(wake_val)}, test={len(wake_test)} | nonwake train={len(negative)} ({'includes own commands' if include_personal_hard_negatives else 'reference commands + noise only'}) | init=from scratch", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_y))
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(order), 64):
            ids = order[start:start + 64]
            xb = torch.from_numpy(train_x[ids].copy()).to(device)
            yb = torch.from_numpy(train_y[ids].copy()).to(device)
            # Batch-level time/frequency masks mirror the current tiny model recipe.
            for row in range(len(xb)):
                fw = int(torch.randint(0, 5, (1,), device=device).item())
                tw = int(torch.randint(0, 18, (1,), device=device).item())
                if fw:
                    fs = int(torch.randint(0, MELS - fw + 1, (1,), device=device).item())
                    xb[row, :, fs:fs + fw, :] = 0
                if tw:
                    ts = int(torch.randint(0, TIME_STEPS - tw + 1, (1,), device=device).item())
                    xb[row, :, :, ts:ts + tw] = 0
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
        scheduler.step()
        val_pos = _predict_torch_positive(model, val_positive_windows, device)
        wake_per_recording = np.asarray([val_pos[val_positive_groups == j].max() for j in range(len(wake_val))])
        val_neg = _predict_torch_positive(model, val_negative_x, device)
        neg_per_clip = np.asarray([val_neg[val_negative_group_ids == j].max() for j in np.unique(val_negative_group_ids)])
        threshold, threshold_metrics = _threshold_for_zero_false(wake_per_recording, neg_per_clip)
        recall = threshold_metrics["wake_recall"]
        entry = {"epoch": epoch, "train_loss": loss_sum / len(train_y), "train_accuracy": correct / len(train_y), "val_zero_FA_wake_recall": recall, "val_threshold": threshold, "val_wake_support": len(wake_val), "val_nonwake_clip_support": len(neg_per_clip), "epoch_seconds": time.perf_counter() - t0}
        history.append(entry)
        if (recall, threshold) > (best_recall, best_threshold):
            best_recall, best_threshold, best_epoch = recall, threshold, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pt")
        _save_json(output_dir / "history.json", {"history": history})
        print(f"Wake epoch {epoch:02d}/{epochs} loss={entry['train_loss']:.4f} train_acc={entry['train_accuracy']:.3f} val_wake_recall@zeroFA={recall:.3f} threshold={threshold:.4f}{' [BEST]' if best_epoch == epoch else ''}", flush=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    fp32 = model_dir / "wake_personalized_fp32.onnx"
    int8 = model_dir / "wake_personalized_int8.onnx"
    torch_onnx_export(model.cpu().eval(), torch.randn(1, *FEATURE_SHAPE), str(fp32), input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))
    calib_rng = np.random.default_rng(seed + 991)
    pos_idx = calib_rng.integers(0, len(positive), size=128)
    neg_idx = calib_rng.integers(0, len(negative), size=128)
    calibration = np.concatenate((positive[pos_idx], negative[neg_idx])).astype(np.float32)

    class Reader(CalibrationDataReader):
        def __init__(self, samples: np.ndarray): self.samples = iter([{"input": x[None]} for x in samples])
        def get_next(self): return next(self.samples, None)
    quantize_static(str(fp32), str(int8), Reader(calibration), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    onnx.checker.check_model(onnx.load(str(int8)))
    val_int8_pos = _predict_onnx(int8, val_positive_windows, binary=True)[:, 1]
    val_int8_wake = np.asarray([val_int8_pos[val_positive_groups == j].max() for j in range(len(wake_val))])
    val_int8_neg = _predict_onnx(int8, val_negative_x, binary=True)[:, 1]
    val_int8_nonwake = np.asarray([val_int8_neg[val_negative_group_ids == j].max() for j in np.unique(val_negative_group_ids)])
    int8_threshold, int8_threshold_metrics = _threshold_for_zero_false(val_int8_wake, val_int8_nonwake)
    val_fp32_pos = _predict_onnx(fp32, val_positive_windows, binary=True)[:, 1]
    val_fp32_wake = np.asarray([val_fp32_pos[val_positive_groups == j].max() for j in range(len(wake_val))])
    val_fp32_neg = _predict_onnx(fp32, val_negative_x, binary=True)[:, 1]
    val_fp32_nonwake = np.asarray([val_fp32_neg[val_negative_group_ids == j].max() for j in np.unique(val_negative_group_ids)])
    fp32_threshold, fp32_threshold_metrics = _threshold_for_zero_false(val_fp32_wake, val_fp32_nonwake)
    wake_speakers = sorted({i["speaker"] for i in data["human"] if i["label"] == "wake_word"})
    meta = {
        "classes": ["NON_WAKE", "WAKE_WORD"], "objective": "personalized wake vs non-wake; no command classes in this model",
        "variant": "personal_hard_negatives" if include_personal_hard_negatives else "broad_negatives_only",
        "pretrained_weights_used": False, "seed": seed, "initial_state_sha256": initial_hash,
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device), "epochs": epochs,
        "best_epoch": best_epoch, "best_validation_recall": best_recall,
        "unique_human_wake_splits": {s: sum(i["label"] == "wake_word" and i["split"] == s for i in data["human"]) for s in ("train", "validation", "test")},
        "positive_train_windows": int(len(positive)), "broad_negative_train_windows": int(len(negative)),
        "personal_command_hard_negatives_added": bool(include_personal_hard_negatives),
        "personal_manifest_sha256": data["human_audit"]["manifest_sha256"],
        "validation_threshold_fp32": float(fp32_threshold), "validation_fp32_metrics": fp32_threshold_metrics,
        "validation_threshold_int8": float(int8_threshold), "validation_int8_metrics": int8_threshold_metrics,
        "optionb_manifest_sha256": str(data["manifest_sha256"].item()),
        "wake_speaker_ids": wake_speakers,
        "limitation": f"Wake recordings cover {len(wake_speakers)} speaker IDs ({', '.join(wake_speakers)}); per-label clip-heldout evaluation does not establish unseen-speaker performance or false activations per hour.",
    }
    summary = {
        **meta, "fp32_size_bytes": fp32.stat().st_size, "int8_size_bytes": int8.stat().st_size,
        "fp32_sha256": _sha256(fp32.read_bytes()), "int8_sha256": _sha256(int8.read_bytes()),
        "calibration_samples": len(calibration), "calibration_policy": "training-only wake and nonwake examples",
        "model_paths": {"fp32": str(fp32), "int8": str(int8)},
        "fp32_validation_threshold_metrics": fp32_threshold_metrics,
        "int8_validation_threshold_metrics": int8_threshold_metrics,
    }
    _save_json(output_dir / "run_info.json", meta)
    _save_json(model_dir / "export_summary.json", summary)
    return {"output_dir": output_dir, "fp32_path": fp32, "int8_path": int8, "summary": summary,
            "val_threshold": int8_threshold, "wake_test": wake_test, "hardneg_added": include_personal_hard_negatives}


def prepare_experiment(output_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    split_manifest = output_dir / "personal_split_manifest.csv"
    # Freeze the manifest first: recording may continue while a run is training.
    snapshot = output_dir / "human_manifest_snapshot.csv"
    shutil.copyfile(project_root / "data" / "human" / "manifest.csv", snapshot)
    human, audit = load_personal_records(project_root, split_manifest, manifest_path=snapshot)
    audit["raw_manifest_snapshot"] = str(snapshot)
    audit["shared_split_manifest"] = str(split_manifest)
    _save_json(output_dir / "dataset_audit.json", audit)
    optionb = load_optionb_train_val(project_root)
    print(f"Human audio: {audit['raw_manifest_rows']} saved files -> {audit['unique_pcm_groups']} unique PCM groups; wake={audit['unique_label_groups'].get('wake_word', 0)}; speaker IDs={audit['speakers']}", flush=True)
    if audit["cross_label_collision_groups_excluded"]:
        print(f"WARNING: excluded {audit['cross_label_collision_groups_excluded']} ambiguous cross-label waveform group(s); original WAVs and manifest rows are preserved and the exact paths are in dataset_audit.json.", flush=True)
    print(f"Mapped human intent support (unique clips): {audit['unique_mapped_intents']}", flush=True)
    print(f"Unmapped command labels kept out of intent training: {audit['unmapped_labels']}", flush=True)
    print(f"Reference train/validation features loaded from verified-count cache: {optionb['train_x'].shape}/{optionb['val_x'].shape}; test split remains unopened.", flush=True)
    return {**optionb, "project_root": project_root, "human": human, "human_audit": audit, "output_dir": output_dir}


def run_wake_pair(data: dict[str, Any], epochs: int = 30) -> dict[str, Any]:
    base = data["output_dir"]
    broad = train_wake_model(data, base / "wake_broad_negatives", False, epochs=epochs, seed=SEED)
    hard = train_wake_model(data, base / "wake_with_personal_intents", True, epochs=epochs, seed=SEED)
    return {"broad": broad, "hard": hard}


def train_intent_candidate_from_data(
    data: dict[str, Any], epochs: int = 60, seed: int = 232, recipe: str = "refined",
) -> dict[str, Any]:
    return train_intent_candidate(
        data, data["output_dir"] / "intent_personalized", epochs=epochs, seed=seed, recipe=recipe,
    )


def _wake_test_windows(items: list[dict[str, Any]], frontend: Frontend) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    groups = []
    for ix, item in enumerate(items):
        w = _context_windows(item["audio"], frontend, 5)
        feats.append(w)
        groups.extend([ix] * len(w))
    if not feats:
        return np.empty((0, *FEATURE_SHAPE), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(feats), np.asarray(groups, dtype=np.int64)


def _latency(path: Path, feature: np.ndarray, binary: bool = False) -> dict[str, float]:
    import onnxruntime as ort
    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 1
    session_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=session_opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    inp = feature[:1].astype(np.float32)
    for _ in range(10): session.run(None, {name: inp})
    times = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {name: inp})
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms_pc_cpu_model_only": float(np.percentile(times, 50)), "p95_ms_pc_cpu_model_only": float(np.percentile(times, 95))}


def evaluate_all(data: dict[str, Any], wake_runs: dict[str, Any], intent_run: dict[str, Any]) -> dict[str, Any]:
    """One-time final test pass after both wake exports and intent export are frozen."""
    import onnx
    frontend = Frontend()
    optb_test_x, optb_test_y = load_optionb_test(data["project_root"])
    test_intents = [i for i in data["human"] if i["split"] == "test" and i["intent"]]
    personal_x, personal_y = _feature_matrix(test_intents)
    ignored_labels = {"wake_word", "_background_noise_", "_silence_", "_unknown_"}
    all_test_commands = [i for i in data["human"] if i["split"] == "test" and i["label"] not in ignored_labels]
    all_test_x = np.stack([i["feature"] for i in all_test_commands]).astype(np.float32) if all_test_commands else np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
    personal_wake = [i for i in data["human"] if i["split"] == "test" and i["label"] == "wake_word"]
    personal_nonwake = [i for i in data["human"] if i["split"] == "test" and i["label"] != "wake_word"]
    personal_nonwake_x, personal_nonwake_groups = _wake_test_windows(personal_nonwake, frontend)
    human_wake_x, human_wake_groups = _wake_test_windows(personal_wake, frontend)
    binary_test_neg_x = np.concatenate((optb_test_x, personal_nonwake_x))
    binary_neg_groups = np.concatenate((np.arange(len(optb_test_y)), personal_nonwake_groups + len(optb_test_y)))
    n_optionb = len(optb_test_y)

    wake_results = {}
    for key, run in wake_runs.items():
        p_path, q_path = run["fp32_path"], run["int8_path"]
        fp_wake = _predict_onnx(p_path, human_wake_x, binary=True)[:, 1]
        fp_wake = np.asarray([fp_wake[human_wake_groups == j].max() for j in range(len(personal_wake))])
        fp_neg = _predict_onnx(p_path, binary_test_neg_x, binary=True)[:, 1]
        fp_neg = np.asarray([fp_neg[binary_neg_groups == j].max() for j in np.unique(binary_neg_groups)])
        fp_threshold = float(run["summary"]["validation_threshold_fp32"])
        fp_accepted_wake, fp_accepted_neg = fp_wake >= fp_threshold, fp_neg >= fp_threshold
        p_pos = _predict_onnx(q_path, human_wake_x, binary=True)[:, 1]
        q_wake = np.asarray([p_pos[human_wake_groups == j].max() for j in range(len(personal_wake))])
        p_neg = _predict_onnx(q_path, binary_test_neg_x, binary=True)[:, 1]
        q_neg = np.asarray([p_neg[binary_neg_groups == j].max() for j in np.unique(binary_neg_groups)])
        thr = float(run["val_threshold"])
        accepted_wake = q_wake >= thr
        accepted_neg = q_neg >= thr
        wake_results[key] = {
            "fp32_validation_selected_threshold": fp_threshold,
            "fp32_heldout_personal_wake_hits": int(fp_accepted_wake.sum()),
            "fp32_heldout_personal_wake_support": int(len(fp_accepted_wake)),
            "fp32_heldout_personal_wake_recall": float(fp_accepted_wake.mean()) if len(fp_accepted_wake) else None,
            "fp32_false_wake_on_optionb_command_clips": int(fp_accepted_neg[:n_optionb].sum()),
            "fp32_false_wake_on_personal_nonwake_clips": int(fp_accepted_neg[n_optionb:].sum()),
            "fp32_model_size_bytes": Path(p_path).stat().st_size,
            "validation_selected_int8_threshold": thr,
            "heldout_personal_wake_hits": int(accepted_wake.sum()), "heldout_personal_wake_support": int(len(q_wake)),
            "heldout_personal_wake_recall": float(accepted_wake.mean()) if len(q_wake) else None,
            "false_wake_on_optionb_command_clips": int(accepted_neg[:n_optionb].sum()), "optionb_command_support": n_optionb,
            "false_wake_on_personal_nonwake_clips": int(accepted_neg[n_optionb:].sum()), "personal_nonwake_support": int(len(accepted_neg) - n_optionb),
            "personal_nonwake_by_label": dict(Counter(i["label"] for i in personal_nonwake)),
            "int8_model_size_bytes": Path(q_path).stat().st_size, "int8_model_sha256": _sha256(Path(q_path).read_bytes()),
            "pc_cpu_latency": _latency(q_path, binary_test_neg_x, binary=True),
        }
    intent_results = {}
    baseline_paths = {
        "canonical_fp32": data["project_root"] / "archive" / "models" / "vcm_optionb_fp32.onnx",
        "canonical_int8": data["project_root"] / "archive" / "models" / "vcm_optionb_int8.onnx",
        "personalized_fp32": Path(intent_run["output_dir"]) / "models" / "intent_personalized_fp32.onnx",
        "personalized_int8": Path(intent_run["output_dir"]) / "models" / "intent_personalized_int8.onnx",
    }
    models_summary = {}
    for name, path in baseline_paths.items():
        pred_optb = _predict_onnx(path, optb_test_x).argmax(axis=1)
        pred_personal = _predict_onnx(path, personal_x).argmax(axis=1) if len(personal_y) else np.empty(0, dtype=np.int64)
        pred_all_commands = _predict_onnx(path, all_test_x).argmax(axis=1) if len(all_test_commands) else np.empty(0, dtype=np.int64)
        per_recorded_label = {}
        for source_label in sorted({i["label"] for i in all_test_commands}):
            ids = [j for j, item in enumerate(all_test_commands) if item["label"] == source_label]
            expected = map_recorded_intent(source_label)
            expected_idx = LABELS.index(expected) if expected in LABELS else None
            correct = int(sum(int(pred_all_commands[j]) == expected_idx for j in ids)) if expected_idx is not None else None
            per_recorded_label[source_label] = {
                "heldout_clip_support": len(ids), "mapped_intent": expected,
                "correct_hits": correct,
                "accuracy": (correct / len(ids)) if correct is not None and ids else None,
                "predicted_class_counts": dict(Counter(LABELS[int(pred_all_commands[j])] for j in ids)),
            }
        models_summary[name] = {
            "optionb_test": _intent_metrics(optb_test_y, pred_optb),
            "personal_heldout_test": _intent_metrics(personal_y, pred_personal) if len(personal_y) else {"count": 0},
            "personal_recorded_commands_all_heldout": {
                "clip_support": len(all_test_commands),
                "mapped_clip_support": sum(v["heldout_clip_support"] for v in per_recorded_label.values() if v["mapped_intent"]),
                "unmapped_label_clip_support": sum(v["heldout_clip_support"] for v in per_recorded_label.values() if not v["mapped_intent"]),
                "per_recorded_label": per_recorded_label,
            },
            "model_size_bytes": path.stat().st_size,
            "model_sha256": _sha256(path.read_bytes()), "pc_cpu_latency": _latency(path, optb_test_x),
        }
    p_probs = _predict_onnx(baseline_paths["personalized_fp32"], optb_test_x)
    q_probs = _predict_onnx(baseline_paths["personalized_int8"], optb_test_x)
    models_summary["personalized_fp32"]["int8_prediction_parity_on_optionb_test"] = float(np.mean(p_probs.argmax(1) == q_probs.argmax(1)))
    p_probs = _predict_onnx(baseline_paths["personalized_fp32"], personal_x)
    q_probs = _predict_onnx(baseline_paths["personalized_int8"], personal_x)
    models_summary["personalized_fp32"]["int8_prediction_parity_on_personal_test"] = float(np.mean(p_probs.argmax(1) == q_probs.argmax(1))) if len(personal_y) else None
    intent_results = models_summary
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "wake_model_comparison": wake_results,
        "intent_model_comparison": intent_results,
        "test_protocol": "ME2 Spoken Command Dataset speaker-disjoint test split opened once after all training/exports; personal data use a content-deduplicated per-label utterance holdout with speaker overlap across partitions; validation-only checkpoint/threshold selection",
        "personal_speaker_ids": data["human_audit"]["speakers"],
        "known_limitations": [
            f"Personal manifest contains {len(data['human_audit']['speakers'])} speaker IDs ({', '.join(data['human_audit']['speakers'])}); per-label clip splitting permits speaker overlap across train/validation/test, so these results do not establish unseen-speaker accuracy.",
            "Personal held-out support is low for several intent classes and is reported per class.",
            "ME2 Spoken Command Dataset test clips have prior project evaluations and are not a pristine first-use benchmark.",
            "Command-window false wake counts are not continuous-audio false activations per hour; no Pi hardware was used.",
            "INT8 latency here is local Windows PC CPU model-only latency, not Raspberry Pi frontend-plus-model latency.",
            "This experiment does not automatically replace the canonical model files or change the Pi release.",
        ],
    }
    _save_json(Path(data["output_dir"]) / "final_evaluation.json", report)
    return report
