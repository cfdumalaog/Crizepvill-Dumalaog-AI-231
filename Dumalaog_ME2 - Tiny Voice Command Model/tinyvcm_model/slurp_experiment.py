"""Leakage-aware, paired scratch training for Option B with SLURP real speech.

The preparation function reads training features and validation features only.
The test loader is called once, from the notebook's last code cell after both
models have finished training, checkpoint selection, export, and quantization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_model.config import FEATURE_SHAPE, LABELS, SAMPLES, SR, WAKE_LABELS, CHANNELS  # noqa: E402
from tinyvcm_model.frontend import Frontend, fit_audio  # noqa: E402
from tinyvcm_model.model import TinyDSCNN  # noqa: E402
from tinyvcm_model.recording_labels import map_recorded_intent  # noqa: E402

SUPPORTED_SLURP_CLASSES = (
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "VOLUME_UP", "VOLUME_DOWN",
    "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE", "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
)
SUPPORTED_INDICES = {WAKE_LABELS.index(label) for label in SUPPORTED_SLURP_CLASSES}
WAKE_INDEX = WAKE_LABELS.index("WAKE_WORD")
PREPROCESSING_VERSION = "tinyvcm-logmel40-16khz-2.5s-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def read_mono_16khz(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != SR:
        raise ValueError(f"Expected {SR} Hz audio, got {sample_rate} Hz: {path}")
    mono = audio.mean(axis=1, dtype=np.float32)
    return mono, int(sample_rate)


def feature_cache_matches(manifest_path: Path, cache_path: Path) -> bool:
    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return (
            sidecar.get("manifest_sha256") == sha256_file(manifest_path)
            and sidecar.get("preprocessing") == PREPROCESSING_VERSION
            and tuple(sidecar.get("feature_shape", ())) == FEATURE_SHAPE
        )
    except (OSError, ValueError, TypeError):
        return False


def extract_features_from_manifest(
    manifest_path: Path,
    cache_path: Path,
    *,
    force: bool = False,
    expected_split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Feature only the requested manifest, with a manifest-keyed cache."""
    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path)
    if not force and feature_cache_matches(manifest_path, cache_path):
        with np.load(cache_path) as cache:
            return cache["x"].astype(np.float32), cache["y"].astype(np.int64), cache["source_ids"].astype(str), int(cache["cropped_count"])

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")
    seen_sources = set()
    frontend = Frontend()
    features: list[np.ndarray] = []
    labels: list[int] = []
    source_ids: list[str] = []
    cropped_count = 0
    missing_or_bad = 0
    for row_index, row in enumerate(rows, start=1):
        if row.get("split") != expected_split:
            raise ValueError(f"Manifest row split mismatch at {manifest_path}:{row_index}")
        if expected_split == "train" and row.get("audio_checked") != "1":
            raise ValueError(f"Training row was not decoded by the manifest audit: {row.get('path')}")
        source_key = (row.get("usrid", ""), row.get("recid", ""))
        if source_key in seen_sources:
            raise ValueError(f"Duplicate SLURP source group in {manifest_path}: {source_key}")
        seen_sources.add(source_key)
        audio_path = (PROJECT_ROOT / row["path"]).resolve()
        if not audio_path.is_relative_to((PROJECT_ROOT / "data" / "external" / "slurp" / "audio" / "slurp_real").resolve()):
            raise ValueError(f"Manifest audio path escapes the official real-audio directory: {row['path']}")
        try:
            waveform, _ = read_mono_16khz(audio_path)
            if len(waveform) > SAMPLES:
                cropped_count += 1
            features.append(frontend(fit_audio(waveform, target_samples=SAMPLES)))
            labels.append(int(row["class_idx"]))
            source_ids.append(f"slurp:{row['usrid']}:{row['recid']}")
        except Exception as error:
            missing_or_bad += 1
            raise RuntimeError(f"Could not decode manifest audio {row.get('path')}: {error}") from error
        if row_index % 500 == 0 or row_index == len(rows):
            print(f"  [{expected_split}] featurized {row_index:,}/{len(rows):,} unique source recordings", flush=True)

    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(source_ids, dtype=str)
    if x.shape[1:] != FEATURE_SHAPE:
        raise AssertionError(f"Unexpected feature tensor shape {x.shape}; expected (*, {FEATURE_SHAPE})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, source_ids=sources, cropped_count=np.asarray(cropped_count))
    sidecar = {
        "manifest_sha256": sha256_file(manifest_path),
        "preprocessing": PREPROCESSING_VERSION,
        "feature_shape": FEATURE_SHAPE,
        "sample_rate": SR,
        "target_samples": SAMPLES,
        "rows": len(y),
        "cropped_count": cropped_count,
        "decode_failures": missing_or_bad,
    }
    cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"  [SAVED] {len(y):,} decoded {expected_split} features; cropped over-2.5s clips: {cropped_count:,}", flush=True)
    return x, y, sources, cropped_count


def _read_wake_groups(human_rows: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    wake_rows = [row for row in human_rows if row.get("label") == "wake_word"]
    for row in wake_rows:
        path = (PROJECT_ROOT / row["path"]).resolve()
        if not path.is_file():
            continue
        waveform, _ = read_mono_16khz(path)
        pcm_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
        groups[pcm_hash].append({
            "path": str(row["path"]),
            "filename": path.name,
            "condition": row.get("condition", "unknown"),
            "wav": waveform,
        })
    if len(groups) != 9:
        raise AssertionError(f"Expected 9 unique human wake recordings after PCM deduplication, got {len(groups)}")
    canonical = {
        digest: sorted(items, key=lambda item: item["path"])[0] | {"pcm_sha256": digest}
        for digest, items in groups.items()
    }
    ordered_hashes = sorted(canonical)
    train_hashes, val_hashes, test_hashes = ordered_hashes[:5], ordered_hashes[5:7], ordered_hashes[7:]
    split_refs = {
        "train": [canonical[digest] | {"pcm_sha256": digest} for digest in train_hashes],
        "val": [canonical[digest] | {"pcm_sha256": digest} for digest in val_hashes],
        "test": [
            {key: value for key, value in canonical[digest].items() if key != "wav"} | {"pcm_sha256": digest}
            for digest in test_hashes
        ],
    }
    return canonical, [
        {"pcm_sha256": digest, "split": split, "paths": sorted(item["path"] for item in groups[digest])}
        for split, hashes in (("train", train_hashes), ("val", val_hashes), ("test", test_hashes))
        for digest in hashes
    ]


def _read_synthetic_wake_groups() -> dict[str, list[Path]]:
    root = PROJECT_ROOT / "data" / "dataset" / "wake_word"
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*.wav")):
        group = re.sub(r"_\d+_[^_]+$", "", path.stem)
        groups[group].append(path)
    if len(groups) != 9:
        raise AssertionError(f"Expected 9 synthetic wake source groups, found {len(groups)}")
    return dict(sorted(groups.items()))


def _read_background_noise() -> list[np.ndarray]:
    root = PROJECT_ROOT / "data" / "dataset" / "_background_noise_"
    waves = []
    for path in sorted(root.glob("*.wav")):
        wave, _ = read_mono_16khz(path)
        if len(wave):
            waves.append(wave)
    return waves


def augment_waveform(waveform: np.ndarray, rng: np.random.Generator, noises: list[np.ndarray]) -> np.ndarray:
    augmented = waveform.astype(np.float32, copy=True)
    augmented *= rng.uniform(0.65, 1.35)
    shift = int(rng.integers(-2400, 2400))
    augmented = np.roll(augmented, shift)
    if shift > 0:
        augmented[:shift] = 0.0
    elif shift < 0:
        augmented[shift:] = 0.0
    if noises and rng.random() < 0.75:
        noise = noises[int(rng.integers(0, len(noises)))]
        if len(noise) >= len(augmented):
            start = int(rng.integers(0, len(noise) - len(augmented) + 1))
            noise_slice = noise[start : start + len(augmented)]
        else:
            noise_slice = np.tile(noise, int(np.ceil(len(augmented) / len(noise))))[: len(augmented)]
        signal_power = float(np.mean(augmented**2)) + 1e-9
        noise_power = float(np.mean(noise_slice**2)) + 1e-9
        target_snr = rng.uniform(20.0, 35.0)
        augmented += noise_slice * np.sqrt(signal_power / (10 ** (target_snr / 10.0) * noise_power))
    return fit_audio(augmented, target_samples=SAMPLES)


def _extract_file_feature(path: Path, frontend: Frontend) -> np.ndarray:
    waveform, _ = read_mono_16khz(path)
    return frontend(fit_audio(waveform, target_samples=SAMPLES))


def prepare_paired_datasets(seed: int = 231, force_slurp_train_cache: bool = False) -> dict:
    """Prepare train/validation only; test audio/features are deferred."""
    rng = np.random.default_rng(seed)
    frontend = Frontend()
    project_data = PROJECT_ROOT / "data"
    optionb_cache = project_data / "option_b" / "features_cache.npz"
    optionb_manifest = project_data / "option_b" / "manifest.csv"
    human_manifest = project_data / "human" / "manifest.csv"
    slurp_dir = project_data / "slurp"
    slurp_train_manifest = slurp_dir / "manifest_slurp_train.csv"
    slurp_test_manifest = slurp_dir / "manifest_slurp_test.csv"

    print("Reading Option B train/validation features only; test arrays remain unopened.", flush=True)
    with np.load(optionb_cache) as cache:
        optb_train_x = cache["train_x"].astype(np.float32)
        optb_train_y = cache["train_y"].astype(np.int64)
        optb_val_x = cache["val_x"].astype(np.float32)
        optb_val_y = cache["val_y"].astype(np.int64)

    human_rows = list(csv.DictReader(human_manifest.open("r", encoding="utf-8", newline="")))
    wake_groups, wake_split_manifest = _read_wake_groups(human_rows)
    wake_split_path = slurp_dir / "human_wake_split_manifest.json"
    wake_split_path.write_text(json.dumps(wake_split_manifest, indent=2), encoding="utf-8")
    wake_rows = json.loads(wake_split_path.read_text(encoding="utf-8"))
    human_split_map = {row["pcm_sha256"]: row["split"] for row in wake_rows}
    train_human_wake = [wake_groups[digest] for digest, split in human_split_map.items() if split == "train"]
    val_human_wake = [wake_groups[digest] for digest, split in human_split_map.items() if split == "val"]
    test_human_wake_refs = [
        {key: value for key, value in wake_groups[digest].items() if key != "wav"} | {"pcm_sha256": digest}
        for digest, split in human_split_map.items() if split == "test"
    ]

    synth_groups = _read_synthetic_wake_groups()
    synth_names = list(synth_groups)
    train_synth_files = [path for group in synth_names[:7] for path in synth_groups[group]]
    val_synth_files = [path for group in synth_names[7:8] for path in synth_groups[group]]
    test_synth_files = [path for group in synth_names[8:] for path in synth_groups[group]]

    personal_x, personal_y, personal_sources = [], [], []
    for row in human_rows:
        target_label = map_recorded_intent(row.get("label", ""))
        if target_label is None or target_label not in LABELS:
            continue
        path = (PROJECT_ROOT / row["path"]).resolve()
        personal_x.append(_extract_file_feature(path, frontend))
        personal_y.append(LABELS.index(target_label))
        personal_sources.append(f"personal:{row['path']}")
    personal_x = np.stack(personal_x).astype(np.float32) if personal_x else np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
    personal_y = np.asarray(personal_y, dtype=np.int64)
    personal_sources = np.asarray(personal_sources, dtype=str)

    noise_waves = _read_background_noise()
    train_wake_features: list[np.ndarray] = []
    train_wake_sources: list[str] = []
    train_wake_kinds: list[str] = []
    for item in train_human_wake:
        pcm_hash = item["pcm_sha256"]
        base_wave = item["wav"]
        train_wake_features.append(frontend(fit_audio(base_wave, target_samples=SAMPLES)))
        train_wake_sources.append(f"human-wake:{pcm_hash}")
        train_wake_kinds.append("human_wake")
        for _ in range(59):
            train_wake_features.append(frontend(augment_waveform(base_wave, rng, noise_waves)))
            train_wake_sources.append(f"human-wake:{pcm_hash}")
            train_wake_kinds.append("human_wake")

    for path in train_synth_files:
        wave, _ = read_mono_16khz(path)
        group = next(name for name in synth_names[:7] if path in synth_groups[name])
        train_wake_features.append(frontend(fit_audio(wave, target_samples=SAMPLES)))
        train_wake_sources.append(f"synth-wake:{group}")
        train_wake_kinds.append("synth_wake")
        train_wake_features.append(frontend(augment_waveform(wave, rng, noise_waves)))
        train_wake_sources.append(f"synth-wake:{group}")
        train_wake_kinds.append("synth_wake")
    train_wake_x = np.stack(train_wake_features).astype(np.float32)
    train_wake_y = np.full(len(train_wake_x), WAKE_INDEX, dtype=np.int64)
    train_wake_sources = np.asarray(train_wake_sources, dtype=str)
    train_wake_kinds = np.asarray(train_wake_kinds, dtype=str)

    val_hw_x = np.stack([frontend(fit_audio(item["wav"], target_samples=SAMPLES)) for item in val_human_wake]).astype(np.float32)
    val_synth_x = np.stack([_extract_file_feature(path, frontend) for path in val_synth_files]).astype(np.float32)

    slurp_train_x, slurp_train_y, slurp_train_sources, slurp_cropped_count = extract_features_from_manifest(
        slurp_train_manifest,
        slurp_dir / "slurp_train_features.npz",
        force=force_slurp_train_cache,
        expected_split="train",
    )

    control_x = np.concatenate([optb_train_x, train_wake_x, personal_x], axis=0)
    control_y = np.concatenate([optb_train_y, train_wake_y, personal_y], axis=0)
    control_sources = np.concatenate([
        np.asarray([f"optionb:{i}" for i in range(len(optb_train_y))], dtype=str),
        train_wake_sources,
        personal_sources,
    ])
    control_kinds = np.concatenate([
        np.full(len(optb_train_y), "optionb", dtype=str), train_wake_kinds,
        np.full(len(personal_y), "personal_command", dtype=str),
    ])

    treatment_x_parts, treatment_y_parts, treatment_source_parts, treatment_kind_parts = [], [], [], []
    mix_by_class = {}
    for class_index in range(len(LABELS)):
        optb_indices = np.flatnonzero(optb_train_y == class_index)
        if not len(optb_indices):
            raise ValueError(f"No Option B training examples for class {LABELS[class_index]}")
        slurp_indices = np.flatnonzero(slurp_train_y == class_index)
        if class_index in SUPPORTED_INDICES and len(slurp_indices):
            target_slurp = int(round(len(optb_indices) * 0.25))
            # Use distinct natural recordings only. Rare SLURP classes get a
            # smaller treatment share instead of duplicate oversampling.
            n_slurp = min(target_slurp, len(slurp_indices))
            n_optb = len(optb_indices) - n_slurp
            chosen_optb = rng.permutation(optb_indices)[:n_optb]
            chosen_slurp = rng.permutation(slurp_indices)[:n_slurp]
            treatment_x_parts += [optb_train_x[chosen_optb], slurp_train_x[chosen_slurp]]
            treatment_y_parts += [optb_train_y[chosen_optb], slurp_train_y[chosen_slurp]]
            treatment_source_parts += [
                np.asarray([f"optionb:{i}" for i in chosen_optb], dtype=str),
                slurp_train_sources[chosen_slurp],
            ]
            treatment_kind_parts += [np.full(n_optb, "optionb", dtype=str), np.full(n_slurp, "slurp", dtype=str)]
            mix_by_class[LABELS[class_index]] = {
                "option_b": int(n_optb),
                "slurp": int(n_slurp),
                "slurp_fraction": float(n_slurp / len(optb_indices)),
                "slurp_available": int(len(slurp_indices)),
                "target_fraction": 0.25,
            }
        else:
            treatment_x_parts.append(optb_train_x[optb_indices])
            treatment_y_parts.append(optb_train_y[optb_indices])
            treatment_source_parts.append(np.asarray([f"optionb:{i}" for i in optb_indices], dtype=str))
            treatment_kind_parts.append(np.full(len(optb_indices), "optionb", dtype=str))
            if class_index in SUPPORTED_INDICES:
                mix_by_class[LABELS[class_index]] = {
                    "option_b": int(len(optb_indices)), "slurp": 0,
                    "slurp_fraction": 0.0, "slurp_available": int(len(slurp_indices)), "target_fraction": 0.25,
                }
    treatment_x_parts += [train_wake_x, personal_x]
    treatment_y_parts += [train_wake_y, personal_y]
    treatment_source_parts += [train_wake_sources, personal_sources]
    treatment_kind_parts += [train_wake_kinds, np.full(len(personal_y), "personal_command", dtype=str)]
    treatment_x = np.concatenate(treatment_x_parts, axis=0).astype(np.float32)
    treatment_y = np.concatenate(treatment_y_parts, axis=0).astype(np.int64)
    treatment_sources = np.concatenate(treatment_source_parts).astype(str)
    treatment_kinds = np.concatenate(treatment_kind_parts).astype(str)
    if len(control_y) != len(treatment_y):
        raise AssertionError(f"Paired epoch step budget differs: {len(control_y)} vs {len(treatment_y)}")
    print(f"Fixed matched sample count: {len(control_y):,} per run; both use the notebook's same epoch and batch schedule.")

    source_manifest_hashes = {
        "option_b_manifest_sha256": sha256_file(optionb_manifest),
        "human_manifest_sha256": sha256_file(human_manifest),
        "slurp_train_manifest_sha256": sha256_file(slurp_train_manifest),
    }
    provenance = {
        "seed": seed,
        "pretrained_weights_used": False,
        "option_b_train_rows": int(len(optb_train_y)),
        "option_b_validation_rows": int(len(optb_val_y)),
        "personal_command_train_rows": int(len(personal_y)),
        "personal_command_unique_files": int(len(set(personal_sources.tolist()))),
        "wake_train_rows": int(len(train_wake_y)),
        "slurp_train_rows": int(len(slurp_train_y)),
        "slurp_train_speakers": int(len({source.split(":")[1] for source in slurp_train_sources})),
        "slurp_train_source_groups": int(len(set(slurp_train_sources))),
        "slurp_train_over_2_5s_rows": slurp_cropped_count,
        "human_wake_split": {"train_unique": len(train_human_wake), "val_unique": len(val_human_wake), "test_unique": len(test_human_wake_refs)},
        "human_wake_split_sha256": sha256_file(wake_split_path),
        "synthetic_wake_source_groups": {"train": 7, "val": 1, "test": 1},
        "control_train_rows": int(len(control_y)),
        "treatment_train_rows": int(len(treatment_y)),
        "control_train_y_sha256": sha256_array(control_y),
        "treatment_train_y_sha256": sha256_array(treatment_y),
        "mixture_by_supported_class": mix_by_class,
        "source_manifest_hashes": source_manifest_hashes,
        "frontend": PREPROCESSING_VERSION,
        "feature_shape": FEATURE_SHAPE,
        "sample_rate": SR,
        "clip_samples": SAMPLES,
    }

    # Discard deduplication-only waveforms, including the two wake test takes.
    del wake_groups
    return {
        "control_train_x": np.ascontiguousarray(control_x),
        "control_train_y": control_y,
        "control_source_ids": control_sources,
        "control_source_kinds": control_kinds,
        "treatment_train_x": np.ascontiguousarray(treatment_x),
        "treatment_train_y": treatment_y,
        "treatment_source_ids": treatment_sources,
        "treatment_source_kinds": treatment_kinds,
        "optb_val_x": optb_val_x,
        "optb_val_y": optb_val_y,
        "val_hw_x": val_hw_x,
        "val_synth_x": val_synth_x,
        "optb_cache_path": optionb_cache,
        "optionb_manifest_path": optionb_manifest,
        "slurp_test_manifest_path": slurp_test_manifest,
        "test_human_wake_refs": test_human_wake_refs,
        "test_synthetic_wake_files": [str(path.relative_to(PROJECT_ROOT)) for path in test_synth_files],
        "provenance": provenance,
    }


class TrainCalibrationReader(CalibrationDataReader):
    def __init__(self, features: np.ndarray):
        self._iter = iter(features[index : index + 1] for index in range(len(features)))

    def get_next(self):
        feature = next(self._iter, None)
        return None if feature is None else {"input": feature}


def _distinct_indices(indices: np.ndarray, source_ids: np.ndarray, requested: int, rng: np.random.Generator) -> list[int]:
    chosen, seen = [], set()
    for index in rng.permutation(indices):
        source = str(source_ids[index])
        if source in seen:
            continue
        seen.add(source)
        chosen.append(int(index))
        if len(chosen) == requested:
            break
    return chosen


def _calibration_indices(
    labels: np.ndarray,
    source_ids: np.ndarray,
    source_kinds: np.ndarray,
    *,
    is_treatment: bool,
    seed: int,
    per_class: int = 8,
) -> tuple[list[int], list[dict]]:
    rng = np.random.default_rng(seed + 1701)
    selected, provenance = [], []
    for class_index, label in enumerate(WAKE_LABELS):
        candidates = np.flatnonzero(labels == class_index)
        if not len(candidates):
            raise ValueError(f"No training-only calibration rows for {label}")
        preferred = []
        if class_index == WAKE_INDEX:
            for kind in ("human_wake", "synth_wake"):
                source_candidates = candidates[source_kinds[candidates] == kind]
                preferred.extend(_distinct_indices(source_candidates, source_ids, min(4, per_class), rng))
        elif is_treatment and class_index in SUPPORTED_INDICES:
            slurp_candidates = candidates[source_kinds[candidates] == "slurp"]
            other_candidates = candidates[source_kinds[candidates] != "slurp"]
            preferred.extend(_distinct_indices(slurp_candidates, source_ids, per_class // 2, rng))
            preferred.extend(_distinct_indices(other_candidates, source_ids, per_class - len(preferred), rng))
        chosen = list(dict.fromkeys(preferred))
        if len(chosen) < per_class:
            rest = np.asarray([index for index in candidates if int(index) not in set(chosen)], dtype=np.int64)
            chosen.extend(_distinct_indices(rest, source_ids, per_class - len(chosen), rng))
        if len(chosen) < per_class:
            # Only fall back to repeated training rows if that class has fewer
            # unique sources than the calibration quota; record the limitation.
            need = per_class - len(chosen)
            chosen.extend(int(value) for value in rng.choice(candidates, size=need, replace=True))
        chosen = chosen[:per_class]
        selected.extend(chosen)
        provenance.append({
            "class": label,
            "rows": [{"source_id": str(source_ids[i]), "source_kind": str(source_kinds[i])} for i in chosen],
        })
    return selected, provenance


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def train_model(
    *,
    run_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_source_ids: np.ndarray,
    train_source_kinds: np.ndarray,
    datasets: dict,
    run_name: str,
    model_name: str,
    is_treatment: bool,
    seed: int = 231,
    epochs: int = 35,
    batch_size: int = 64,
) -> dict:
    """Train, select from validation, and export. This function never loads tests."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDSCNN(classes=len(WAKE_LABELS), channels=CHANNELS).to(device)
    initial_hash = hashlib.sha256(b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())).hexdigest()
    params = int(sum(parameter.numel() for parameter in model.parameters()))
    run_provenance = {
        **datasets["provenance"],
        "run_name": run_name,
        "model_name": model_name,
        "is_treatment": is_treatment,
        "pretrained_weights_used": False,
        "initial_parameters_sha256": initial_hash,
        "parameter_count": params,
        "labels": WAKE_LABELS,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "device": str(device),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": 3e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "optimizer_steps_requested": int(math.ceil(len(train_y) / batch_size) * epochs),
    }
    _write_json(run_dir / "initial_provenance.json", run_provenance)
    print("=" * 84, flush=True)
    print(f"START {run_name}: {device}, {len(train_y):,} training rows, {epochs} epochs", flush=True)
    print(f"Fresh initialization SHA-256: {initial_hash}", flush=True)

    counts = np.bincount(train_y, minlength=len(WAKE_LABELS)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Missing training classes: {[WAKE_LABELS[i] for i in np.flatnonzero(counts == 0)]}")
    class_weights = np.clip(len(train_y) / (len(WAKE_LABELS) * counts), 0.25, 4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device), label_smoothing=0.02)
    gpu_x = torch.from_numpy(np.ascontiguousarray(train_x)).to(device)
    gpu_y = torch.from_numpy(np.ascontiguousarray(train_y)).to(device)
    optb_val_x, optb_val_y = datasets["optb_val_x"], datasets["optb_val_y"]
    val_hw_x, val_synth_x = datasets["val_hw_x"], datasets["val_synth_x"]

    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(gpu_y), device=device)
        running_loss, running_correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            xb = gpu_x[ids].clone()
            yb = gpu_y[ids]
            frequency_start = int(np.random.randint(0, 36))
            time_start = int(np.random.randint(0, 235))
            xb[:, :, frequency_start : frequency_start + 4, :] = 0.0
            xb[:, :, :, time_start : time_start + 12] = 0.0
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(ids)
            running_correct += int((logits.argmax(1) == yb).sum().item())
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            val_logits = []
            for start in range(0, len(optb_val_x), 128):
                val_batch = torch.from_numpy(optb_val_x[start : start + 128]).to(device)
                val_logits.append(model(val_batch).cpu())
            command_logits = torch.cat(val_logits).numpy()
            command_probs = torch.softmax(torch.from_numpy(command_logits), dim=-1).numpy()
            command_predictions = command_logits.argmax(axis=-1)
            command_accuracy = float(np.mean(command_predictions == optb_val_y))
            command_f1 = float(f1_score(optb_val_y, command_predictions, average="macro", zero_division=0))
            human_wake_logits = model(torch.from_numpy(val_hw_x).to(device)).cpu().numpy()
            synthetic_wake_logits = model(torch.from_numpy(val_synth_x).to(device)).cpu().numpy()
            human_wake_probability = torch.softmax(torch.from_numpy(human_wake_logits), dim=-1).numpy()[:, WAKE_INDEX]
            synthetic_wake_probability = torch.softmax(torch.from_numpy(synthetic_wake_logits), dim=-1).numpy()[:, WAKE_INDEX]
            false_wakes = int(np.sum(command_probs[:, WAKE_INDEX] >= 0.45))
            human_recall = float(np.mean(human_wake_probability >= 0.45))
            synthetic_recall = float(np.mean(synthetic_wake_probability >= 0.45))
            score = 0.55 * command_f1 + 0.30 * human_recall + 0.15 * synthetic_recall - 5.0 * false_wakes / len(optb_val_y)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, run_dir / "best_model.pt")
        elapsed = time.perf_counter() - started
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_y),
            "train_accuracy": running_correct / len(train_y),
            "option_b_validation_accuracy": command_accuracy,
            "option_b_validation_macro_f1": command_f1,
            "human_wake_validation_recall_at_0_45": human_recall,
            "synthetic_wake_validation_recall_at_0_45": synthetic_recall,
            "option_b_validation_false_wakes_at_0_45": false_wakes,
            "selection_score": score,
            "selected_checkpoint": improved,
            "seconds": elapsed,
        }
        history.append(entry)
        print(
            f"{run_name} epoch {epoch:02d}/{epochs} | loss {entry['train_loss']:.4f} "
            f"| OptionB val F1 {command_f1:.4f} | wake recall {human_recall:.3f} "
            f"| false wakes {false_wakes} | {elapsed:.1f}s" + (" [best]" if improved else ""),
            flush=True,
        )
    _write_json(run_dir / "training_history.json", {"epochs": history})
    if best_state is None:
        raise RuntimeError("Training produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval().cpu()

    fp32_path = models_dir / f"{run_name}_fp32.onnx"
    torch.onnx.export(
        model,
        torch.randn((1, *FEATURE_SHAPE), dtype=torch.float32),
        str(fp32_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    calibration_ids, calibration_manifest = _calibration_indices(
        train_y, train_source_ids, train_source_kinds, is_treatment=is_treatment, seed=seed,
    )
    calibration_x = np.ascontiguousarray(train_x[calibration_ids], dtype=np.float32)
    int8_path = models_dir / f"{run_name}_int8.onnx"
    quantize_static(
        str(fp32_path), str(int8_path), TrainCalibrationReader(calibration_x),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    calibration_hash = sha256_array(calibration_x)
    _write_json(run_dir / "calibration_manifest.json", {
        "split": "train-only",
        "selected_features_sha256": calibration_hash,
        "samples_per_class": 8,
        "selected_sources_by_class": calibration_manifest,
    })
    artifacts = {
        "fp32": {"path": str(fp32_path.relative_to(PROJECT_ROOT)), "size_bytes": fp32_path.stat().st_size, "sha256": sha256_file(fp32_path)},
        "int8": {"path": str(int8_path.relative_to(PROJECT_ROOT)), "size_bytes": int8_path.stat().st_size, "sha256": sha256_file(int8_path)},
    }
    for name, details in artifacts.items():
        if details["size_bytes"] >= 500_000:
            raise AssertionError(f"{name} model exceeds the 500 KB target: {details['size_bytes']} bytes")
    metrics = {
        "status": "TRAINED_EXPORTED_NOT_TEST_EVALUATED",
        "run_name": run_name,
        "model_name": model_name,
        "is_treatment": is_treatment,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation_threshold": 0.45,
        "checkpoint_validation_metrics": history[best_epoch - 1],
        "training_rows": int(len(train_y)),
        "optimizer_steps": int(math.ceil(len(train_y) / batch_size) * epochs),
        "initial_parameters_sha256": initial_hash,
        "artifacts": artifacts,
        "test_accessed": False,
    }
    _write_json(run_dir / "training_metrics.json", metrics)
    print(f"Finished {run_name}; exported FP32/INT8. Test data remains unopened.", flush=True)
    return metrics


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def _inference(session: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(features), 128):
        output.append(session.run(None, {"input": np.asarray(features[start : start + 128], dtype=np.float32)})[0])
    return np.concatenate(output, axis=0) if output else np.empty((0, len(WAKE_LABELS)), dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def _command_metrics(logits: np.ndarray, labels: np.ndarray, label_names: list[str]) -> dict:
    probabilities = _softmax(logits)
    predictions = logits.argmax(axis=-1)
    per_class = {}
    for class_index, label in enumerate(label_names):
        mask = labels == class_index
        if np.any(mask):
            per_class[label] = {
                "support": int(mask.sum()),
                "accuracy": float(np.mean(predictions[mask] == class_index)),
            }
    return {
        "accuracy": float(np.mean(predictions == labels)) if len(labels) else None,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if len(labels) else None,
        "wake_false_accepts_at_0_45": int(np.sum(probabilities[:, WAKE_INDEX] >= 0.45)),
        "wake_false_accept_rate_at_0_45": float(np.mean(probabilities[:, WAKE_INDEX] >= 0.45)) if len(labels) else None,
        "per_class": per_class,
        "predictions": predictions,
        "wake_probabilities": probabilities[:, WAKE_INDEX],
    }


def _load_final_test_data(datasets: dict) -> dict:
    """Called only after both scratch runs are exported and ready."""
    frontend = Frontend()
    cache_path = datasets["optb_cache_path"]
    with np.load(cache_path) as cache:
        optb_test_x = cache["test_x"].astype(np.float32)
        optb_test_y = cache["test_y"].astype(np.int64)

    slurp_test_x, slurp_test_y, slurp_test_sources, slurp_cropped = extract_features_from_manifest(
        datasets["slurp_test_manifest_path"],
        PROJECT_ROOT / "data" / "slurp" / "slurp_test_features.npz",
        force=False,
        expected_split="test",
    )
    if len(slurp_test_x) != len(slurp_test_y):
        raise AssertionError("SLURP test feature/label mismatch")
    if len(slurp_test_y) == 0:
        raise ValueError("No decoded SLURP test examples")

    human_wake_features, human_wake_groups = [], []
    for row in datasets["test_human_wake_refs"]:
        path = (PROJECT_ROOT / row["path"]).resolve()
        waveform, _ = read_mono_16khz(path)
        actual_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
        if actual_hash != row["pcm_sha256"]:
            raise AssertionError(f"Held-out human wake file changed after dedup: {row['path']}")
        human_wake_features.append(frontend(fit_audio(waveform, target_samples=SAMPLES)))
        human_wake_groups.append(row["pcm_sha256"])
    human_wake_x = np.stack(human_wake_features).astype(np.float32)

    synth_features = []
    for relative_path in datasets["test_synthetic_wake_files"]:
        synth_features.append(_extract_file_feature(PROJECT_ROOT / relative_path, frontend))
    synth_wake_x = np.stack(synth_features).astype(np.float32)

    return {
        "optb_test_x": optb_test_x,
        "optb_test_y": optb_test_y,
        "slurp_test_x": slurp_test_x,
        "slurp_test_y": slurp_test_y,
        "slurp_test_source_ids": slurp_test_sources,
        "slurp_test_cropped_rows": slurp_cropped,
        "human_wake_x": human_wake_x,
        "human_wake_groups": human_wake_groups,
        "synth_wake_x": synth_wake_x,
    }


def evaluate_paired_models(
    *,
    control_dir: Path,
    treatment_dir: Path,
    datasets: dict,
    output_path: Path,
) -> dict:
    """One paired final evaluation; safe to rerun the report cell from saved JSON."""
    control_dir, treatment_dir, output_path = Path(control_dir), Path(treatment_dir), Path(output_path)
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    control_prov = json.loads((control_dir / "initial_provenance.json").read_text(encoding="utf-8"))
    treatment_prov = json.loads((treatment_dir / "initial_provenance.json").read_text(encoding="utf-8"))
    control_train = json.loads((control_dir / "training_metrics.json").read_text(encoding="utf-8"))
    treatment_train = json.loads((treatment_dir / "training_metrics.json").read_text(encoding="utf-8"))
    for name, prov in (("control", control_prov), ("treatment", treatment_prov)):
        if prov.get("pretrained_weights_used") is not False:
            raise AssertionError(f"{name} did not prove scratch initialization")
    if control_prov["initial_parameters_sha256"] != treatment_prov["initial_parameters_sha256"]:
        raise AssertionError("Control/treatment initial parameter hashes differ")
    if control_train.get("test_accessed") is not False or treatment_train.get("test_accessed") is not False:
        raise AssertionError("A training run records test access before final paired evaluation")
    for run_dir in (control_dir, treatment_dir):
        if not (run_dir / "models").is_dir() or not list((run_dir / "models").glob("*_fp32.onnx")) or not list((run_dir / "models").glob("*_int8.onnx")):
            raise FileNotFoundError(f"Both FP32 and INT8 exports are required before test loading: {run_dir}")

    # This is the first audio/feature access for any held-out test split.
    tests = _load_final_test_data(datasets)
    results_by_run = {}
    for run_name, run_dir in (("control", control_dir), ("treatment", treatment_dir)):
        run_models = {}
        predictions_by_quantization = {}
        model_paths = {
            "fp32": next((run_dir / "models").glob("*_fp32.onnx")),
            "int8": next((run_dir / "models").glob("*_int8.onnx")),
        }
        for quantization, model_path in model_paths.items():
            session = _session(model_path)
            optb = _command_metrics(_inference(session, tests["optb_test_x"]), tests["optb_test_y"], LABELS)
            slurp = _command_metrics(_inference(session, tests["slurp_test_x"]), tests["slurp_test_y"], WAKE_LABELS[: len(LABELS)])
            human_logits = _inference(session, tests["human_wake_x"])
            human_wake_prob = _softmax(human_logits)[:, WAKE_INDEX]
            synth_logits = _inference(session, tests["synth_wake_x"])
            synth_wake_prob = _softmax(synth_logits)[:, WAKE_INDEX]
            predictions_by_quantization[quantization] = {
                "option_b": optb["predictions"],
                "slurp_real": slurp["predictions"],
            }
            run_models[quantization] = {
                "option_b_test": {
                    key: value for key, value in optb.items() if key not in ("predictions", "wake_probabilities")
                },
                "slurp_real_test": {
                    key: value for key, value in slurp.items() if key not in ("predictions", "wake_probabilities")
                },
                "human_wake_test": {
                    "unique_recordings": len(tests["human_wake_groups"]),
                    "recall_at_0_45": float(np.mean(human_wake_prob >= 0.45)),
                    "probabilities": human_wake_prob.tolist(),
                },
                "synthetic_wake_test": {
                    "files": len(tests["synth_wake_x"]),
                    "recall_at_0_45": float(np.mean(synth_wake_prob >= 0.45)),
                },
            }
        results_by_run[run_name] = {
            **run_models,
            "fp32_int8_prediction_parity": {
                "option_b_test": float(np.mean(predictions_by_quantization["fp32"]["option_b"] == predictions_by_quantization["int8"]["option_b"])),
                "slurp_real_test": float(np.mean(predictions_by_quantization["fp32"]["slurp_real"] == predictions_by_quantization["int8"]["slurp_real"])),
            },
        }

    optionb_manifest_rows = list(csv.DictReader(datasets["optionb_manifest_path"].open("r", encoding="utf-8", newline="")))
    optionb_test_speakers = len({row["speaker"] for row in optionb_manifest_rows if row.get("split") == "test"})
    test_source_summary = {
        "option_b_test_rows": int(len(tests["optb_test_y"])),
        "option_b_test_speakers": optionb_test_speakers,
        "slurp_test_rows": int(len(tests["slurp_test_y"])),
        "slurp_test_speakers": len({source.split(":")[1] for source in tests["slurp_test_source_ids"]}),
        "slurp_test_source_groups": len(set(tests["slurp_test_source_ids"])),
        "slurp_test_cropped_rows": tests["slurp_test_cropped_rows"],
        "slurp_test_support_by_class": dict(sorted(Counter(WAKE_LABELS[int(i)] for i in tests["slurp_test_y"]).items())),
        "human_wake_unique_recordings": len(tests["human_wake_groups"]),
        "synthetic_wake_files": len(tests["synth_wake_x"]),
    }
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluation_threshold": 0.45,
        "test_sets_loaded_once_after_both_runs_fixed": True,
        "no_test_use_in_training_or_checkpoint_selection": True,
        "initial_parameters_sha256": control_prov["initial_parameters_sha256"],
        "test_source_summary": test_source_summary,
        "control": results_by_run["control"],
        "treatment": results_by_run["treatment"],
        "treatment_minus_control_int8": {
            "option_b_accuracy": results_by_run["treatment"]["int8"]["option_b_test"]["accuracy"]
            - results_by_run["control"]["int8"]["option_b_test"]["accuracy"],
            "option_b_macro_f1": results_by_run["treatment"]["int8"]["option_b_test"]["macro_f1"]
            - results_by_run["control"]["int8"]["option_b_test"]["macro_f1"],
            "slurp_real_accuracy": results_by_run["treatment"]["int8"]["slurp_real_test"]["accuracy"]
            - results_by_run["control"]["int8"]["slurp_real_test"]["accuracy"],
            "slurp_real_macro_f1": results_by_run["treatment"]["int8"]["slurp_real_test"]["macro_f1"]
            - results_by_run["control"]["int8"]["slurp_real_test"]["macro_f1"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    for run_name, run_dir in (("control", control_dir), ("treatment", treatment_dir)):
        _write_json(run_dir / "test_metrics.json", {
            "one_time_paired_test_evaluation": result[run_name],
            "test_source_summary": test_source_summary,
            "test_evaluation_threshold": 0.45,
        })
    print(f"[SAVED] Final paired held-out metrics: {output_path}", flush=True)
    return result
