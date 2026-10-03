"""Evaluate the deployed intent and binary wake ONNX models on labeled WAVs.

The input CSV columns are dataset, mode, label, and path. Paths are resolved
against --audio-root. This script runs model inference only; it does not start
the assistant, capture a microphone, or execute GPIO/device actions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_model.config import LABELS, SAMPLES, SR
from tinyvcm_model.frontend import Frontend, fit_audio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("WAV must contain finite, non-empty mono audio")
    sample_rate = int(sample_rate)
    if sample_rate != SR:
        divisor = int(np.gcd(sample_rate, SR))
        audio = resample_poly(audio, SR // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = SR
    return audio.astype(np.float32, copy=False), sample_rate


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def class_metrics(rows: list[dict], labels: list[str]) -> dict:
    confusion = Counter((row["label"], row["prediction"]) for row in rows)
    support = Counter(row["label"] for row in rows)
    predicted = Counter(row["prediction"] for row in rows)
    correct = sum(confusion[(label, label)] for label in labels)
    by_class = {}
    f1_values = []
    for label in labels:
        tp = confusion[(label, label)]
        n = support[label]
        fp = predicted[label] - tp
        fn = n - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n if n else None
        f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else (0.0 if recall is not None else None)
        if n:
            f1_values.append(f1)
        by_class[label] = {
            "support": n,
            "correct": tp,
            "accuracy_recall": round(recall, 6) if recall is not None else None,
            "precision": round(precision, 6),
            "f1": round(f1, 6) if f1 is not None else None,
        }
    total = len(rows)
    confusion_pairs = [
        {"actual": actual, "predicted": prediction, "count": count}
        for (actual, prediction), count in confusion.most_common()
        if actual != prediction
    ][:20]
    missing = [label for label in labels if not support[label]]
    return {
        "count": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "macro_f1_supported_classes": round(statistics.mean(f1_values), 6) if f1_values else None,
        "classes_supported": len(labels) - len(missing),
        "classes_expected": len(labels),
        "missing_classes": missing,
        "per_class": by_class,
        "top_confusions": confusion_pairs,
    }


def binary_metrics(rows: list[dict]) -> dict:
    metrics = class_metrics(rows, ["NON_WAKE", "WAKE_WORD"])
    tp = sum(row["label"] == row["prediction"] == "WAKE_WORD" for row in rows)
    fp = sum(row["label"] == "NON_WAKE" and row["prediction"] == "WAKE_WORD" for row in rows)
    positives = sum(row["label"] == "WAKE_WORD" for row in rows)
    negatives = sum(row["label"] == "NON_WAKE" for row in rows)
    metrics.update({
        "wake_hits": tp,
        "wake_support": positives,
        "wake_recall": round(tp / positives, 6) if positives else None,
        "false_accepts": fp,
        "nonwake_support": negatives,
        "false_accept_rate": round(fp / negatives, 6) if negatives else None,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--intent-model", type=Path, required=True)
    parser.add_argument("--wake-model", type=Path, required=True)
    parser.add_argument("--wake-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.wake_threshold <= 1.0:
        raise SystemExit("--wake-threshold must be in [0, 1]")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    intent_session = ort.InferenceSession(str(args.intent_model), sess_options=options, providers=["CPUExecutionProvider"])
    wake_session = ort.InferenceSession(str(args.wake_model), sess_options=options, providers=["CPUExecutionProvider"])
    if intent_session.get_outputs()[0].shape[-1] != len(LABELS):
        raise ValueError("Intent ONNX output width does not match the 31-class label list")
    if wake_session.get_outputs()[0].shape[-1] != 2:
        raise ValueError("Wake ONNX output width is not binary")

    frontend = Frontend()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_rates: Counter = Counter()
    errors = []
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        manifest_rows = list(csv.DictReader(stream))
    required = {"dataset", "mode", "label", "path"}
    if not manifest_rows or not required.issubset(manifest_rows[0]):
        raise ValueError(f"Manifest must contain columns {sorted(required)}")

    for index, item in enumerate(manifest_rows, 1):
        dataset, mode, expected = item["dataset"], item["mode"], item["label"]
        if mode not in ("command", "wake"):
            raise ValueError(f"Invalid mode on manifest row {index}: {mode}")
        if mode == "command" and expected not in LABELS:
            raise ValueError(f"Unknown command label on manifest row {index}: {expected}")
        if mode == "wake" and expected not in ("WAKE_WORD", "NON_WAKE"):
            raise ValueError(f"Unknown wake label on manifest row {index}: {expected}")
        path = (args.audio_root / item["path"]).resolve()
        if not path.is_relative_to(args.audio_root.resolve()):
            raise ValueError(f"Manifest path escapes the audio root: {item['path']}")
        try:
            audio, sample_rate = read_audio(path)
            sample_rates[sample_rate] += 1
            started = time.perf_counter()
            if mode == "command":
                features = frontend(fit_audio(audio))[None]
                logits = intent_session.run(None, {intent_session.get_inputs()[0].name: features})[0][0]
                probs = probabilities(logits)
                prediction = LABELS[int(np.argmax(probs))]
                score = float(np.max(probs))
            else:
                if len(audio) > SAMPLES:
                    audio = fit_audio(audio)
                max_start = max(0, SAMPLES - len(audio))
                wake_probs = []
                for start in np.linspace(0, max_start, 5, dtype=int):
                    window = np.zeros(SAMPLES, dtype=np.float32)
                    window[start:start + len(audio)] = audio
                    features = frontend(window)[None]
                    logits = wake_session.run(None, {wake_session.get_inputs()[0].name: features})[0][0]
                    wake_probs.append(float(probabilities(logits)[1]))
                score = max(wake_probs)
                prediction = "WAKE_WORD" if score >= args.wake_threshold else "NON_WAKE"
            elapsed = (time.perf_counter() - started) * 1000
            grouped[(dataset, mode)].append({"label": expected, "prediction": prediction, "score": score})
            latencies[(dataset, mode)].append(elapsed)
        except Exception as exc:  # keep the report complete, but never silently drop a failed sample
            errors.append({"dataset": dataset, "mode": mode, "path": item["path"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 500 == 0:
            print(f"evaluated {index}/{len(manifest_rows)} rows", flush=True)

    reports = {}
    for (dataset, mode), rows in sorted(grouped.items()):
        result = binary_metrics(rows) if mode == "wake" else class_metrics(rows, LABELS)
        times = latencies[(dataset, mode)]
        result["latency_ms_end_to_end"] = {
            "p50": round(float(np.percentile(times, 50)), 3) if times else None,
            "p95": round(float(np.percentile(times, 95)), 3) if times else None,
        }
        result["audio_sample_rates_after_resampling"] = dict(sample_rates)
        reports[f"{dataset}:{mode}"] = result

    report = {
        "title": "ME2 - VCM on Raspberry Pi 5 — file-based command and wake evaluation",
        "host": platform.node(),
        "platform": platform.platform(),
        "providers": {"intent": intent_session.get_providers(), "wake": wake_session.get_providers()},
        "wake_threshold": args.wake_threshold,
        "labels": LABELS,
        "models": {
            "intent": {"path": str(args.intent_model), "bytes": args.intent_model.stat().st_size, "sha256": sha256(args.intent_model)},
            "wake": {"path": str(args.wake_model), "bytes": args.wake_model.stat().st_size, "sha256": sha256(args.wake_model)},
        },
        "manifest_rows": len(manifest_rows),
        "evaluated_rows": sum(len(rows) for rows in grouped.values()),
        "failed_rows": errors,
        "results": reports,
        "protocol": {
            "intent": "Top-1 classification from the 31-class INT8 ONNX using the deployed 16 kHz, 2.5 s, 40-band log-mel frontend; no device actions are executed.",
            "wake": "Binary INT8 ONNX at the deployed 0.95 threshold; five temporal placements per clip, max wake probability decides the label.",
            "audio": "Mono conversion and polyphase resampling to 16 kHz are applied when needed; inference reads one WAV at a time.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"host": report["host"], "evaluated_rows": report["evaluated_rows"], "failed_rows": len(errors), "results": {k: {"count": v["count"], "accuracy": v["accuracy"], "macro_f1_supported_classes": v.get("macro_f1_supported_classes"), "wake_recall": v.get("wake_recall"), "false_accepts": v.get("false_accepts")} for k, v in reports.items()}}, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} audio rows failed; see {args.output}")


if __name__ == "__main__":
    main()
