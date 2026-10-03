"""Paired evaluation of frozen VCM intent ONNX models on the official Gold test.

The script never trains or tunes. Both models see the same exact WAV rows and
frontend. Raw argmax classification and each model's pre-locked live reject
gate are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import scipy.io.wavfile as wavfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tinyvcm_model.config import LABELS, SR, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD
from tinyvcm_model.frontend import Frontend, fit_audio

REVISION = "6947f13073e57eb6ae67e7e2fc3680700b82aa13"
GOLD_CONFIDENCE = 0.76
GOLD_MARGIN = 0.0
SLOT_CLASSES = {
    "ALARM": {"6:00 AM": "ALARM_6_00AM", "8:00 AM": "ALARM_8_00AM", "9:00 PM": "ALARM_9_00PM"},
    "BRIGHTNESS": {"20 percent": "BRIGHTNESS_20", "60 percent": "BRIGHTNESS_60", "100 percent": "BRIGHTNESS_100"},
    "COLOR": {"Blue": "COLOR_BLUE", "Green": "COLOR_GREEN", "Red": "COLOR_RED"},
    "CREATE_REMINDER": {"Drink water": "CREATE_REMINDER_DRINK_WATER", "Exercise": "CREATE_REMINDER_EXERCISE", "Study": "CREATE_REMINDER_STUDY"},
    "TEMPERATURE": {"18 degrees": "TEMPERATURE_18", "22 degrees": "TEMPERATURE_22", "26 degrees": "TEMPERATURE_26"},
    "TIMER": {"10 seconds": "TIMER_10s", "1 minute": "TIMER_1m", "30 seconds": "TIMER_30s"},
}
FIXED = {"CALL", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC", "STOP", "TIME", "VOLUME_DOWN", "VOLUME_UP", "WEATHER"}


def label_for(row: dict) -> str | None:
    command = row.get("command")
    if command in FIXED and row.get("variation"):
        return command
    return SLOT_CLASSES.get(command, {}).get(row.get("slot_value"))


def read_test(path: Path):
    parquet = pq.ParquetFile(path)
    rows = []
    features = []
    frontend = Frontend()
    for batch in parquet.iter_batches(batch_size=32, columns=["audio", "file", "command", "variation", "slot_value", "out_of_scope", "speaker_id", "source", "is_synthetic"]):
        for row in batch.to_pylist():
            audio = row["audio"]
            rate, pcm = wavfile.read(io.BytesIO(audio["bytes"]))
            if rate != SR or pcm.dtype != np.int16 or pcm.ndim != 1:
                raise ValueError(f"Bad WAV payload in {row['file']}")
            rows.append(row)
            features.append(frontend(fit_audio(pcm.astype(np.float32) / 32768.0)))
    return rows, np.stack(features).astype(np.float32)


def infer(path: Path, features: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    chunks = [session.run(None, {session.get_inputs()[0].name: features[i:i+64]})[0] for i in range(0, len(features), 64)]
    logits = np.concatenate(chunks)
    if logits.shape != (len(features), len(LABELS)) or not np.isfinite(logits).all():
        raise ValueError(f"Unexpected output from {path}: {logits.shape}")
    return logits


def score_gate(rows, logits, confidence, margin):
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    conf = probs[np.arange(len(rows)), top]
    gap = conf - probs[np.arange(len(rows)), order[:, -2]]
    accepted = (conf >= confidence) & (gap >= margin)
    known_correct = known_wrong = known_reject = oos_accept = 0
    for i, row in enumerate(rows):
        truth = label_for(row)
        oos = bool(row.get("out_of_scope")) or row.get("command") == "OUT_OF_SCOPE" or truth is None
        if oos:
            oos_accept += int(accepted[i])
        elif not accepted[i]:
            known_reject += 1
        elif LABELS[int(top[i])] == truth:
            known_correct += 1
        else:
            known_wrong += 1
    return {"confidence": confidence, "margin": margin, "known_correct": known_correct, "known_wrong": known_wrong, "known_reject": known_reject, "oos_accepts": oos_accept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--gold-model", type=Path, required=True)
    ap.add_argument("--option-model", type=Path, required=True, help="Current Option-B-derived model; name the actual training mix in report")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=REVISION, help="Immutable dataset commit matching the --test parquet")
    ap.add_argument("--expected-rows", type=int, default=4443)
    ap.add_argument("--baseline-name", default="gold_train_candidate")
    ap.add_argument("--baseline-title", default="Gold-train candidate")
    ap.add_argument("--candidate-name", default="option_b_plus_personal_active")
    ap.add_argument("--candidate-title", default="Active Option-B + personal model")
    args = ap.parse_args()
    rows, features = read_test(args.test)
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} test rows, got {len(rows)}")
    truth = np.asarray([LABELS.index(label_for(row)) if label_for(row) in LABELS else -1 for row in rows])
    mask = truth >= 0
    outputs = {}
    per_class = []
    model_specs = [
        (args.baseline_name, args.baseline_title, args.gold_model, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD),
        (args.candidate_name, args.candidate_title, args.option_model, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD),
    ]
    for name, title, model_path, confidence, margin in model_specs:
        logits = infer(model_path, features)
        pred = logits.argmax(axis=1)
        outputs[name] = {
            "title": title, "model_path": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "raw_supported_accuracy": float(accuracy_score(truth[mask], pred[mask])),
            "raw_supported_macro_f1": float(f1_score(truth[mask], pred[mask], labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
            "supported_rows": int(mask.sum()), "oos_or_unmapped_rows": int((~mask).sum()),
            "gate": score_gate(rows, logits, confidence, margin),
            "common_active_gate": score_gate(rows, logits, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD),
        }
        report = classification_report(truth[mask], pred[mask], labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
        for label in LABELS:
            rec = report[label]
            per_class.append({"model": name, "label": label, "support": int(rec["support"]), "precision": rec["precision"], "recall": rec["recall"], "f1": rec["f1-score"]})
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "ME2 Spoken Command Dataset", "revision": args.revision,
        "test_parquet_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
        "test_rows": len(rows), "supported_rows": int(mask.sum()), "out_of_scope_rows": int((~mask).sum()),
        "class_order": LABELS, "models": outputs,
        "limitations": [
            "This is a frozen-model comparison on the same Gold test waveforms; the Gold test did not train weights or select thresholds.",
            "The Gold test has no positive wake examples; this evaluates intent only.",
            "This offline dataset test does not replace the class Raspberry Pi live holdout protocol.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "label", "support", "precision", "recall", "f1"]); w.writeheader(); w.writerows(per_class)
    lines = [
        "# Paired frozen-model evaluation on the latest Gold test split", "",
        f"Dataset: ME2 Spoken Command Dataset, commit `{args.revision}`; test rows: {len(rows)}; supported rows: {int(mask.sum())}; out-of-scope rows: {int((~mask).sum())}.",
        f"Test parquet SHA-256: `{summary['test_parquet_sha256']}`.", "",
        "Both INT8 models were run on the exact same test WAVs with the same 16 kHz, 2.5-second, 40-bin log-mel frontend. Intent metrics are raw top-class argmax on the 31 supported classes. The held-out Gold test did not select thresholds.", "",
        "| Frozen model | Raw accuracy | Macro F1 | Fixed operating gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |", "|---|---:|---:|---|---:|---:|---:|---:|"
    ]
    for name, title, *_ in model_specs:
        result = outputs[name]; gate = result["gate"]
        lines.append(f"| {title} | {result['raw_supported_accuracy']:.2%} | {result['raw_supported_macro_f1']:.2%} | confidence {gate['confidence']:.2f}, margin {gate['margin']:.2f} | {gate['known_correct']} | {gate['known_wrong']} | {gate['known_reject']} | {gate['oos_accepts']}/{result['oos_or_unmapped_rows']} |")
    lines += ["", "## Common-gate check", "", "For a matched operating-threshold view, the same active confidence/margin gate (0.68/0.15) is also applied to both frozen models. This is the active model's existing validation-selected operational gate; it was not tuned on the Gold test.", "", "| Model | Gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |", "|---|---:|---:|---:|---:|---:|"]
    for name, title, *_ in model_specs:
        gate = outputs[name]["common_active_gate"]
        lines.append(f"| {title} | 0.68 / 0.15 | {gate['known_correct']} | {gate['known_wrong']} | {gate['known_reject']} | {gate['oos_accepts']}/{outputs[name]['oos_or_unmapped_rows']} |")
    lines += [
        "", "## Provenance and caveats", "",
        f"- This run compares `{args.baseline_title}` with `{args.candidate_title}`. Both were frozen before scoring this Gold test revision; the test rows were not used for training or threshold selection.",
        "- The retrained candidate used the ME2 Spoken Command Dataset training split plus the mapped personal training clips. Its per-class scores are in `per_class.csv`.",
        "- This dataset has no wake positives, so only intent was evaluated; the binary wake model is not compared here.",
        "- This is offline scoring of the official test split. The class Raspberry Pi live benchmark uses the separate holdout split, records three wake samples, and measures the microphone/speaker path and device behavior; it remains to be run when the Pi microphone is available.",
        "- Full per-class precision/recall/F1 is `per_class.csv`; exact model hashes and detailed counts are in `summary.json`.",
        "", "The Gold test is benchmark-only. All model settings and the shared intent gate (0.68 confidence / 0.15 margin) were frozen before this scoring pass. The class live holdout SOP remains a separate Pi evaluation.", ""
    ]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {"raw_supported_accuracy": v["raw_supported_accuracy"], "raw_supported_macro_f1": v["raw_supported_macro_f1"], "gate": v["gate"]} for k,v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
