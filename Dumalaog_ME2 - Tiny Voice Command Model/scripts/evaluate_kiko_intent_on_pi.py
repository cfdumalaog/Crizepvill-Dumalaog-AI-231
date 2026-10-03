"""Run the installed Kiko ONNX intent/slot model on the frozen ME2 test set.

The Kiko folder is treated as read-only. Temporary features and scores are
staged under /tmp on the Pi, then removed. Feature extraction runs locally with
the exact librosa calls present in Kiko's Pi controller because its currently
configured Pi runtime environment does not have librosa installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.io import wavfile


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "ai231-me2-voice-commands-hf-a90b8d10"
REVISION = "a90b8d106349b02c5570a1a258503386043f63b2"
EXPECTED_KIKO_MODEL_SHA = "9924f24dc376b385f9a2754d2559aeb9850581b012bbf260d427b8b887b075b5"
EXPECTED_KIKO_LABELS_SHA = "55da67a438989bcea970f76ee4dde13ba2ff7d7b13a55535e49d0c1f4c267a8b"
EXPECTED_KIKO_SLOTS_SHA = "8b51fd535bfc97b679d9e0ac74e8f3fb932158f47e3291a9daf6aaf775d6693e"
KIKO_MODEL = "/home/dalmacio/Desktop/kiko/pi_deployment/vcm_int8.onnx"
PI_PYTHON = "/home/dalmacio/.venvs/tinyvcm-rpi5/bin/python"
PI_HOST = "dalmacio@cfdfnjrpi5.local"
KIKO_INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE", "STOP", "NEXT",
    "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM",
    "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER", "UNKNOWN",
]
KIKO_SLOT_VALUES = [
    "No Slot", "10 seconds", "30 seconds", "1 minute", "6 AM", "8 AM", "9 PM",
    "18 degrees", "22 degrees", "26 degrees", "20 percent", "60 percent", "100 percent",
    "red", "blue", "green", "drink water", "study", "exercise",
]
SLOT_COMMANDS: dict[str, dict[str, str]] = {
    "TIMER": {"10 seconds": "TIMER_10s", "30 seconds": "TIMER_30s", "1 minute": "TIMER_1m"},
    "ALARM": {"6 am": "ALARM_6_00AM", "6:00 am": "ALARM_6_00AM", "8 am": "ALARM_8_00AM", "8:00 am": "ALARM_8_00AM", "9 pm": "ALARM_9_00PM", "9:00 pm": "ALARM_9_00PM"},
    "TEMPERATURE": {"18 degrees": "TEMPERATURE_18", "22 degrees": "TEMPERATURE_22", "26 degrees": "TEMPERATURE_26"},
    "BRIGHTNESS": {"20 percent": "BRIGHTNESS_20", "60 percent": "BRIGHTNESS_60", "100 percent": "BRIGHTNESS_100"},
    "COLOR": {"red": "COLOR_RED", "green": "COLOR_GREEN", "blue": "COLOR_BLUE"},
    "CREATE_REMINDER": {"drink water": "CREATE_REMINDER_DRINK_WATER", "exercise": "CREATE_REMINDER_EXERCISE", "study": "CREATE_REMINDER_STUDY"},
}
SLOT_INTENTS = set(SLOT_COMMANDS)
FIXED_COMMANDS = {x for x in KIKO_INTENTS if x not in SLOT_INTENTS | {"UNKNOWN"}}
MODEL_SHA_OURS = "4af9e1a0a6fb8a3a6c98eb02ed49d02dab6b92b0f18c47adbadaffb77ced35d1"
OURS_RESULTS = PROJECT / "runs" / "classagreementvcm-20261002-a90b8d10-r3" / "frozen_test_comparison.json"
RESULT_DIR = PROJECT / "docs" / "evaluations" / "kiko-pi-comparison-20261002"


def command_leaf(command: str, slot: str, out_of_scope: int) -> str:
    if out_of_scope or command == "OUT_OF_SCOPE":
        return "UNKNOWN"
    if command in SLOT_COMMANDS:
        try:
            return SLOT_COMMANDS[command][slot.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"Unmapped Kiko dataset slot: {command} / {slot}") from exc
    if command in FIXED_COMMANDS:
        return command
    raise ValueError(f"Unsupported command in frozen test: {command}")


def fit_runtime_window(y: np.ndarray, samples: int = 40_000) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) < samples:
        diff = samples - len(y)
        return np.pad(y, (diff // 2, diff - diff // 2))
    if len(y) > samples:
        start = (len(y) - samples) // 2
        return y[start:start + samples]
    return y


def make_feature_batch(audio_dicts: list[dict[str, Any]]) -> np.ndarray:
    logmels: list[np.ndarray] = []
    for audio in audio_dicts:
        rate, raw = wavfile.read(io.BytesIO(audio["bytes"]))
        if raw.ndim > 1:
            raw = raw.astype(np.float32).mean(axis=1)
        if raw.dtype.kind in "iu":
            wave = librosa.util.buf_to_float(raw)
        else:
            wave = raw.astype(np.float32)
        if rate != 16_000:
            wave = librosa.resample(wave, orig_sr=rate, target_sr=16_000)
        # Match Kiko's checked inference code: transform the provided utterance,
        # then take/pad the first 251 frames. Its live recorder itself captures
        # a 2.5-second window; test files are retained at their source duration.
        mel = librosa.feature.melspectrogram(
            y=wave, sr=16_000, n_fft=512, hop_length=160, win_length=512,
            center=True, pad_mode="reflect", n_mels=64, htk=True, norm="slaney",
        )
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        if logmel.shape[1] > 251:
            logmel = logmel[:, :251]
        elif logmel.shape[1] < 251:
            logmel = np.pad(logmel, ((0, 0), (0, 251 - logmel.shape[1])), mode="constant")
        logmels.append(logmel)
    logmel = np.stack(logmels)
    if logmel.shape[1:] != (64, 251):
        raise ValueError(f"Unexpected Kiko runtime feature shape: {logmel.shape}")
    return logmel[:, None, :, :].astype(np.float32, copy=False)


def build_test_payload(destination: Path) -> dict[str, Any]:
    shard = DATASET / "data" / "test-00000-of-00001.parquet"
    features: list[np.ndarray] = []
    commands: list[str] = []
    leaves: list[str] = []
    slots: list[str] = []
    oos_flags: list[bool] = []
    sources: list[str] = []
    started = time.perf_counter()
    parquet = pq.ParquetFile(shard)
    total_batches = (parquet.metadata.num_rows + 31) // 32
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=32,
            columns=["audio", "command", "slot_value", "out_of_scope", "source"],
        ),
        1,
    ):
        rows = batch.to_pylist()
        features.append(make_feature_batch([row["audio"] for row in rows]))
        for row in rows:
            oos = bool(int(row.get("out_of_scope") or 0)) or row["command"] == "OUT_OF_SCOPE"
            commands.append("UNKNOWN" if oos else row["command"])
            slot = row.get("slot_value") or ""
            slots.append(slot)
            leaves.append(command_leaf(row["command"], slot, int(oos)))
            oos_flags.append(oos)
            sources.append(row.get("source") or "")
        if batch_index % 20 == 0 or batch_index == total_batches:
            print(f"Feature batches: {batch_index}/{total_batches}; rows: {len(commands)}/{parquet.metadata.num_rows}", flush=True)
    x = np.concatenate(features, axis=0)
    if len(x) != 4418:
        raise ValueError(f"Unexpected frozen test size: {len(x)}")
    np.savez_compressed(
        destination,
        features=x,
        gold_command=np.asarray(commands, dtype="U32"),
        gold_leaf=np.asarray(leaves, dtype="U48"),
        gold_slot=np.asarray(slots, dtype="U40"),
        out_of_scope=np.asarray(oos_flags, dtype=np.bool_),
        source=np.asarray(sources, dtype="U40"),
    )
    return {
        "rows": len(x),
        "supported": int(np.sum(~np.asarray(oos_flags, dtype=np.bool_))),
        "true_oos": int(np.sum(oos_flags)),
        "features_shape": list(x.shape),
        "feature_bytes_uncompressed": int(x.nbytes),
        "payload_bytes_compressed": destination.stat().st_size,
        "feature_seconds": time.perf_counter() - started,
        "dataset_revision": REVISION,
    }


REMOTE_RUNNER = r'''import hashlib,json,sys,time
import numpy as np
import onnxruntime as ort
model, inp, out = sys.argv[1:4]
h=hashlib.sha256(open(model,"rb").read()).hexdigest()
if h != "9924f24dc376b385f9a2754d2559aeb9850581b012bbf260d427b8b887b075b5": raise SystemExit("Kiko model SHA mismatch")
z=np.load(inp)
x=z["features"]
so=ort.SessionOptions()
session=ort.InferenceSession(model,sess_options=so,providers=["CPUExecutionProvider"])
if session.get_inputs()[0].shape != [1,1,64,251]: raise SystemExit("Unexpected Kiko input shape")
for _ in range(20): session.run(None,{"input":x[0:1]})
intent=[]; slot=[]; times=[]
for i in range(len(x)):
    t=time.perf_counter_ns()
    yi,ys=session.run(None,{"input":x[i:i+1]})
    times.append((time.perf_counter_ns()-t)/1e6)
    intent.append(yi[0]); slot.append(ys[0])
np.savez_compressed(out,intent_logits=np.asarray(intent,dtype=np.float32),slot_logits=np.asarray(slot,dtype=np.float32),inference_ms=np.asarray(times,dtype=np.float32))
print(json.dumps({"model_sha256":h,"rows":len(x),"input_shape":list(x.shape),"intent_shape":list(np.asarray(intent).shape),"slot_shape":list(np.asarray(slot).shape),"providers":session.get_providers(),"warmup":20,"p50_model_ms":float(np.percentile(times,50)),"p95_model_ms":float(np.percentile(times,95)),"max_model_ms":float(np.max(times))}))
'''


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result


def softmax_confidence(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    ids = probs.argmax(axis=1)
    confidence = probs[np.arange(len(probs)), ids]
    return ids, confidence


def score_outputs(test_payload: Path, predictions: Path, pi_meta: dict[str, Any]) -> dict[str, Any]:
    x = np.load(test_payload)
    y = np.load(predictions)
    gold_leaf = x["gold_leaf"].astype(str)
    gold_command = x["gold_command"].astype(str)
    oos = x["out_of_scope"]
    intent_logits = y["intent_logits"].astype(np.float64)
    slot_logits = y["slot_logits"].astype(np.float64)
    intent_ids, intent_conf = softmax_confidence(intent_logits)
    slot_ids, slot_conf = softmax_confidence(slot_logits)
    raw_intents = np.asarray(KIKO_INTENTS, dtype=object)[intent_ids]
    raw_slots = np.asarray(KIKO_SLOT_VALUES, dtype=object)[slot_ids]
    predicted_leaf: list[str] = []
    effective_intent: list[str] = []
    for intent, slot, confidence in zip(raw_intents, raw_slots, intent_conf):
        if confidence < 0.30:
            effective_intent.append("UNKNOWN")
            predicted_leaf.append("UNKNOWN")
        else:
            effective_intent.append(str(intent))
            if intent in SLOT_INTENTS:
                predicted_leaf.append(SLOT_COMMANDS[str(intent)].get(str(slot).strip().lower(), "UNKNOWN"))
            else:
                predicted_leaf.append(str(intent))
    predicted_leaf_array = np.asarray(predicted_leaf, dtype=str)
    effective_intents = np.asarray(effective_intent, dtype=str)
    supported = ~oos
    gold_supported_leaf = gold_leaf[supported]
    raw_supported_leaf = predicted_leaf_array[supported]
    raw_supported_intents = raw_intents[supported]
    accepted = (effective_intents[supported] != "UNKNOWN")
    correct_leaf = raw_supported_leaf == gold_supported_leaf
    exact_accepted = correct_leaf & accepted
    fixed_core = sorted({x for x in gold_supported_leaf})
    top_labels = sorted(set(gold_command[supported]))
    oos_false_action = int(np.sum(effective_intents[oos] != "UNKNOWN"))
    per_label = classification_report(
        gold_supported_leaf,
        raw_supported_leaf,
        labels=fixed_core,
        output_dict=True,
        zero_division=0,
    )
    top_level = {
        "accuracy_raw": float(accuracy_score(gold_command[supported], raw_supported_intents)),
        "macro_f1_raw": float(f1_score(gold_command[supported], raw_supported_intents, labels=top_labels, average="macro", zero_division=0)),
    }
    accepted_count = int(np.sum(accepted))
    correct_accepted = int(np.sum(exact_accepted))
    paired = json.loads(OURS_RESULTS.read_text(encoding="utf-8"))
    return {
        "dataset": "ME2 Spoken Command Dataset",
        "revision": REVISION,
        "split": "published test (previously used once by the in-project paired comparison; this run is a diagnostic and not a pristine evaluation)",
        "supports": {"total": len(gold_leaf), "supported": int(np.sum(supported)), "true_oos": int(np.sum(oos))},
        "kiko": {
            "model_sha256": pi_meta["model_sha256"],
            "model_bytes": 83163,
            "output_heads": {"intent_classes": len(KIKO_INTENTS), "slot_classes": len(KIKO_SLOT_VALUES)},
            "confidence_gate": 0.30,
            "supported_leaf_accuracy_at_030_gate": float(accuracy_score(gold_supported_leaf, raw_supported_leaf)),
            "supported_leaf_macro_f1_at_030_gate": float(f1_score(gold_supported_leaf, raw_supported_leaf, labels=fixed_core, average="macro", zero_division=0)),
            "top_level": top_level,
            "accepted_supported_coverage": accepted_count / int(np.sum(supported)),
            "accuracy_among_accepted_actions": correct_accepted / accepted_count if accepted_count else None,
            "correct_action_coverage": correct_accepted / int(np.sum(supported)),
            "accepted_supported": accepted_count,
            "correct_accepted": correct_accepted,
            "oos_false_actions": oos_false_action,
            "oos_false_action_rate": oos_false_action / int(np.sum(oos)) if np.sum(oos) else None,
            "oos_correct_rejections": int(np.sum(effective_intents[oos] == "UNKNOWN")),
            "inference_timing_pi_model_only": pi_meta,
            "per_leaf": {label: per_label[label] for label in fixed_core},
        },
        "ours_saved_comparison": {
            "candidate_run": "classagreementvcm-20261002-a90b8d10-r3",
            "candidate_model_sha256": MODEL_SHA_OURS,
            "baseline": paired["baseline"],
            "candidate": paired["candidate"],
            "per_class_delta": paired["per_class"],
            "note": "These are prior saved results; our model was not rerun as part of the Kiko Pi trial.",
        },
    }


def write_reports(result: dict[str, Any], raw_results: Path | None = None) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    kiko = result["kiko"]
    ours = result["ours_saved_comparison"]
    base = ours["baseline"]["raw_supported"]
    cand = ours["candidate"]["raw_supported"]
    cand_top = ours["candidate"]["top_level_19"]
    base_top = ours["baseline"]["top_level_19"]
    base_gate = ours["baseline"]["validation_calibrated_gate"]["results"]
    cand_gate = ours["candidate"]["validation_calibrated_gate"]["results"]
    rows = [
        "# ME2 - VCM on Raspberry Pi 5: Kiko model comparison",
        "",
        "**Date:** 2026-10-02 (Asia/Manila)  ",
        "**Test:** Installed Kiko ONNX model executed on the Raspberry Pi 5 CPU; same 4,418 published test clips and local pinned revision `a90b8d106349b02c5570a1a258503386043f63b2`.  ",
        "**Scope:** Inference-only diagnostic. No training, GPIO, dashboard action, microphone, file changes inside `~/Desktop/kiko`, or deployment was performed.",
        "",
        "## Main result",
        "",
        f"At Kiko's fixed 30% intent-confidence gate, it made the correct exact command/value action on **{kiko['correct_action_coverage']:.2%}** of supported clips. It accepted **{kiko['accepted_supported_coverage']:.2%}** of supported clips and was correct on **{kiko['accuracy_among_accepted_actions']:.2%}** of accepted actions; its pre-gate top-level intent accuracy was **{kiko['top_level']['accuracy_raw']:.2%}**. It produced **{kiko['oos_false_actions']}/{result['supports']['true_oos']}** actions on true out-of-scope clips. The saved Pi logits were not retained, so Kiko's ungated exact-leaf accuracy cannot be compared directly with the raw leaf figures for our models.",
        "",
        "| Model | Top-level intent accuracy (argmax) | Correct-action coverage | Accepted supported coverage | Accepted-action accuracy | OOS false actions | ONNX size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| ME2 active baseline (saved) | {base_top['accuracy']:.2%} | {base_gate['correct_action_coverage']:.2%} | {(base_gate['supported_support'] - base_gate['supported_rejects']) / base_gate['supported_support']:.2%} | {base_gate['accepted_supported_accuracy']:.2%} | {base_gate['noncommand_false_actions']}/47 | 39,315 B |",
        f"| ME2 dataset candidate (saved) | {cand_top['accuracy']:.2%} | {cand_gate['correct_action_coverage']:.2%} | {(cand_gate['supported_support'] - cand_gate['supported_rejects']) / cand_gate['supported_support']:.2%} | {cand_gate['accepted_supported_accuracy']:.2%} | {cand_gate['noncommand_false_actions']}/47 | 39,315 B |",
        f"| Kiko ONNX on Pi (run here) | {kiko['top_level']['accuracy_raw']:.2%} | {kiko['correct_action_coverage']:.2%} | {kiko['accepted_supported_coverage']:.2%} | {kiko['accuracy_among_accepted_actions']:.2%} | {kiko['oos_false_actions']}/{result['supports']['true_oos']} | 83,163 B |",
        "",
        "The shared 4,418-clip test was used before by the ME2 comparison and is reused here only as a diagnostic; it is not pristine or independently locked. The operating columns apply each system's stated gate (ME2 thresholds were validation-calibrated; Kiko uses its fixed 0.30 rule). Kiko's training provenance and possible test overlap are unknown. Its per-class F1 table below uses the 0.30 gate; ME2 per-class F1 is raw argmax. Those class F1 values are shown for diagnosis, not as an apples-to-apples ranking.",
        "",
        "## What is running on the Pi",
        "",
        f"- Kiko model file: `/home/dalmacio/Desktop/kiko/pi_deployment/vcm_int8.onnx`, SHA-256 `{kiko['model_sha256']}`, size 83,163 bytes.",
        "- ONNX CPU input `[1, 1, 64, 251]`, with intent logits `[1, 20]` and slot logits `[1, 19]`; Kiko labels contain 19 command groups plus `UNKNOWN`.",
        f"- Pi model-only p95: {kiko['inference_timing_pi_model_only']['p95_model_ms']:.3f} ms. Kiko's librosa frontend ran on Windows for this test because the configured Pi runtime venv is missing `librosa`; therefore this is not end-to-end Pi latency and should not be compared directly with our 5.225 ms frontend-plus-model figure.",
        "- Kiko was not running as a service; the only observed voice process was the separate ME2 dataset candidate on port 7865. The Kiko ONNX was invoked directly with audio features. No microphone/voice-response path was tested.",
        "- The Kiko controller loads a pretrained `hey_jarvis` detector through openWakeWord and calls `download_models()` at startup. That is not the same wake phrase as the custom Dandan wake model and it does not satisfy a from-scratch/offline wake-model comparison.",
        "- The Kiko README runtime requirements list only Flask and requests; its controller additionally imports audio, GPIO, pygame, librosa and openWakeWord libraries. The Pi Python environment used for this test has ONNX Runtime but not librosa.",
        "",
        "## Limits and next step",
        "",
        "Kiko's 64-band librosa features were generated on Windows because the configured Pi environment lacks librosa; only the ONNX inference was timed on the Pi. Kiko's training script, training revision, split manifest, checkpoint and feature-normalization details are absent from the installed folder. Before choosing a winner, agree one training revision, preserve an untouched common speaker-disjoint holdout, then run both exact released files and the same live wake/false-trigger SOP.",
        "",
        ("Raw Pi logits are stored in `pi_predictions.npz` beside this report. " if raw_results and raw_results.exists() else "Raw Pi logits were not retained; aggregate and per-class scores are stored in `metrics.json` and `per_class.csv` beside this report. ") + "Temporary test features and scripts were removed from the Pi after scoring; `~/Desktop/kiko` was not modified.",
    ]
    per_class_csv = RESULT_DIR / "per_class.csv"
    with per_class_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "support", "kiko_precision_at_030_gate", "kiko_recall_at_030_gate", "kiko_f1_at_030_gate", "me2_baseline_raw_argmax_f1", "me2_candidate_raw_argmax_f1"])
        for label in sorted(kiko["per_leaf"]):
            metrics = kiko["per_leaf"][label]
            ours_metrics = ours["per_class_delta"][label]
            writer.writerow([label, int(metrics["support"]), metrics["precision"], metrics["recall"], metrics["f1-score"], ours_metrics["baseline_f1"], ours_metrics["candidate_f1"]])
    rows += ["", "## Per-class F1 comparison", "", "See `per_class.csv` for all 31 labels, aligned by leaf intent/value label. Kiko F1 is measured with its fixed 0.30 gate; ME2 scores are raw argmax, so this table is descriptive only and is not a direct ranking.", ""]
    rows += ["| Label | Kiko F1 | ME2 baseline F1 | ME2 candidate F1 |", "|---|---:|---:|---:|"]
    for label in sorted(kiko["per_leaf"]):
        metrics = kiko["per_leaf"][label]
        ours_metrics = ours["per_class_delta"][label]
        rows.append(f"| `{label}` | {metrics['f1-score']:.2%} | {ours_metrics['baseline_f1']:.2%} | {ours_metrics['candidate_f1']:.2%} |")
    (RESULT_DIR / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if raw_results and raw_results.exists():
        shutil.copy2(raw_results, RESULT_DIR / "pi_predictions.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=PI_HOST)
    parser.add_argument("--allow-used-test", action="store_true", help="Explicitly acknowledge the previously scored test split")
    args = parser.parse_args()
    if not args.allow_used_test:
        raise SystemExit("This frozen test was already scored. Re-run only with --allow-used-test for the user's requested diagnostic comparison.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kiko-me2-pi-eval-") as work:
        work_dir = Path(work)
        payload = work_dir / "kiko_test_features.npz"
        predictions = work_dir / "kiko_pi_predictions.npz"
        remote_name = f"/tmp/kiko-vcm-eval-{int(time.time())}"
        remote_input = remote_name + "/test_features.npz"
        remote_runner = remote_name + "/run_kiko_onnx.py"
        remote_output = remote_name + "/predictions.npz"

        print("Preparing the previously used frozen test as an explicitly diagnostic comparison.", flush=True)
        prep = build_test_payload(payload)
        print(json.dumps(prep, indent=2), flush=True)
        (work_dir / "run_kiko_onnx.py").write_text(REMOTE_RUNNER, encoding="utf-8")

        hash_cmd = (
            f"test \"$(hostname)\" = cfdfnjrpi5 && sha256sum '{KIKO_MODEL}' "
            "'/home/dalmacio/Desktop/kiko/pi_deployment/labels.json' "
            "'/home/dalmacio/Desktop/kiko/pi_deployment/slots.json'"
        )
        host_check = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host, hash_cmd])
        if any(expected not in host_check.stdout for expected in (EXPECTED_KIKO_MODEL_SHA, EXPECTED_KIKO_LABELS_SHA, EXPECTED_KIKO_SLOTS_SHA)):
            raise RuntimeError(f"Kiko file hash mismatch; Pi reported: {host_check.stdout.strip()}")
        print("Pi and Kiko model hash verified; staging files under /tmp, outside Kiko folder.", flush=True)
        staged = False
        operation_failed = False
        try:
            run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host, f"mkdir -p -- '{remote_name}'"])
            staged = True
            run(["scp", "-q", str(payload), f"{args.host}:{remote_input}"])
            run(["scp", "-q", str(work_dir / "run_kiko_onnx.py"), f"{args.host}:{remote_runner}"])
            remote_cmd = f"{PI_PYTHON} '{remote_runner}' '{KIKO_MODEL}' '{remote_input}' '{remote_output}'"
            pi_run = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host, remote_cmd])
            print("Pi model results:", pi_run.stdout.strip(), flush=True)
            pi_meta = json.loads(pi_run.stdout.strip().splitlines()[-1])
            if pi_meta["model_sha256"] != EXPECTED_KIKO_MODEL_SHA:
                raise RuntimeError("The scored Kiko model hash differs from the inspected Pi model")
            run(["scp", "-q", f"{args.host}:{remote_output}", str(predictions)])
        except Exception:
            operation_failed = True
            raise
        finally:
            if staged:
                # Verify the canonical path before removing only this run's scratch directory.
                cleanup_cmd = (
                    f"p='{remote_name}'; r=$(realpath -- \"$p\"); "
                    f"if [ \"$r\" = \"$p\" ] && [ \"$p\" = /tmp/kiko-vcm-eval-{remote_name.rsplit('-', 1)[-1]} ]; "
                    "then rm -rf -- \"$p\"; else echo REFUSED_CLEANUP; exit 12; fi"
                )
                cleanup = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host, cleanup_cmd], check=False)
                if cleanup.returncode and not operation_failed:
                    raise RuntimeError(f"Pi temporary cleanup failed or refused; staging path: {remote_name}\n{cleanup.stdout}\n{cleanup.stderr}")
        pi_meta["model_bytes"] = 83163
        results = score_outputs(payload, predictions, pi_meta)
        write_reports(results, predictions)
        print(json.dumps({
            "report": str(RESULT_DIR / "report.md"),
            "metrics": str(RESULT_DIR / "metrics.json"),
            "raw_predictions": str(RESULT_DIR / "pi_predictions.npz"),
            "kiko": {k: v for k, v in results["kiko"].items() if k != "per_leaf"},
        }, indent=2))


if __name__ == "__main__":
    main()
