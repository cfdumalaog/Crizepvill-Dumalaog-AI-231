"""Stage the selected two-model ONNX candidate and audit metadata for Pi review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import onnxruntime as ort


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "personalized-vcm-20260928-155109"
DEST = PROJECT / "deployment" / "personalized_candidate"
WAKE_SRC = RUN / "wake_broad_negatives" / "models" / "wake_personalized_int8.onnx"
INTENT_SRC = RUN / "intent_personalized" / "models" / "intent_personalized_int8.onnx"
REPORT = RUN / "final_evaluation.json"
WAKE_THRESHOLD = 0.9992183446884155


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_signature(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "input_type": session.get_inputs()[0].type,
        "output_name": session.get_outputs()[0].name,
        "output_shape": session.get_outputs()[0].shape,
        "output_type": session.get_outputs()[0].type,
    }


def main() -> None:
    for path in (WAKE_SRC, INTENT_SRC, REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    wake = report["wake_model_comparison"]["broad"]
    intent = report["intent_model_comparison"]["personalized_int8"]
    DEST.mkdir(parents=True, exist_ok=True)
    models = DEST / "models"
    models.mkdir(exist_ok=True)
    wake_dest = models / "binary_wake_int8.onnx"
    intent_dest = models / "intent_int8.onnx"
    shutil.copy2(WAKE_SRC, wake_dest)
    shutil.copy2(INTENT_SRC, intent_dest)

    source_wake_summary = json.loads((WAKE_SRC.parent / "export_summary.json").read_text(encoding="utf-8"))
    wake_summary = {
        "classes": source_wake_summary["classes"],
        "validation_selected_threshold": WAKE_THRESHOLD,
        "validation_threshold_int8": WAKE_THRESHOLD,
        "int8_path": "models/binary_wake_int8.onnx",
        "int8_size_bytes": wake_dest.stat().st_size,
        "int8_sha256": sha256(wake_dest),
        "pretrained_weights_used": False,
        "threshold_selection": "zero false accepts on validation windows",
        "deployment_status": "experimental_candidate_not_deployed",
    }
    (DEST / "export_summary.json").write_text(json.dumps(wake_summary, indent=2) + "\n", encoding="utf-8")
    classes = json.loads((RUN / "intent_personalized" / "models" / "export_summary.json").read_text(encoding="utf-8"))["classes"]
    metadata = {
        "status": "experimental_candidate_not_deployed",
        "created_from_run": RUN.name,
        "training": {
            "pretrained_weights_used": False,
            "seed": 231,
            "intent_training": "Option B train split plus 3 exact-mapped person-01 examples per batch",
            "wake_training": "positive person-01 wake clips plus Option B command/background negatives",
            "speaker_count_for_personal_data": 1,
        },
        "frontend": {
            "sample_rate_hz": 16000,
            "mono": True,
            "window_seconds": 2.5,
            "feature": "40-band log-mel frontend used by tinyvcm_model.frontend.Frontend",
        },
        "wake": {
            "labels": ["NON_WAKE", "WAKE_WORD"],
            "wake_class_index": 1,
            "threshold": WAKE_THRESHOLD,
            "threshold_selection": "INT8 validation threshold with zero false accepts on validation windows",
            "threshold_source": "wake_broad_negatives/models/export_summary.json",
            "threshold_validation_constraint": "zero false accepts on validation windows",
            "heldout_personal_wake_hits": wake["heldout_personal_wake_hits"],
            "heldout_personal_wake_support": wake["heldout_personal_wake_support"],
            "heldout_command_false_accepts": wake["false_wake_on_optionb_command_clips"],
            "heldout_command_support": wake["optionb_command_support"],
            "model": {
                "path": "models/binary_wake_int8.onnx",
                "size_bytes": wake_dest.stat().st_size,
                "sha256": sha256(wake_dest),
                "signature": model_signature(wake_dest),
            },
        },
        "intent": {
            "labels": classes,
            "personal_heldout_accuracy": intent["personal_heldout_test"]["accuracy"],
            "personal_heldout_clips": intent["personal_heldout_test"]["count"],
            "optionb_test_accuracy": intent["optionb_test"]["accuracy"],
            "optionb_test_clips": intent["optionb_test"]["count"],
            "model": {
                "path": "models/intent_int8.onnx",
                "size_bytes": intent_dest.stat().st_size,
                "sha256": sha256(intent_dest),
                "signature": model_signature(intent_dest),
            },
        },
        "limitations": [
            "Only one real personal speaker is present; this is not a multi-speaker validation.",
            "Wake test support is five held-out clips from that same speaker.",
            "The PC physical microphone was silent during the controlled mic probe; live audio was tested through Stereo Mix loopback.",
            "Model-only latency was measured on PC CPU and does not establish Raspberry Pi latency or full-pipeline latency.",
            "The original Option B test set was used by earlier work; its score is a comparison, not a new untouched benchmark.",
            "This directory contains the model pair and metadata, not the complete Pi application image or installer.",
        ],
    }
    (DEST / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme = """# ME2 - VCM on Raspberry Pi 5 — personalized model candidate

This folder stages the selected, fresh-trained binary wake and 31-class intent
INT8 ONNX files with their threshold and integrity metadata. It is a **candidate
pair for integration and hardware audit**, not an installed Pi release.

The total ONNX size is approximately 75 KiB. Input is mono 16 kHz audio, padded
or cropped by the current frontend to 2.5 seconds and converted to 40-band
log-mel features. `metadata.json` records model signatures, SHA-256 checksums,
threshold provenance, measured candidate metrics, and limits. The wake runtime
expects `export_summary.json` beside this file so it can load the calibrated
threshold.

The wake threshold is 0.9992183447. It yielded 4/5 held-out personal wake clips
and zero false accepts across 1,798 held-out Option B command clips plus 42
held-out personal non-wake clips. These small, one-speaker sets do not establish
robustness or false wakes per hour. The personalized intent model classified
24/28 strict-mapped personal command clips correctly (85.7%); personal classes
and counts are listed in the full evaluation report.

No model has been copied to the powered-down Raspberry Pi. Before promotion,
retest with the actual microphone, gather data from at least three speakers and
room conditions, then measure end-to-end accuracy, false accepts, latency, and
memory on the Pi. See `../../docs/PERSONAL_WAKE_INTENT_EVALUATION_20260928.md`.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    print(f"Staged model pair: {wake_dest.stat().st_size + intent_dest.stat().st_size} bytes total")
    print(f"Metadata: {DEST / 'metadata.json'}")


if __name__ == "__main__":
    main()
