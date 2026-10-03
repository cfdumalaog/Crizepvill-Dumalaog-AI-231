"""Retrain the intent model from scratch while retaining the active binary wake model.

The current deployed pair is backed up before this script is run. The script
freezes the current human manifest, trains the 31-class intent model on the
dataset train split plus mapped personal training clips, opens the reused
dataset test only after export, and writes a standard completed-pair run that
can be staged and deployed by deploy_latest_vcm.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tinyvcm_model.personalized_retrain import (  # noqa: E402
    _export_intent_candidate,
    evaluate_all,
    prepare_experiment,
    train_intent_candidate_from_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"me2-vcm-intent-refresh-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--wake-source", default="me2-vcm-20261001-wake-refresh")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    run_dir = PROJECT / "runs" / args.run_name
    source = PROJECT / "runs" / args.wake_source
    wake_dir = source / "wake_binary"
    for required in (wake_dir / "models/wake_personalized_int8.onnx", wake_dir / "models/wake_personalized_fp32.onnx", wake_dir / "models/export_summary.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    data = prepare_experiment(run_dir, PROJECT)
    shutil.copytree(wake_dir, run_dir / "wake_binary")
    wake_int8 = run_dir / "wake_binary/models/wake_personalized_int8.onnx"
    active = json.loads((PROJECT / "deployment/current_vcm/metadata.json").read_text(encoding="utf-8"))
    active_wake_hash = active["wake"]["model"]["sha256"]
    digest = hashlib.sha256(wake_int8.read_bytes()).hexdigest()
    if digest != active_wake_hash:
        raise ValueError(f"Retained wake model does not match current deployment: {digest} != {active_wake_hash}")

    provenance = {
        "wake": {"training": "retained_byte_identical", "source_run": args.wake_source,
                 "int8_sha256": digest, "active_threshold": active["wake"]["local_operating_threshold"]},
        "intent": {"training": "from_random_initialization", "source_run": args.run_name,
                   "recipe": "refined", "seed": args.seed, "epochs": args.epochs,
                   "dataset": "ME2 Spoken Command Dataset train split + mapped personal train clips"},
    }
    (run_dir / "pair_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    intent = train_intent_candidate_from_data(data, epochs=args.epochs, seed=args.seed, recipe="refined")
    _export_intent_candidate(intent, data, seed=args.seed)

    wake_export = json.loads((run_dir / "wake_binary/models/export_summary.json").read_text(encoding="utf-8"))
    wake = {
        "fp32_path": run_dir / "wake_binary/models/wake_personalized_fp32.onnx",
        "int8_path": wake_int8,
        "summary": wake_export,
        "val_threshold": float(wake_export["validation_threshold_int8"]),
    }
    report = evaluate_all(data, {"binary": wake}, intent)
    report["pair_provenance"] = provenance
    (run_dir / "final_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run": str(run_dir),
        "human_manifest_rows": data["human_audit"]["raw_manifest_rows"],
        "unique_human_audio": data["human_audit"]["unique_pcm_groups"],
        "manifest_sha256": data["human_audit"]["manifest_sha256"],
        "wake_model_retained_sha256": digest,
        "intent_model_sha256": hashlib.sha256((run_dir / "intent_personalized/models/intent_personalized_int8.onnx").read_bytes()).hexdigest(),
        "optionb_test": report["intent_model_comparison"]["personalized_int8"]["optionb_test"],
        "personal_heldout": report["intent_model_comparison"]["personalized_int8"]["personal_heldout_test"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
