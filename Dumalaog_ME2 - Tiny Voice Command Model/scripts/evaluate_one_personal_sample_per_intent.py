"""One deterministic random, held-out personal recording per intent smoke test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.config import INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD, LABELS  # noqa: E402
from tinyvcm_model.personalized_retrain import load_personal_records  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    run = PROJECT / "runs" / args.run
    snapshot = run / "human_manifest_snapshot.csv"
    model_path = run / "intent_personalized/models/intent_personalized_int8.onnx"
    items, _ = load_personal_records(PROJECT, manifest_path=snapshot)
    test_items = [x for x in items if x["split"] == "test" and x["intent"] in LABELS]
    rng = np.random.default_rng(args.seed)
    chosen = []
    for label in LABELS:
        choices = [x for x in test_items if x["intent"] == label]
        if not choices:
            raise RuntimeError(f"No personal held-out clip for {label}")
        chosen.append(choices[int(rng.integers(0, len(choices)))])

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    features = np.stack([x["feature"] for x in chosen]).astype(np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: features})[0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    order = np.argsort(probs, axis=1)
    records = []
    for i, item in enumerate(chosen):
        top = int(order[i, -1])
        confidence = float(probs[i, top])
        margin = confidence - float(probs[i, order[i, -2]])
        accepted = confidence >= INTENT_CONFIDENCE_THRESHOLD and margin >= INTENT_MARGIN_THRESHOLD
        predicted = LABELS[top] if accepted else "REJECT"
        records.append({
            "expected_intent": item["intent"], "model_output": predicted,
            "raw_top_intent": LABELS[top], "correct": predicted == item["intent"],
            "confidence": confidence, "margin": margin,
            "accepted_by_live_gate": accepted,
            "confidence_threshold": INTENT_CONFIDENCE_THRESHOLD,
            "margin_threshold": INTENT_MARGIN_THRESHOLD,
            "recording": item["path"], "speaker": item["speaker"], "condition": item["condition"],
            "source_id": item.get("source_id"),
        })
    report = {
        "run": args.run, "sampling": "one uniform random recording per intent from the fixed personal held-out clip split",
        "seed": args.seed, "count": len(records), "correct_after_live_gate": sum(r["correct"] for r in records),
        "accuracy_after_live_gate": sum(r["correct"] for r in records) / len(records),
        "raw_top1_accuracy": sum(r["raw_top_intent"] == r["expected_intent"] for r in records) / len(records),
        "gate": {"confidence": INTENT_CONFIDENCE_THRESHOLD, "margin": INTENT_MARGIN_THRESHOLD},
        "limitation": "Small 31-clip smoke check, one clip per class; clip-heldout clips may share speakers with training and do not replace the official Gold benchmark.",
        "results": records,
    }
    out = run / "one_random_personal_clip_per_intent.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (run / "one_random_personal_clip_per_intent.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: report[k] for k in ("run", "count", "correct_after_live_gate", "accuracy_after_live_gate", "raw_top1_accuracy", "gate", "limitation")}, indent=2))
    for result in records:
        print(f"{result['expected_intent']:<32} -> {result['model_output']:<32} conf={result['confidence']:.3f} margin={result['margin']:.3f} {'OK' if result['correct'] else 'MISS'}")


if __name__ == "__main__":
    main()
