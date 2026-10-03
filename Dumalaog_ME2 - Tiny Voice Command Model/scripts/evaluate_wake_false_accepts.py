"""Measure wake-word false accepts on the speaker-disjoint Option B command test split."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vcm_app import VCMPredictor
from tinyvcm_model.config import OPTION_B_DATA, SR
from tinyvcm_model.frontend import fit_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic limit; default evaluates the full test split")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")

    with (OPTION_B_DATA / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    rows.sort(key=lambda row: row["path"])
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("No held-out test utterances were selected")

    predictor = VCMPredictor(ROOT / "models" / "vcm_wake32_int8.onnx")
    wake_count = 0
    vad_count = 0
    candidates_by_label = Counter()
    wake_scores = []
    for row in rows:
        path = OPTION_B_DATA / row["path"]
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SR:
            from math import gcd
            divisor = gcd(int(rate), SR)
            audio = resample_poly(audio, SR // divisor, int(rate) // divisor).astype(np.float32)
        audio = fit_audio(audio)
        result = predictor.predict(audio)
        score = result["wake_probability"]
        wake_scores.append(float(score))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms > 0.012:
            vad_count += 1
            if result["label"] == "WAKE_WORD":
                candidates_by_label[row["label"]] += 1
                if result["confidence"] >= args.threshold:
                    wake_count += 1

    summary = {
        "model": "vcm_wake32_int8.onnx",
        "test_split": "Option B speaker-disjoint command test utterances; no wake words",
        "utterances_evaluated": len(rows),
        "speech_level_vad_utterances": vad_count,
        "wake_threshold": args.threshold,
        "false_wake_candidates_at_threshold": wake_count,
        "false_wake_rate_over_vad_utterances": wake_count / vad_count if vad_count else None,
        "model_top1_wake_candidates_by_true_command": dict(candidates_by_label),
        "mean_wake_probability": float(np.mean(wake_scores)),
        "p95_wake_probability": float(np.percentile(wake_scores, 95)),
        "limitation": "One offline utterance prediction per command; live streaming consecutiveness and room noise are not represented.",
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
