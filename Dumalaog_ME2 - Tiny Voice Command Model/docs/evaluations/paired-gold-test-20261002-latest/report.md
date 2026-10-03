# Paired frozen-model evaluation on the latest Gold test split

Dataset: ME2 Spoken Command Dataset, commit `6947f13073e57eb6ae67e7e2fc3680700b82aa13`; test rows: 4443; supported rows: 4367; out-of-scope rows: 76.
Test parquet SHA-256: `40e9b66b4e12bc5b012bfa2dc7367602740ac394467e48d4924364c46f912f32`.

Both INT8 models were run on the exact same test WAVs with the same 16 kHz, 2.5-second, 40-bin log-mel frontend. Intent metrics are raw top-class argmax on the 31 supported classes. The held-out Gold test did not select thresholds.

| Frozen model | Raw accuracy | Macro F1 | Fixed operating gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |
|---|---:|---:|---|---:|---:|---:|---:|
| Gold-train candidate | 82.37% | 82.55% | confidence 0.76, margin 0.00 | 2817 | 30 | 1520 | 1/76 |
| Active Option-B + personal model | 84.29% | 84.25% | confidence 0.68, margin 0.15 | 3584 | 283 | 500 | 36/76 |

## Common-gate check

For a matched operating-threshold view, the same active confidence/margin gate (0.68/0.15) is also applied to both frozen models. This is the active model's existing validation-selected operational gate; it was not tuned on the Gold test.

| Model | Gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |
|---|---:|---:|---:|---:|---:|
| Gold-train candidate | 0.68 / 0.15 | 3016 | 62 | 1289 | 1/76 |
| Active Option-B + personal model | 0.68 / 0.15 | 3584 | 283 | 500 | 36/76 |

## Provenance and caveats

- The Gold-train candidate was trained from scratch using the older pinned Gold training split at `a90b8d106349b02c5570a1a258503386043f63b2`. It is frozen here and tested against the newer `6947f13073e57eb6ae67e7e2fc3680700b82aa13` test split; this is a cross-revision evaluation, not a retraining result.
- The active Option-B-derived intent model was trained on Option B plus personal recordings. It is not a pure Option-B-only model. No retained pure Option-B-only ONNX was found in active or archived model artifacts, so this paired comparison cannot isolate the effect of Option B alone.
- This dataset has no wake positives, so only intent was evaluated; the binary wake model is not compared here.
- The refreshed test is materially different from the prior published test, not merely 25 appended rows: 4,192 of 4,418 old test audio waveforms are present byte-identically in the new test; 226 old waveforms were removed and 251 new waveforms added. The new test has 76 OOS vs 47 previously. Therefore compare the two models within one revision; do not interpret its higher scores as model improvement over the old test.
- Full per-class precision/recall/F1 is `per_class.csv`; exact model hashes and detailed counts are in `summary.json`.

The Gold test is benchmark-only: do not train or tune on its rows. Train a Gold-baseline model on Gold train, select settings on a speaker/source-disjoint validation carved from train, then freeze it before this common test and live holdout SOP.
