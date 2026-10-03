# Paired frozen-model evaluation on the latest Gold test split

Dataset: ME2 Spoken Command Dataset, commit `5abbe539a46b9b26ff24e73d2860d1698d44e81f`; test rows: 4443; supported rows: 4367; out-of-scope rows: 76.
Test parquet SHA-256: `40e9b66b4e12bc5b012bfa2dc7367602740ac394467e48d4924364c46f912f32`.

Both INT8 models were run on the exact same test WAVs with the same 16 kHz, 2.5-second, 40-bin log-mel frontend. Intent metrics are raw top-class argmax on the 31 supported classes. The held-out Gold test did not select thresholds.

| Frozen model | Raw accuracy | Macro F1 | Fixed operating gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |
|---|---:|---:|---|---:|---:|---:|---:|
| Previously active ME2 intent model | 84.29% | 84.25% | confidence 0.68, margin 0.15 | 3584 | 283 | 500 | 36/76 |
| Retrained ME2 intent model | 84.57% | 84.43% | confidence 0.68, margin 0.15 | 3597 | 247 | 523 | 29/76 |

## Common-gate check

For a matched operating-threshold view, the same active confidence/margin gate (0.68/0.15) is also applied to both frozen models. This is the active model's existing validation-selected operational gate; it was not tuned on the Gold test.

| Model | Gate | Correct accepted | Wrong accepted | Rejected known | OOS false actions |
|---|---:|---:|---:|---:|---:|
| Previously active ME2 intent model | 0.68 / 0.15 | 3584 | 283 | 500 | 36/76 |
| Retrained ME2 intent model | 0.68 / 0.15 | 3597 | 247 | 523 | 29/76 |

## Provenance and caveats

- This run compares `Previously active ME2 intent model` with `Retrained ME2 intent model`. Both were frozen before scoring this Gold test revision; the test rows were not used for training or threshold selection.
- The retrained candidate used the ME2 Spoken Command Dataset training split plus the mapped personal training clips. Its per-class scores are in `per_class.csv`.
- This dataset has no wake positives, so only intent was evaluated; the binary wake model is not compared here.
- This is offline scoring of the official test split. The class Raspberry Pi live benchmark uses the separate holdout split, records three wake samples, and measures the microphone/speaker path and device behavior; it remains to be run when the Pi microphone is available.
- Full per-class precision/recall/F1 is `per_class.csv`; exact model hashes and detailed counts are in `summary.json`.

The Gold test is benchmark-only. All model settings and the shared intent gate (0.68 confidence / 0.15 margin) were frozen before this scoring pass. The class live holdout SOP remains a separate Pi evaluation.
