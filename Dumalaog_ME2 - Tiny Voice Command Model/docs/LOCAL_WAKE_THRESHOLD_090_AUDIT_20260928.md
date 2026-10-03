# ME2 - VCM on Raspberry Pi 5 — Local Wake Threshold Replay

**Date:** 2026-09-28 (Asia/Manila)
**Scope:** Offline replay of saved recordings through the desktop candidate's binary-wake INT8 ONNX model. No audio was recorded, no Pi was accessed, and the model weights were not changed.

## Model and method

- Wake model: `deployment/personalized_candidate_retrain-20260928-190911/models/binary_wake_int8.onnx`
- SHA-256: `f7d1c3f7826bc16426eac02c10c05e077502f28bbc764c81510c7af08fbfb002`
- Input path: the same `VCMPredictor` and 16 kHz / 2.5-second / 40-band frontend used by the desktop VCM.
- For each held-out recording, five evenly shifted 2.5-second contexts were inferred; the maximum wake probability represents whether the rolling detector would cross a threshold at any tested placement.
- Personal test set: 65 deduplicated held-out clips from the recorded-data split manifest, including 8 `wake_word` and 57 non-wake clips. The personal split is clip-held-out, not speaker-disjoint.
- General command check: 1,798 Option B test command clips, each evaluated at five temporal placements. This set has been used by earlier project experiments.

## Results

| Operating threshold | Personal wake accepted | Personal non-wake accepted | Option B command clips falsely accepted |
|---:|---:|---:|---:|
| 0.900000 | 8/8 | 1/57 | 0/1,798 |
| 0.998543 (validation-selected) | 7/8 | 0/57 | 0/1,798 |

The personal false accept at 0.90 was a `volume_down` recording with maximum wake probability **0.989236**. The one missed wake at the validation-selected threshold scored **0.997595**.

## Local operating configuration

The user's requested **0.90** threshold is now the default for `scripts/Start-Local-VCM.ps1` and is passed explicitly to `vcm_app.py`. The candidate's `export_summary.json` and `metadata.json` are unchanged: they preserve the training/validation-selected value of 0.9985431433. To restore the model-selected operating point, start with `-WakeThreshold 0.9985431433`.

This threshold improves recall on the eight recorded wake test clips, at the cost of accepting one recorded non-wake command in this small personal set. These finite utterance checks do not measure false wakes per hour in continuous room audio or guarantee speaker-general performance. The desktop candidate remains experimental and has not been deployed to the Raspberry Pi.
