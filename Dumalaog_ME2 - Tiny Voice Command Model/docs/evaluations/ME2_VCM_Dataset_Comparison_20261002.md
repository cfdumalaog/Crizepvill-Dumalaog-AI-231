# ME2 - VCM on Raspberry Pi 5: dataset-trained candidate comparison

**Date:** 2026-10-02 (Asia/Manila)  
**Candidate:** `classagreementvcm-20261002-a90b8d10-r3`  
**Decision:** The candidate is deployed for an isolated Pi trial, but it does not replace the current model.

## Why this dataset was previously not used

The first pass only audited the repulled files because the dataset combines several upstream sources, classmate recordings and generated speech with incomplete usage/consent evidence. The user then directed a private coursework training and comparison. I trained and scored a candidate from that request, kept the audio on this PC, and put only the ONNX models and runtime files on the user's Pi. Rights and consent evidence still need to be resolved before sharing the dataset or model beyond this private test.

## Training and evaluation

- Source: **ME2 Spoken Command Dataset**, pinned revision `a90b8d106349b02c5570a1a258503386043f63b2`.
- Intent model: TinyDSCNN-48, 14,527 parameters, randomly initialized; no pretrained weights, ASR, LLM or network service. Fit on the 31 supported command/value labels only. Selected epoch 82 of a 100-epoch run on the RTX 3050.
- Data partition: 8,201 supported fitting clips and 2,122 supported validation clips, split by source/speaker group from the train partition. The 47 true OOS test clips were not used for fitting or gate selection. The 158 unsupported-slot train rows were kept distinct from true OOS and used only for validation rejection calibration when applicable. The 66,390 numeral rows and 196-row holdout were untouched.
- The model, gates, hashes and scoring code were frozen before a single paired scoring pass on 4,418 published test clips (4,371 supported commands + 47 OOS). The test was not used for threshold selection.
- Wake: the separate binary model was retained byte-identically. This dataset has no wake-positive labels, so it cannot train wake detection. Its test clips provide only a negative/wake-false-accept check.
- INT8: 39,315 bytes. Paired with the 37,425-byte wake model, total ONNX size is 76,740 bytes (74.9 KiB). Validation FP32/INT8 prediction agreement is 95.63%.

## Paired results

All scores below are on the same 4,371 supported test clips. The coarse “top-level” score collapses the 31 command/value leaves to 19 intent names.

| Model | Accuracy | Macro F1 | Top-level intent accuracy | Top-level macro F1 |
|---|---:|---:|---:|---:|
| Current intent model | 79.64% | 79.47% | 81.38% | 76.74% |
| New dataset-trained intent model | 78.95% | 79.12% | 82.25% | 76.56% |
| Candidate change | -0.69 pp | -0.35 pp | +0.87 pp | -0.18 pp |

The current model is slightly better on aggregate command accuracy and macro F1. Independent source-speaker cluster bootstrap intervals overlap; they do not establish a statistically decisive overall winner. Candidate versus current accuracy by test source is:

| Test source | n | Current | Candidate |
|---|---:|---:|---:|
| FluentSpeechCommands | 147 | 22.45% | 62.59% |
| SLURP | 553 | 16.82% | 26.04% |
| SNIPS | 101 | 16.83% | 32.67% |
| Other real voice | 179 | 12.85% | 24.02% |
| Group synthetic | 3,368 | 98.22% | 92.84% |
| Xela Multi-Sensor | 11 | 18.18% | 36.36% |
| Xela temperature | 10 | 30.00% | 60.00% |
| Timers and Such | 2 | 100.00% | 100.00% |

The candidate is better on the small real/varied-source slices and worse on the large synthetic slice. Those slices are not balanced, and the test overall is dominated by the synthetic group. This makes the candidate useful for a live personal comparison, not an overall replacement decision.

## Intent action gates

The gates are calibrated accept/reject rules; their probability values are not accuracy or F1 scores.

| Model/gate | Confidence | Top-two margin | Correct-action coverage | Accepted command accuracy | OOS false actions |
|---|---:|---:|---:|---:|---:|
| Current deployed gate | 0.68 | 0.15 | 77.30% | 90.59% | 22/47 |
| Current model, train-derived validation gate | 0.98 | 0.00 | 65.41% | 98.59% | 2/47 |
| Candidate, train-derived validation gate | 0.76 | 0.00 | 61.36% | 98.82% | 0/47 |

The validation policy allowed at most 1% false actions on reject examples. With only 27 such validation examples, the selected gates made zero false actions. The candidate gate rejects more supported speech than the deployed current gate. The comparison shows the safety/coverage trade-off; it does not mean “76% confidence equals 98.82% accuracy.”

## Per-class F1

Each test leaf has 141 examples. F1 values are macro-relevant classwise F1, not confidence thresholds.

| Label | Current F1 | Candidate F1 | Δ pp |
|---|---:|---:|---:|
| `ALARM_6_00AM` | 86.02% | 83.15% | -2.87 |
| `ALARM_8_00AM` | 72.58% | 87.23% | +14.65 |
| `ALARM_9_00PM` | 76.38% | 92.14% | +15.76 |
| `BRIGHTNESS_100` | 88.59% | 89.89% | +1.30 |
| `BRIGHTNESS_20` | 92.42% | 90.51% | -1.91 |
| `BRIGHTNESS_60` | 96.35% | 96.35% | +0.00 |
| `CALL` | 67.55% | 72.06% | +4.51 |
| `COLOR_BLUE` | 79.66% | 66.41% | -13.25 |
| `COLOR_GREEN` | 80.15% | 78.95% | -1.20 |
| `COLOR_RED` | 84.50% | 71.48% | -13.02 |
| `CREATE_REMINDER_DRINK_WATER` | 82.57% | 93.91% | +11.34 |
| `CREATE_REMINDER_EXERCISE` | 84.54% | 92.47% | +7.93 |
| `CREATE_REMINDER_STUDY` | 96.06% | 95.17% | -0.89 |
| `LIGHT_OFF` | 62.45% | 49.80% | -12.65 |
| `LIGHT_ON` | 67.13% | 63.87% | -3.26 |
| `LIST_REMINDERS` | 79.85% | 80.60% | +0.74 |
| `MESSAGE` | 87.77% | 83.69% | -4.08 |
| `NEXT` | 80.47% | 72.36% | -8.11 |
| `PAUSE` | 75.94% | 80.67% | +4.73 |
| `PLAY_MUSIC` | 55.41% | 48.43% | -6.98 |
| `STOP` | 62.35% | 67.31% | +4.96 |
| `TEMPERATURE_18` | 80.00% | 91.03% | +11.03 |
| `TEMPERATURE_22` | 93.43% | 87.20% | -6.23 |
| `TEMPERATURE_26` | 94.89% | 92.96% | -1.93 |
| `TIME` | 70.97% | 69.15% | -1.82 |
| `TIMER_10s` | 88.44% | 94.29% | +5.85 |
| `TIMER_1m` | 89.49% | 89.20% | -0.29 |
| `TIMER_30s` | 87.92% | 93.23% | +5.31 |
| `VOLUME_DOWN` | 71.43% | 59.52% | -11.90 |
| `VOLUME_UP` | 61.11% | 54.62% | -6.49 |
| `WEATHER` | 67.27% | 65.15% | -2.11 |

Only 2 of 31 candidate classes reach 95% F1. The project's per-class 95% target is not met.

## Pi trial and next evaluation

- Separate install: `/home/dalmacio/Desktop/me2-dataset-candidate`; current `/home/dalmacio/Desktop/dandan` hashes are preserved.
- Candidate app is manually running at `127.0.0.1:7865`, with no boot autostart, GPIO disabled, wake gate 0.95, and intent gates 0.76/0.00. PC SSH tunnel: `http://127.0.0.1:7865/assistant` and `/studio`.
- Pi verifier passed 24 file hashes, ARM64 ONNX CPU execution and both class counts. Pi frontend-plus-model p95 was 5.510 ms wake and 5.225 ms intent.
- `/api/state` reported the exact candidate intent hash, unchanged wake hash and correct thresholds. The UI endpoint returned HTTP 200.
- The Pi reports **no microphone connected**. Thus Pi install, UI and model inference are verified; live wake/command speech on Pi is not.
- Wake test on all 4,418 gold test clips (all negative for wake) produced 1 false accept at 0.95 (0.02%). There are no positive wake clips in this test, so it gives no wake recall estimate.

**Recommended next step:** connect the Pi microphone, then compare the two model versions with the same speakers, mic placement, phrases, background conditions and randomized order. Track wake hits/false wakes per hour and accepted/rejected intent results. Keep the current release as default unless the personal Pi trial shows a clear useful gain without hurting lights-off/color or OOS rejection. Do not tune on this already-used frozen test.

### Evidence files

- Frozen metrics and per-class/source tables: `runs/classagreementvcm-20261002-a90b8d10-r3/frozen_test_comparison.json` and `COMPARISON_REPORT.md`.
- Training and quantization: `training_history.json`, `training_summary.json`, `candidate_metadata.json`, `quantization_validation.json` in the same run.
- Isolated release receipt: `deployment/classagreementvcm_candidate_deployment.json`.
- Full training walk-through: `notebooks/ME2_Tiny_VCM_Training.ipynb`.
