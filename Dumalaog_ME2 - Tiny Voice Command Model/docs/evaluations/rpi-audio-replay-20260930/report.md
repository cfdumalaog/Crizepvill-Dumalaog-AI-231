# Raspberry Pi command and wake replay evaluation — 2026-09-30

## Result

Both ONNX models ran on the Raspberry Pi 5 CPU with zero failed audio rows. The host was `cfdfnjrpi5` (ARM64, Linux; ONNX Runtime 1.30.0, CPUExecutionProvider). PC and Pi reported identical aggregate and per-class metrics and identical top confusions in every tested group. The tested model hashes match the current deployment pair.

| Audio set / mode | Clips | Accuracy | Macro F1 | Extra wake result | Pi p50 / p95 (ms) |
|---|---:|---:|---:|---|---:|
| Personal held-out / intent | 101 | 92.1% | 93.6% | — | 5.49 / 5.58 |
| Personal held-out / wake | 128 | 100.0% | 100.0% | 8/8 wakes; 0/120 false accepts | 27.75 / 27.86 |
| Legacy synthetic / intent | 2268 | 65.8% | 66.0% | — | 5.53 / 5.59 |
| Legacy synthetic / wake | 4383 | 92.8% | 48.1% | 0/315 wakes; 0/4068 false accepts | 27.76 / 27.90 |
| ME2 Spoken Command Dataset / intent | 1798 | 97.2% | 97.2% | — | 5.54 / 5.61 |
| ME2 Spoken Command Dataset / wake negatives | 1798 | 100.0% (negative-only) | — | 0 positive wakes; 0/1,798 false accepts | 27.81 / 27.96 |

Wake threshold was fixed at **0.95**. Personal held-out replay had 8 positive wake clips and 120 negatives; the 1,798 reference wake rows are all negatives. The legacy synthetic set has 315 positive wake clips and 4,068 negatives; none of its 315 positives crossed the 0.95 threshold. Its overall binary accuracy of 92.81% is therefore misleading because it comes from rejecting every clip in an imbalanced set.

## Per-command scores

The columns show `support / recall / F1`. Personal data is the frozen held-out subset from the user-recorded corpus; support is small (one to nine clips per command). The legacy synthetic folder taxonomy exactly maps to only 12 of the current 31 intent labels; unmapped categories were excluded rather than guessed. Every current command has coverage in the personal holdout and the full reference test.

| Intent | Personal | Legacy synthetic | ME2 Spoken Command Dataset test |
|---|---:|---:|---:|
| `ALARM_6_00AM` | 2 / 100.0% / 100.0% | 0 / — / — | 58 / 98.3% / 99.1% |
| `ALARM_8_00AM` | 2 / 100.0% / 80.0% | 0 / — / — | 56 / 100.0% / 99.1% |
| `ALARM_9_00PM` | 3 / 100.0% / 100.0% | 0 / — / — | 58 / 100.0% / 100.0% |
| `BRIGHTNESS_100` | 4 / 100.0% / 100.0% | 0 / — / — | 60 / 100.0% / 96.8% |
| `BRIGHTNESS_20` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 98.3% / 97.5% |
| `BRIGHTNESS_60` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 96.7% / 98.3% |
| `CALL` | 2 / 50.0% / 66.7% | 162 / 32.1% / 38.8% | 54 / 96.3% / 97.2% |
| `COLOR_BLUE` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 91.7% / 94.0% |
| `COLOR_GREEN` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 96.7% / 98.3% |
| `COLOR_RED` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 100.0% / 94.5% |
| `CREATE_REMINDER_DRINK_WATER` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 100.0% / 97.6% |
| `CREATE_REMINDER_EXERCISE` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 98.3% / 99.2% |
| `CREATE_REMINDER_STUDY` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 98.3% / 99.2% |
| `LIGHT_OFF` | 5 / 80.0% / 80.0% | 216 / 52.8% / 54.7% | 60 / 96.7% / 94.3% |
| `LIGHT_ON` | 4 / 100.0% / 100.0% | 216 / 69.0% / 64.6% | 54 / 92.6% / 95.2% |
| `LIST_REMINDERS` | 2 / 100.0% / 100.0% | 216 / 77.8% / 86.6% | 58 / 93.1% / 94.7% |
| `MESSAGE` | 3 / 100.0% / 100.0% | 0 / — / — | 58 / 94.8% / 97.3% |
| `NEXT` | 2 / 50.0% / 66.7% | 162 / 52.5% / 68.8% | 60 / 98.3% / 99.2% |
| `PAUSE` | 9 / 44.4% / 61.5% | 162 / 29.0% / 38.8% | 56 / 98.2% / 98.2% |
| `PLAY_MUSIC` | 4 / 100.0% / 61.5% | 216 / 89.8% / 82.6% | 58 / 100.0% / 98.3% |
| `STOP` | 7 / 100.0% / 100.0% | 0 / — / — | 54 / 96.3% / 94.5% |
| `TEMPERATURE_18` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 93.3% / 95.7% |
| `TEMPERATURE_22` | 4 / 100.0% / 100.0% | 0 / — / — | 60 / 100.0% / 96.0% |
| `TEMPERATURE_26` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 96.7% / 98.3% |
| `TIME` | 3 / 100.0% / 85.7% | 216 / 91.7% / 79.0% | 54 / 90.7% / 95.1% |
| `TIMER_10s` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 95.0% / 95.0% |
| `TIMER_1m` | 3 / 100.0% / 100.0% | 162 / 85.8% / 90.0% | 56 / 100.0% / 99.1% |
| `TIMER_30s` | 3 / 100.0% / 100.0% | 0 / — / — | 60 / 95.0% / 96.6% |
| `VOLUME_DOWN` | 3 / 100.0% / 100.0% | 162 / 34.0% / 43.8% | 52 / 100.0% / 98.1% |
| `VOLUME_UP` | 1 / 100.0% / 100.0% | 162 / 63.6% / 60.4% | 60 / 98.3% / 99.2% |
| `WEATHER` | 2 / 100.0% / 100.0% | 216 / 87.0% / 84.1% | 52 / 100.0% / 98.1% |

## Runtime and protocol notes

- Pi p95 intent latency is **5.61 ms** on the 31-class reference test, including the project frontend and ONNX inference. This is below the 10 ms target for intent classification.
- Pi p95 wake latency is **27.96 ms** for the evaluator’s five temporal placements per saved clip. This is five frontend/model passes per WAV, not a measured single streaming pass; it should not be presented as single-window model latency.
- Saved files were replayed one at a time at 16 kHz mono through the same frontend and deployed INT8 ONNX files. Wake scoring used the maximum of five temporal placements at the fixed 0.95 threshold.
- This was inference-only: no GPIO, calls, weather, volume, lights, or other action handlers were invoked. The Pi has no usable microphone listed in the latest handoff, so this does not verify a live microphone session.
- The reference test is the existing speaker-disjoint partition and has been reused during model selection. Personal split is clip-disjoint but the same speakers occur across train and test. Small personal support makes its per-class values noisy.

## Model identity

- Intent: `39,315` bytes, SHA-256 `85696f8ebbf17c44b8641785fcdb7d2e61c68042271a818d943de58d66caa0c9`.
- Wake: `37,425` bytes, SHA-256 `44bf989b7f2cbd3823c2cbb975b0fbf26e4ca6c489075430433bec19db2d569e`.
- Machine-readable reports: `local_results.json` and `pi_results.json` in this directory.
- Evaluation utility: `scripts/evaluate_command_audio.py` in the project. The staged evaluation bundle was temporary and removed from the Pi after the report was copied back.
