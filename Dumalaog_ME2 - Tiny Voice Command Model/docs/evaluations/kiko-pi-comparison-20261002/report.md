# ME2 - VCM on Raspberry Pi 5: Kiko model comparison

**Date:** 2026-10-02 (Asia/Manila)  
**Test:** Installed Kiko ONNX model executed on the Raspberry Pi 5 CPU; same 4,418 published test clips and local pinned revision `a90b8d106349b02c5570a1a258503386043f63b2`.  
**Scope:** Inference-only diagnostic. No training, GPIO, dashboard action, microphone, file changes inside `~/Desktop/kiko`, or deployment was performed.

## Main result

At Kiko's fixed 30% intent-confidence gate, it made the correct exact command/value action on **73.14%** of supported clips. It accepted **95.06%** of supported clips and was correct on **76.94%** of accepted actions; its pre-gate top-level intent accuracy was **78.06%**. It produced **43/47** actions on true out-of-scope clips. The saved Pi logits were not retained, so Kiko's ungated exact-leaf accuracy cannot be compared directly with the raw leaf figures for our models.

| Model | Top-level intent accuracy (argmax) | Correct-action coverage | Accepted supported coverage | Accepted-action accuracy | OOS false actions | ONNX size |
|---|---:|---:|---:|---:|---:|---:|---:|
| ME2 active baseline (saved) | 81.38% | 65.41% | 66.35% | 98.59% | 2/47 | 39,315 B |
| ME2 dataset candidate (saved) | 82.25% | 61.36% | 62.09% | 98.82% | 0/47 | 39,315 B |
| Kiko ONNX on Pi (run here) | 78.06% | 73.14% | 95.06% | 76.94% | 43/47 | 83,163 B |

The shared 4,418-clip test was used before by the ME2 comparison and is reused here only as a diagnostic; it is not pristine or independently locked. The operating columns apply each system's stated gate (ME2 thresholds were validation-calibrated; Kiko uses its fixed 0.30 rule). Kiko's training provenance and possible test overlap are unknown. Its per-class F1 table below uses the 0.30 gate; ME2 per-class F1 is raw argmax. Those class F1 values are shown for diagnosis, not as an apples-to-apples ranking.

## What is running on the Pi

- Kiko model file: `/home/dalmacio/Desktop/kiko/pi_deployment/vcm_int8.onnx`, SHA-256 `9924f24dc376b385f9a2754d2559aeb9850581b012bbf260d427b8b887b075b5`, size 83,163 bytes.
- ONNX CPU input `[1, 1, 64, 251]`, with intent logits `[1, 20]` and slot logits `[1, 19]`; Kiko labels contain 19 command groups plus `UNKNOWN`.
- Pi model-only p95: 1.858 ms. Kiko's librosa frontend ran on Windows for this test because the configured Pi runtime venv is missing `librosa`; therefore this is not end-to-end Pi latency and should not be compared directly with our 5.225 ms frontend-plus-model figure.
- Kiko was not running as a service; the only observed voice process was the separate ME2 dataset candidate on port 7865. The Kiko ONNX was invoked directly with audio features. No microphone/voice-response path was tested.
- The Kiko controller loads a pretrained `hey_jarvis` detector through openWakeWord and calls `download_models()` at startup. That is not the same wake phrase as the custom Dandan wake model and it does not satisfy a from-scratch/offline wake-model comparison.
- The Kiko README runtime requirements list only Flask and requests; its controller additionally imports audio, GPIO, pygame, librosa and openWakeWord libraries. The Pi Python environment used for this test has ONNX Runtime but not librosa.

## Limits and next step

Kiko's 64-band librosa features were generated on Windows because the configured Pi environment lacks librosa; only the ONNX inference was timed on the Pi. Kiko's training script, training revision, split manifest, checkpoint and feature-normalization details are absent from the installed folder. Before choosing a winner, agree one training revision, preserve an untouched common speaker-disjoint holdout, then run both exact released files and the same live wake/false-trigger SOP.

Raw Pi logits were not retained; aggregate and per-class scores are stored in `metrics.json` and `per_class.csv` beside this report. Temporary test features and scripts were removed from the Pi after scoring; `~/Desktop/kiko` was not modified.

## Per-class F1 comparison

See `per_class.csv` for all 31 labels, aligned by leaf intent/value label. Kiko F1 is measured with its fixed 0.30 gate; ME2 scores are raw argmax, so this table is descriptive only and is not a direct ranking.

| Label | Kiko F1 | ME2 baseline F1 | ME2 candidate F1 |
|---|---:|---:|---:|
| `ALARM_6_00AM` | 88.12% | 86.02% | 83.15% |
| `ALARM_8_00AM` | 93.43% | 72.58% | 87.23% |
| `ALARM_9_00PM` | 92.31% | 76.38% | 92.14% |
| `BRIGHTNESS_100` | 79.09% | 88.59% | 89.89% |
| `BRIGHTNESS_20` | 73.23% | 92.42% | 90.51% |
| `BRIGHTNESS_60` | 86.72% | 96.35% | 96.35% |
| `CALL` | 64.15% | 67.55% | 72.06% |
| `COLOR_BLUE` | 71.60% | 79.66% | 66.41% |
| `COLOR_GREEN` | 76.35% | 80.15% | 78.95% |
| `COLOR_RED` | 68.81% | 84.50% | 71.48% |
| `CREATE_REMINDER_DRINK_WATER` | 93.29% | 82.57% | 93.91% |
| `CREATE_REMINDER_EXERCISE` | 96.06% | 84.54% | 92.47% |
| `CREATE_REMINDER_STUDY` | 95.31% | 96.06% | 95.17% |
| `LIGHT_OFF` | 53.91% | 62.45% | 49.80% |
| `LIGHT_ON` | 62.26% | 67.13% | 63.87% |
| `LIST_REMINDERS` | 82.73% | 79.85% | 80.60% |
| `MESSAGE` | 89.63% | 87.77% | 83.69% |
| `NEXT` | 69.89% | 80.47% | 72.35% |
| `PAUSE` | 59.79% | 75.94% | 80.67% |
| `PLAY_MUSIC` | 51.96% | 55.41% | 48.43% |
| `STOP` | 53.90% | 62.35% | 67.31% |
| `TEMPERATURE_18` | 89.61% | 80.00% | 91.03% |
| `TEMPERATURE_22` | 86.52% | 93.43% | 87.20% |
| `TEMPERATURE_26` | 89.81% | 94.89% | 92.96% |
| `TIME` | 67.77% | 70.97% | 69.15% |
| `TIMER_10s` | 89.20% | 88.44% | 94.29% |
| `TIMER_1m` | 93.48% | 89.49% | 89.20% |
| `TIMER_30s` | 86.01% | 87.92% | 93.23% |
| `VOLUME_DOWN` | 64.79% | 71.43% | 59.52% |
| `VOLUME_UP` | 61.13% | 61.11% | 54.62% |
| `WEATHER` | 56.10% | 67.26% | 65.15% |
