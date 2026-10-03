# ME2 - VCM on Raspberry Pi 5 — fresh personal wake/intent retraining

**Date:** 2026-09-28 (Asia/Manila)  
**Run:** `runs/personalized-vcm-retrain-20260928-190911/`  
**Status:** Scratch-trained local candidate pair; not loaded by the PC assistant and not deployed to the Raspberry Pi.

## What was trained

The final two-stage design was trained from random initialization (seed 231, no pretrained weights): a separate binary `NON_WAKE` / `WAKE_WORD` gate followed by the existing 31-class intent taxonomy. The wake experiment compared a broad-negative model with one that also included the speakers' own command recordings as hard negatives. The selected candidate uses the hard-negative variant because its held-out wake recall was higher at the frozen validation threshold, with the same observed false-accept counts. The intent experiment trained a new 14,527-parameter TinyDSCNN-48 from Option B train data plus a small, exact-mapped personal-data mix. It keeps all 31 output classes.

Both models consume the existing 16 kHz mono, 2.5-second window and 40-band/251-frame log-mel frontend. The held-out Option B test set was opened only after training and export. That test set has been used by earlier project experiments and is not a pristine first-use benchmark.

## Data audit

- Human manifest: 524 saved entries; 292 unique decoded PCM clips after exact deduplication (232 duplicate rows removed).
- Speaker IDs: person-01, person-02, person-03 (3 IDs). Wake labels: 50 unique clips; 35/7/8 train/validation/test.
- Unique mapped personal intent recordings: 175; 15/31 intent classes have mapped personal examples. The intent trainer used 112 unique personal training clips; the held-out personal test contains 42 mapped clips.
- Unmapped legacy command labels were excluded from intent targets, with no guessed relabeling: alarm_set, media_resume, temp_cooler, temp_warmer, timer_10min, timer_5min. Cross-label exact-PCM collisions: 0.
- Important split limit: personal clips were content-deduplicated then split by label, not by speaker. The same speaker IDs can occur in training, validation, and test, so personal test accuracy is not unseen-speaker accuracy.

## Held-out results

### Binary wake comparison

| Variant | INT8 threshold | Validation wake recall | Held-out wake | False accepts: personal non-wake | False accepts: Option B command speech | INT8 bytes |
|---|---:|---:|---:|---:|---:|---:|
| broad negatives | 0.9998670816 | 100.0% (7 clips; 0 false accepts on 1848 non-wake clips) | 4/8 (50.0%) | 0/57 | 0/1798 | 37,425 |
| personal hard negatives | 0.9985431433 | 100.0% (7 clips; 0 false accepts on 1848 non-wake clips) | 7/8 (87.5%) | 0/57 | 0/1798 | 37,425 |

The selected wake threshold was chosen on validation only. A zero false-accept count on these fixed clips does not estimate false activations per hour in continuous audio. The selected model detected 7 of 8 held-out wake utterances; this result is too small to support a production reliability claim.

### 31-class intent comparison

| Model | Personal held-out accuracy | Personal macro-F1 | Option B held-out accuracy | Option B macro-F1 | INT8 bytes |
|---|---:|---:|---:|---:|---:|
| Canonical Option B INT8 | 45.2% (42 clips) | 0.399 | 96.3% (1,798 clips) | 0.963 | 39,315 |
| Fresh personalized INT8 | 88.1% (42 clips) | 0.918 | 96.0% (1,798 clips) | 0.960 | 39,315 |

The personalized model improves the within-dataset personal result by a large margin, while reducing reused Option B accuracy by about 0.28 percentage points. This is a practical tradeoff for this user, not evidence of generalization. It remains a candidate until separately audited and tested with live audio.

### Personal intent clips by recorded label

| Recorded label | Mapped class | Held-out clips | Correct | Accuracy |
|---|---|---:|---:|---:|
| `ALARM_6_00AM` | `ALARM_6_00AM` | 2 | 2 | 100.0% |
| `ALARM_8_00AM` | `ALARM_8_00AM` | 2 | 2 | 100.0% |
| `LIGHT_ON` | `LIGHT_ON` | 2 | 2 | 100.0% |
| `PAUSE` | `PAUSE` | 3 | 2 | 66.7% |
| `PLAY_MUSIC` | `PLAY_MUSIC` | 3 | 3 | 100.0% |
| `STOP` | `STOP` | 2 | 2 | 100.0% |
| `call_mom` | `CALL` | 2 | 2 | 100.0% |
| `lights_off` | `LIGHT_OFF` | 5 | 3 | 60.0% |
| `lights_on` | `LIGHT_ON` | 2 | 2 | 100.0% |
| `media_next` | `NEXT` | 2 | 1 | 50.0% |
| `media_pause` | `PAUSE` | 2 | 1 | 50.0% |
| `play_music` | `PLAY_MUSIC` | 1 | 1 | 100.0% |
| `question_time` | `TIME` | 3 | 3 | 100.0% |
| `question_weather` | `WEATHER` | 2 | 2 | 100.0% |
| `reminders_check` | `LIST_REMINDERS` | 2 | 2 | 100.0% |
| `timer_1min` | `TIMER_1m` | 3 | 3 | 100.0% |
| `volume_down` | `VOLUME_DOWN` | 3 | 3 | 100.0% |
| `volume_up` | `VOLUME_UP` | 1 | 1 | 100.0% |

## Next recordings to collect

The current Option B taxonomy has 16 classes with no mapped personal examples: `ALARM_9_00PM`, `BRIGHTNESS_100`, `BRIGHTNESS_20`, `BRIGHTNESS_60`, `COLOR_BLUE`, `COLOR_GREEN`, `COLOR_RED`, `CREATE_REMINDER_DRINK_WATER`, `CREATE_REMINDER_EXERCISE`, `CREATE_REMINDER_STUDY`, `MESSAGE`, `TEMPERATURE_18`, `TEMPERATURE_22`, `TEMPERATURE_26`, `TIMER_10s`, `TIMER_30s`. Record these exact commands from all three speakers, in multiple sessions and near/far/noisy conditions. Also continue collecting wake calls, wake-like phrases, ordinary speech, silence, and room noise. Do not treat the unmapped legacy timer/thermostat/generic alarm labels as one of these target classes.

## Exported candidate and verification

- Dated executed notebook: `notebooks/ME2_Personalized_Wake_Intent_Retraining_20260928-190911.ipynb`.
- Fresh run ONNX artifacts: `wake_with_personal_intents/models/wake_personalized_int8.onnx` (37,425 bytes), `intent_personalized/models/intent_personalized_int8.onnx` (39,315 bytes); combined size 76,740 bytes (<500 KB).
- Versioned Pi-review staging: `deployment/personalized_candidate_retrain-20260928-190911/`. Both staged ONNX models passed CPU ONNX Runtime smoke tests on a `(1,1,40,251)` zero-feature input and returned finite outputs with the expected 2- and 31-class shapes.
- Focused data-mapping/retraining tests: 9 passed. Python compilation and `git diff --check` passed. Report notebook executed successfully with five code cells producing outputs. ONNX Runtime CPU smoke tests passed for both staged models.
- Local CPU model-only p95: wake about 0.54 ms; intent about 0.52 ms. These are Windows PC results, not Raspberry Pi timings and not end-to-end frontend latency.
- No microphone capture, app model replacement, Pi access, GPIO test, or deployment was performed during this run. The Pi remains out of scope/powered down.

Full machine-readable metrics and provenance are in `final_evaluation.json`, `dataset_audit.json`, each `run_info.json`, and the corresponding `models/export_summary.json` files under the run directory.
