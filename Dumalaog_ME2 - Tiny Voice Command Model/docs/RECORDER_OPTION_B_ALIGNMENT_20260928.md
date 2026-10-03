# Recorder and Option B label alignment

**Date:** 2026-09-28 (Asia/Manila)  
**Scope:** Correct the human recorder schema and its current-model ingestion paths. No model weights were trained or replaced, and existing recordings were not renamed or rewritten.

## Finding

The recorder previously imported `LABELS` and `PHRASES` from `tinyvcm.config`, a retained legacy 26-label, 1.5-second taxonomy. The active intent model uses the 31 labels in `tinyvcm_model.config.LABELS` and a 2.5-second window. This caused new recordings to be stored with different label IDs, and longer commands to be rejected at the legacy window length.

The supplied audit described 130 saved recordings. A current read-only pass of `data/human/manifest.csv` found **458 manifest rows and 230 unique decoded PCM clips**, still from one speaker. Legacy labels map exactly to only 12 of the 31 model classes in the personalized pipeline. The raw manifest and WAV files were not modified.

## Correction

- `record_dataset.py` now gets its command IDs from `tinyvcm_model.config.LABELS` through `tinyvcm_model.recording_labels`. New command rows therefore store a model class verbatim, such as `COLOR_RED`, `TIMER_30s`, or `TEMPERATURE_22`.
- The recorder now saves the model's 2.5-second, 16 kHz mono input window. Speech longer than that window is rejected instead of silently cropped.
- Four separate collection labels remain available: `wake_word`, `_unknown_`, `_background_noise_`, and `_silence_`. They are not among the 31 intent classes. Unrelated speech is retained as a speech example for wake-gate negative training; room noise and silence have separate capture rules.
- The personalized retraining loader now maps canonical model labels by identity and maps only exact legacy equivalents. Paired and SLURP experiment data preparation use that same mapping.
- `tinyvcm.config.LABELS` remains unchanged for historical model artifacts. It is explicitly marked legacy so it cannot be mistaken for the active recorder taxonomy.
- The 13 suggested phrases that previously did not occur verbatim in the Option B transcript manifest were changed to exact observed transcripts; each uses that class's most frequent transcript. An automated data-backed test confirms all 31 intent prompts occur in the current manifest and those 13 are the most frequent choices. Chat's report said 19/31 phrases were exact and 12 were misses, but its table contains 13 classes; the current manifest showed 18/31 exact and 13 misses before this phrase correction.

## Existing legacy recordings

The existing manifest remains intact. The current content-addressed preflight found these unique mapped intent counts:

| Current intent | Existing legacy label | Unique clips |
|---|---|---:|
| `CALL` | `call_mom` | 7 |
| `LIGHT_OFF` | `lights_off` | 24 |
| `LIGHT_ON` | `lights_on` | 7 |
| `LIST_REMINDERS` | `reminders_check` | 10 |
| `NEXT` | `media_next` | 8 |
| `PAUSE` | `media_pause` | 7 |
| `PLAY_MUSIC` | `play_music` | 3 |
| `TIME` | `question_time` | 13 |
| `TIMER_1m` | `timer_1min` | 12 |
| `VOLUME_DOWN` | `volume_down` | 15 |
| `VOLUME_UP` | `volume_up` | 3 |
| `WEATHER` | `question_weather` | 7 |

Legacy `alarm_set`, `media_resume`, `temp_cooler`, `temp_warmer`, `timer_5min`, and `timer_10min` remain unmapped in the intent pipeline. Brightness 25/50/75% are not rounded to a different model class. Those recordings stay available as negatives for wake classification when included by that training path, but are excluded from intent targets unless an exact label mapping exists.

## Verification and limits

Focused checks verified all 31 recorder command IDs equal the canonical model labels, each ID has a suggested phrase, newly saved command WAVs have exactly 40,000 samples at 16 kHz, canonical labels load into the personalized model pipeline, and near-match legacy labels remain unmapped. The existing data audit was read-only. Re-run personalized training only after collecting new canonical recordings; this change alone does not change the existing model or its accuracy.

At the time of this recorder-alignment report, the audio set came from one speaker and did not meet the three-speaker collection target. A later fresh training audit found that the manifest had grown to 524 entries / 292 unique PCM clips from three speaker IDs. The follow-up training results and their clip-split limitation are in [`PERSONAL_WAKE_INTENT_RETRAINING_20260928.md`](PERSONAL_WAKE_INTENT_RETRAINING_20260928.md). Collect the 16 still-missing mapped intent classes and use speaker/session-held-out evaluation before treating personal accuracy as a general performance measure.
