# Recorder phrase alignment and model audit

**Date:** 2026-10-01 (Asia/Manila)  
**Dataset:** ME2 Spoken Command Dataset  
**Latest deployed pair:** `me2-vcm-20261001-human936` (scratch training; promoted at the user's request after the original candidate audit)

## Phrase and label audit

- All 31 command labels are present in the model and recorder taxonomy. Every class has a ground-truth phrase file under `docs/ground_truth_phrases/`.
- Each file equals the most frequent exact transcript for that class in the model-training corpus; the dataset audit checked all 31 files and recorder prompts against the transcripts.
- The recorder loads those files directly and now writes the displayed phrase to `data/human/manifest.csv` as `suggested_phrase` when a new take is saved.
- The manifest has 936 historical takes and 0 populated prompt values. Those rows stay blank: they were saved before prompt logging and their actual utterances cannot be reconstructed. The field is the displayed prompt, not automatic transcription.
- The historical `lights_off` prompt was `Turn off the lights`; the most frequent training transcript is `Kill the lights` (200 clips). The old takes have no transcript metadata, so the wording they actually contain is unverified. New takes now show and record `Kill the lights`.
- The personal WAV manifest has 936 rows, 699 content-unique PCM groups, 235 collapsed exact duplicates, and one ambiguous cross-label PCM group excluded from training while preserving both original files. It spans three speakers, but most clips are `quiet-near` (631); `quiet-far` has 58 and `fan-near` 10.
- Six old personal labels remain unmapped (`alarm_set`, `media_resume`, `temp_cooler`, `temp_warmer`, `timer_5min`, `timer_10min`). They are preserved and excluded from the 31-way intent training rather than silently relabeled.

### Canonical recorder prompts

| Label | Suggested phrase | Unique human examples |
|---|---|---:|
| `ALARM_6_00AM` | Wake me up at 6 AM | 9 |
| `ALARM_8_00AM` | Wake me up at 8 AM | 10 |
| `ALARM_9_00PM` | Set an alarm for 9 PM | 11 |
| `BRIGHTNESS_100` | Brightness 100 percent | 18 |
| `BRIGHTNESS_20` | Adjust brightness to 20 percent | 13 |
| `BRIGHTNESS_60` | Brightness 60 percent | 12 |
| `CALL` | Make a phone call | 31 |
| `COLOR_BLUE` | Change color to blue | 13 |
| `COLOR_GREEN` | Change color to green | 15 |
| `COLOR_RED` | Set color to red | 15 |
| `CREATE_REMINDER_DRINK_WATER` | Create a reminder to drink water | 15 |
| `CREATE_REMINDER_EXERCISE` | Remind me to exercise | 15 |
| `CREATE_REMINDER_STUDY` | Create a reminder to study | 14 |
| `LIGHT_OFF` | Kill the lights | 24 |
| `LIGHT_ON` | Turn on the lights | 14 |
| `LIST_REMINDERS` | Show my reminders | 10 |
| `MESSAGE` | Send my message | 20 |
| `NEXT` | Next song | 30 |
| `PAUSE` | Pause audio | 63 |
| `PLAY_MUSIC` | Start music | 39 |
| `STOP` | Stop playing | 41 |
| `TEMPERATURE_18` | Change the temperature to 18 degrees | 16 |
| `TEMPERATURE_22` | Temperature 22 degrees | 18 |
| `TEMPERATURE_26` | Change the temperature to 26 degrees | 15 |
| `TIME` | What time is it? | 13 |
| `TIMER_10s` | Countdown for 10 seconds | 15 |
| `TIMER_1m` | Start a timer for 1 minute | 12 |
| `TIMER_30s` | Countdown for 30 seconds | 15 |
| `VOLUME_DOWN` | Turn the volume down | 15 |
| `VOLUME_UP` | Increase the volume | 3 |
| `WEATHER` | What's the weather? | 7 |

## Model audit

| Pair / test | Accuracy | Macro F1 | N | Notes |
|---|---:|---:|---:|---|
| Previously staged INT8 — reference test | 97.22% | 97.22% | 1798 | Same reused reference test partition |
| Previously staged INT8 — personal holdout | 92.08% | 93.62% | 101 | Earlier split; not directly paired with candidate |
| New candidate INT8 — reference test | 96.89% | 96.90% | 1798 | Reused partition; previously evaluated in this project |
| New candidate INT8 — personal holdout | 92.17% | 92.44% | 115 | Content-deduplicated clip split; speakers overlap between splits |

The candidate used 14,370 reference training clips plus 374 unique eligible personal clips; it trained both networks from random initialization (wake 30 epochs; intent 60 epochs; seeds 231/232). It has 14,527 intent parameters and 13,106 wake parameters. The candidate ONNX files are 39,315 bytes (intent) and 37,424 bytes (wake), 76,739 bytes combined. PC CPU model-only p95 is 0.766 ms for intent and 0.555 ms for wake; this is not Raspberry Pi latency.

The candidate reference accuracy is 96.89%, down from 97.22% for the staged pair. Its personal score is 92.17% versus 92.08%, but those personal splits have different sizes (115 vs 101) and are not a controlled paired comparison. The pair does not achieve 95% for every class. The user subsequently requested the latest trained pair on the Pi; it is now promoted and hash-verified there. This deployment does not erase the measured regression or evaluation limitations.

### Per-class performance

Values are **recall / F1**. Recall is the per-class correct-class rate; support is the number of test clips. The full precision/recall/F1/support values for both pairs are in [`phrase-prompt-class-metrics-20261001.csv`](phrase-prompt-class-metrics-20261001.csv).

| Intent | Unique human examples | Staged reference R/F1 | Candidate reference R/F1 | Candidate personal R/F1 (n) |
|---|---:|---:|---:|---:|
| `ALARM_6_00AM` | 9 | 98.28% / 99.13% | 94.83% / 97.35% | 100.00% / 80.00% (2.0) |
| `ALARM_8_00AM` | 10 | 100.00% / 99.12% | 100.00% / 96.55% | 100.00% / 80.00% (2.0) |
| `ALARM_9_00PM` | 11 | 100.00% / 100.00% | 98.28% / 98.28% | 100.00% / 100.00% (3.0) |
| `BRIGHTNESS_100` | 18 | 100.00% / 96.77% | 98.33% / 93.65% | 75.00% / 85.71% (4.0) |
| `BRIGHTNESS_20` | 13 | 98.33% / 97.52% | 88.33% / 92.98% | 100.00% / 85.71% (3.0) |
| `BRIGHTNESS_60` | 12 | 96.67% / 98.31% | 100.00% / 100.00% | 100.00% / 100.00% (3.0) |
| `CALL` | 31 | 96.30% / 97.20% | 98.15% / 96.36% | 100.00% / 100.00% (7.0) |
| `COLOR_BLUE` | 13 | 91.67% / 94.02% | 95.00% / 97.44% | 100.00% / 100.00% (3.0) |
| `COLOR_GREEN` | 15 | 96.67% / 98.31% | 98.33% / 99.16% | 100.00% / 100.00% (3.0) |
| `COLOR_RED` | 15 | 100.00% / 94.49% | 100.00% / 98.36% | 66.67% / 80.00% (3.0) |
| `CREATE_REMINDER_DRINK_WATER` | 15 | 100.00% / 97.56% | 100.00% / 97.56% | 100.00% / 100.00% (3.0) |
| `CREATE_REMINDER_EXERCISE` | 15 | 98.33% / 99.16% | 98.33% / 96.72% | 100.00% / 100.00% (3.0) |
| `CREATE_REMINDER_STUDY` | 14 | 98.33% / 99.16% | 96.67% / 98.31% | 100.00% / 100.00% (3.0) |
| `LIGHT_OFF` | 24 | 96.67% / 94.31% | 98.33% / 96.72% | 60.00% / 66.67% (5.0) |
| `LIGHT_ON` | 14 | 92.59% / 95.24% | 98.15% / 97.25% | 100.00% / 100.00% (4.0) |
| `LIST_REMINDERS` | 10 | 93.10% / 94.74% | 100.00% / 96.67% | 100.00% / 80.00% (2.0) |
| `MESSAGE` | 20 | 94.83% / 97.35% | 93.10% / 95.58% | 100.00% / 85.71% (3.0) |
| `NEXT` | 30 | 98.33% / 99.16% | 96.67% / 97.48% | 83.33% / 90.91% (6.0) |
| `PAUSE` | 63 | 98.21% / 98.21% | 100.00% / 99.12% | 72.73% / 84.21% (11.0) |
| `PLAY_MUSIC` | 39 | 100.00% / 98.31% | 100.00% / 100.00% | 85.71% / 80.00% (7.0) |
| `STOP` | 41 | 96.30% / 94.55% | 100.00% / 97.30% | 100.00% / 100.00% (7.0) |
| `TEMPERATURE_18` | 16 | 93.33% / 95.73% | 95.00% / 90.48% | 100.00% / 100.00% (3.0) |
| `TEMPERATURE_22` | 18 | 100.00% / 96.00% | 90.00% / 93.91% | 100.00% / 100.00% (4.0) |
| `TEMPERATURE_26` | 15 | 96.67% / 98.31% | 95.00% / 97.44% | 100.00% / 100.00% (3.0) |
| `TIME` | 13 | 90.74% / 95.15% | 92.59% / 93.46% | 100.00% / 100.00% (3.0) |
| `TIMER_10s` | 15 | 95.00% / 95.00% | 93.33% / 96.55% | 100.00% / 100.00% (3.0) |
| `TIMER_1m` | 12 | 100.00% / 99.12% | 98.21% / 97.35% | 100.00% / 100.00% (3.0) |
| `TIMER_30s` | 15 | 95.00% / 96.61% | 98.33% / 99.16% | 100.00% / 100.00% (3.0) |
| `VOLUME_DOWN` | 15 | 100.00% / 98.11% | 98.08% / 97.14% | 100.00% / 100.00% (3.0) |
| `VOLUME_UP` | 3 | 98.33% / 99.16% | 93.33% / 96.55% | 100.00% / 66.67% (1.0) |
| `WEATHER` | 7 | 100.00% / 98.11% | 98.08% / 99.03% | 100.00% / 100.00% (2.0) |

Candidate reference classes below 95% F1: `BRIGHTNESS_100`, `BRIGHTNESS_20`, `TEMPERATURE_18`, `TEMPERATURE_22`, `TIME`.  
Candidate reference classes below 95% recall: `ALARM_6_00AM`, `BRIGHTNESS_20`, `MESSAGE`, `TEMPERATURE_22`, `TIME`, `TIMER_10s`, `VOLUME_UP`.  
Candidate personal classes below 95% F1: `ALARM_6_00AM`, `ALARM_8_00AM`, `BRIGHTNESS_100`, `BRIGHTNESS_20`, `COLOR_RED`, `LIGHT_OFF`, `LIST_REMINDERS`, `MESSAGE`, `NEXT`, `PAUSE`, `PLAY_MUSIC`, `VOLUME_UP`.

The personal report has very small per-class support (often 1–5 clips); its percentages should not be interpreted as precise population estimates. The aggregate result also does not establish performance on unseen speakers or rooms.

### Focused light-command check

On the candidate's held-out personal clips for these four light commands, the staged intent model predicted `COLOR_RED` 3/3, `COLOR_GREEN` 3/3, `COLOR_BLUE` 3/3 and `LIGHT_OFF` 4/5. The fresh candidate predicted `COLOR_RED` 2/3, `COLOR_GREEN` 3/3, `COLOR_BLUE` 3/3 and `LIGHT_OFF` 3/5. The candidate's wrong red clip was classified as `VOLUME_UP`; two wrong `LIGHT_OFF` clips were classified as `MESSAGE` and `ALARM_8_00AM`. The staged model's apparent advantage is diagnostic only: the candidate split re-splits historical personal clips, so some held-out clips may overlap the staged model's earlier training data. The candidate split is the valid personal estimate for the candidate.

On these 14 clips, applying a 0.95 top-class confidence rejection cutoff would retain only 2/14 predictions from either pair (one blue and one green); it would reject every red and `LIGHT_OFF` prediction. This small check does not support setting a universal 0.95 intent cutoff.

The Assistant's local action dispatcher and simulated RGB device were checked independently for red, green and blue. Each sets its own exact simulated color (`#ff0000`, `#00ff00`, `#0000ff`) and turns the light on; `LIGHT_OFF` sets brightness to 0 and the simulated output to `#000000`. The Pi SSH connection timed out and its local UI tunnel was absent during this check, so physical GPIO color changes were not re-tested in this session.

## Threshold finding

- The currently staged wake model uses a **0.95 wake-probability cutoff**. Its saved replay accepted 8/8 personal wake placements and rejected 120/120 personal non-wake clips and 1,798/1,798 reference command clips. This is not a live false-activations-per-hour test.
- The new wake candidate's validation-selected INT8 threshold is 0.999827; at that operating point it accepts 6/8 held-out personal wakes and accepts 0/134 personal non-wakes and 0/1798 reference commands. Only eight wake positives were held out, so the threshold is unstable and candidate wake recall is 75%.
- The intent runtime currently uses **top-class argmax**; there are no 31 individual intent confidence thresholds. Setting a confidence cutoff of 0.95 would reject predictions below that score, changing coverage and false-command rates. It cannot raise measured accuracy or guarantee 95% correctness. Calibrate a rejection cutoff on validation data and measure the accepted-command coverage before enabling it.

## Recording recommendation

Keep the new exact prompt as ground truth for each new take. Prioritize classes with both weak held-out performance and small/uncertain personal support: `LIGHT_OFF`, `TIME`, `BRIGHTNESS_20`, `BRIGHTNESS_100`, `TEMPERATURE_18`, `TEMPERATURE_22`, `PAUSE`, `PLAY_MUSIC`, `VOLUME_UP`, and the alarm/reminder/color classes listed below 95% in the metrics table. `VOLUME_UP` has only 3 eligible unique human clips; `WEATHER` has 7; `ALARM_6_00AM` 9; `ALARM_8_00AM` and `LIST_REMINDERS` 10 each.

Target at least 30 distinct clips per weak class across three or more speakers (about 10 per speaker), captured in quiet-near, quiet-far, and fan/noise conditions. Keep an entire speaker out of the next final evaluation; the current personal split is not speaker-disjoint. Re-train and evaluate on a new frozen test set after the prompt-aligned recordings are collected. These counts are a data-collection target, not a promise of 95% performance.

## Verification performed

- `scripts/audit_me2_dataset.py`: PASS; all 31 labels and exact transcript prompts match, 936 historical rows preserved, classes present.
- Focused held-out audio replay: staged and candidate INT8 models were run on the same 14 `COLOR_RED` / `COLOR_GREEN` / `COLOR_BLUE` / `LIGHT_OFF` personal test rows; individual predictions and wrong-class outcomes are summarized above. This is not a clean paired comparison for the staged model.
- Focused simulated RGB action check: red, green, blue, and off return the expected per-command color/brightness state. Physical Pi checks were unavailable (SSH timeout; no local tunnel).
- `pytest -q`: PASS, 51 tests.
- Candidate scratch run completed and wrote its full report under `runs/me2-vcm-20261001-human936/final_evaluation.json`.
- No candidate was staged or deployed to Raspberry Pi; no Pi microphone test was performed.

## Subsequent deployment verification

The user requested the newest trained pair, so human936 was promoted and installed in `/home/dalmacio/Desktop/dandan`. Fixed operating wake threshold **0.95** replay accepted **8/8** personal positives, **0/134** personal negatives and **0/1,798** reference commands. This is distinct from the validation-selected near-one cutoff in the original report. All 24 bundle files, live model hashes, two/31-class ordering, Assistant/Studio routes and ARM64 inference passed. Pi frontend-plus-model p95 was **5.698 ms wake / 5.210 ms intent**. The app runs manually; no boot service was enabled. The Fifine microphone disconnected and was reconnected during the session; the app recovered capture without restarting (audio age 0.06 s, zero drops/errors). No controlled live voice trial was scored. `deployment/last_pi_deployment.json` is the installation receipt.
