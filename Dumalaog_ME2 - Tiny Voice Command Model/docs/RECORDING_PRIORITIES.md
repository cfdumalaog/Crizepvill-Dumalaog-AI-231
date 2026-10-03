# ME2 - VCM on Raspberry Pi 5: recording priorities

Latest deployed pair: `me2-vcm-20261001-wake-refresh`; intent weights retained from `me2-vcm-20261001-human936`, so intent priorities below remain unchanged. These are measured weaknesses, not missing label choices. Suggested phrases below are read from the canonical per-class files; use the exact displayed recorder phrase. Existing recordings remain preserved.

## Personal holdout below 95% F1

| Intent | Exact recorder phrase | Personal recall | Personal F1 | Holdout clips |
|---|---|---:|---:|---:|
| `LIGHT_OFF` | Kill the lights | 60.00% | 66.67% | 5 |
| `COLOR_RED` | Set color to red | 66.67% | 80.00% | 3 |
| `PAUSE` | Pause audio | 72.73% | 84.21% | 11 |
| `PLAY_MUSIC` | Start music | 85.71% | 80.00% | 7 |
| `VOLUME_UP` | Increase the volume | 100.00% | 66.67% | 1 |
| `BRIGHTNESS_100` | Brightness 100 percent | 75.00% | 85.71% | 4 |
| `BRIGHTNESS_20` | Adjust brightness to 20 percent | 100.00% | 85.71% | 3 |
| `ALARM_6_00AM` | Wake me up at 6 AM | 100.00% | 80.00% | 2 |
| `ALARM_8_00AM` | Wake me up at 8 AM | 100.00% | 80.00% | 2 |
| `LIST_REMINDERS` | Show my reminders | 100.00% | 80.00% | 2 |
| `MESSAGE` | Send my message | 100.00% | 85.71% | 3 |
| `NEXT` | Next song | 83.33% | 90.91% | 6 |

Start with LIGHT_OFF, COLOR_RED, PAUSE, PLAY_MUSIC and VOLUME_UP. LIGHT_OFF errors were MESSAGE and ALARM_8_00AM; COLOR_RED had a VOLUME_UP error. A 100% recall with low F1 means other commands are mistaken for that class, so record contrastive examples too. VOLUME_UP has only three eligible unique personal recordings and one held-out example.

## Additional reference-test weaknesses

| Intent | Exact recorder phrase | Reference recall | Reference F1 |
|---|---|---:|---:|
| `TEMPERATURE_18` | Change the temperature to 18 degrees | 95.00% | 90.48% |
| `TEMPERATURE_22` | Temperature 22 degrees | 90.00% | 93.91% |
| `TIME` | What time is it? | 92.59% | 93.46% |

BRIGHTNESS_100 and BRIGHTNESS_20 also miss 95% reference F1 and are already listed above. CALL, LIGHT_ON, STOP, COLOR_GREEN and COLOR_BLUE no longer appear in the latest below-95%-F1 list. Green/blue each have only three personal holdout clips, so their 100% scores are small-sample evidence.

Record distinct takes at different distances, normal/soft voices and mild fan noise, preferably across at least three speakers. Aim for at least 30 distinct examples per weak class and reserve a new speaker/session for evaluation. Do not duplicate existing WAVs to inflate counts. Personal test clips share speakers and the reference test is reused; these scores do not guarantee live accuracy.

The wake cutoff is 0.95. Saved-file intent metrics use top-class argmax. Live execution uses confidence >= 0.68 and a top-two probability margin >= 0.15 for every class; there are no class-specific cutoffs. A 0.95 confidence cutoff is not a 95% accuracy guarantee.

Wake refresh: 1,025 saved rows, 65 distinct wake clips, 45/9/11 train/validation/test; deployed cutoff 0.95 accepted 11/11 held-out wakes and 0/136 personal plus 0/1,798 reference negatives on Windows and Pi. Continue collecting new speech rather than resaving identical takes; measure live false activations separately.
