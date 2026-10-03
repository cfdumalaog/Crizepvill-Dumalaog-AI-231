# ME2 - VCM on Raspberry Pi 5: wake refresh and deployment audit

Date: 2026-10-01 (Asia/Manila)

Verified deployed pair: `me2-vcm-20261001-wake-refresh`. Wake was trained from random initialization for 30 epochs (selected epoch 16); intent weights are retained byte-identical from `me2-vcm-20261001-human936`. No new mapped intent recordings or split changes were found. Both models originally use scratch weights; no pretrained model is used.

## Data and phrase evidence

Frozen source: `human_manifest_snapshot.csv` in the run. 1,025 raw rows; 724 eligible unique PCM groups; 299 duplicate rows collapsed; one cross-label collision group excluded with originals preserved. Compared with the preceding run: 72 new wake rows and 17 negative rows, adding 23 unique wake waveforms and two unique negative waveforms. Seven speaker IDs exist; IDs alone do not verify seven different people. Wake has 65 unique clips split 45/9/11 for training/validation/test.

The binary task is NON_WAKE versus WAKE_WORD. All 14,370 reference training command clips are negatives; eligible personal command speech, unknown speech, silence and noise are negatives too. Validation/test examples never enter optimization or INT8 calibration.

All 31 canonical phrase files equal their class's most frequent exact reference transcript and remain unchanged. `data/human/manifest.xlsx` contains four sheets: Command phrases, Recordings, Intent performance and Wake performance. All 1,025 rows and 35 recorder prompts verified; 89 prompts logged at save and 936 historical prompts unknown. Displayed prompts are not spoken transcripts. The recorder source includes a Refresh Excel manifest button; its live instance needs one restart after saving any loaded take to expose the control.

## Performance at the deployed operating point

| Model/test | Accuracy | Macro F1 / wake F1 | Support | Cutoff |
|---|---:|---:|---:|---|
| Wake binary | 100.00% | 100.00% | 11 wakes + 136 personal negatives + 1,798 reference negatives | 0.95 |
| Intent reference | 96.89% | 96.90% | 1,798 | Offline argmax; live 0.68 / 0.15 margin |
| Intent personal | 92.17% | 92.44% | 115 | Offline argmax; live 0.68 / 0.15 margin |

Wake: 11/11 accepted; 0/136 personal and 0/1,798 reference negative accepts on Windows and Pi. The separate validation-selected INT8 cutoff 0.9975437522 accepted 10/11 held-out wakes. The previous pair accepted 8/8 at 0.95 on its earlier split; the supports differ, so this is not a direct improvement percentage. Personal splits overlap speakers; the reference benchmark is reused. This is saved-file replay, not room false activations per hour.

## Intents needing more recordings

| Intent | Exact recorder phrase | Personal recall | Personal F1 | Personal n | Reference F1 |
|---|---|---:|---:|---:|---:|
| `ALARM_6_00AM` | Wake me up at 6 AM | 100.00% | 80.00% | 2 | 97.35% |
| `ALARM_8_00AM` | Wake me up at 8 AM | 100.00% | 80.00% | 2 | 96.55% |
| `BRIGHTNESS_100` | Brightness 100 percent | 75.00% | 85.71% | 4 | 93.65% |
| `BRIGHTNESS_20` | Adjust brightness to 20 percent | 100.00% | 85.71% | 3 | 92.98% |
| `COLOR_RED` | Set color to red | 66.67% | 80.00% | 3 | 98.36% |
| `LIGHT_OFF` | Kill the lights | 60.00% | 66.67% | 5 | 96.72% |
| `LIST_REMINDERS` | Show my reminders | 100.00% | 80.00% | 2 | 96.67% |
| `MESSAGE` | Send my message | 100.00% | 85.71% | 3 | 95.58% |
| `NEXT` | Next song | 83.33% | 90.91% | 6 | 97.48% |
| `PAUSE` | Pause audio | 72.73% | 84.21% | 11 | 99.12% |
| `PLAY_MUSIC` | Start music | 85.71% | 80.00% | 7 | 100.00% |
| `TEMPERATURE_18` | Change the temperature to 18 degrees | 100.00% | 100.00% | 3 | 90.48% |
| `TEMPERATURE_22` | Temperature 22 degrees | 100.00% | 100.00% | 4 | 93.91% |
| `TIME` | What time is it? | 100.00% | 100.00% | 3 | 93.46% |
| `VOLUME_UP` | Increase the volume | 100.00% | 66.67% | 1 | 96.55% |

Prioritize LIGHT_OFF, COLOR_RED, PAUSE, PLAY_MUSIC and VOLUME_UP. Collect at least 30 distinct takes per weak class over multiple real speakers/sessions, near/far distances and mild fan noise. Reserve a new session or entire speaker for testing. Include contrasting MESSAGE/ALARM_8_00AM for LIGHT_OFF errors. Do not resave an identical waveform under different conditions or speaker IDs as if it were a new recording.

## Raspberry Pi, RGB and integrity evidence

Installed at `/home/dalmacio/Desktop/dandan`; Windows launcher/package remains `C:/Users/danda/Desktop/dandan`. Release ZIP SHA-256: `83f1e59c710e4d124bf7f685af6f0b045ae4588f7e50cb9e84bf5278c0ed5ea7`. All 24 payload hashes verified; live API source run and both model hashes match. Boot autostart remains off. Repeated sync confirmed no restart needed.

Wake SHA-256: `6c2e941580d8ea778ed664c929e893c226efb91e3e787e65837908c81153deac` (37,425 B). Intent SHA-256: `12984401fe312758dbd7ecd53ae9358b0a41b27ffcd024a9ae263614e2e7b977` (39,315 B). Combined: 76,740 B.

Pi ARM64 frontend-plus-model p95: wake 5.781 ms; intent 5.330 ms. Frozen feature replay reproduced every per-class precision/recall/F1 value on Pi and Windows for both intent partitions. No actions were invoked by that inference audit. Temporary Pi fixtures/scripts/results were removed after retrieving reports.

`LIGHT_ON` now sets white/all three channels at 100%, including after color and off. Individual colors still select one channel, and brightness scales the selected color. For each red/green/blue -> off -> on sequence, two settled `pinctrl` readings verified BCM 17/27/22 at the expected high/low levels. LEDs were left off. This proves GPIO levels, not an optical color measurement.

Evidence: `deployment/validation/pi_saved_audio_wake_refresh_20261001.json`, `pi_rgb_wake_refresh_20261001.json`, `last_pi_deployment.json`. Neither Kiko folder nor the separate Pi legacy project was inspected or modified. Active code, imports, launchers, staged pair and new payload use neutral project names. Archives keep original provenance.

## Verification and remaining work

63 project tests passed; label/phrase audit passed; Python compile and project diff checks passed. Current notebook executed all 13 cells / 7 code cells with zero errors and displays retained intent provenance and latest deployment. Workbook contents and every sheet preview were checked.

The last reachable Pi check listed no capture hardware (`arecord -l` empty); the app was STANDBY with inactive capture. Current reachability/capture is unverified after the later connection failure. Reconnect the USB mic and use Refresh audio devices / the microphone selector before live trials. Per-intent 95% F1 is not met. The existing PDF is an older build; public networking remains on hold.

Final source audit correction: live intent execution has always used confidence >= 0.68 and top-two margin >= 0.15. These now share configuration constants and appear in API/metadata/Excel; offline accuracy uses argmax and does not measure rejected-command coverage. The final release was re-synchronized with this reporting change.

Final source audit correction: live intent execution has always used confidence >= 0.68 and top-two margin >= 0.15. These now share configuration constants and are included in API/metadata/Excel; offline accuracy uses argmax and does not measure rejected-command coverage. Final API-reporting package synchronization is being retried after hostname resolution failed; the trained model hashes are unchanged from the already verified deployment.

Prepared final reporting package: `deployment/dist/dandan-vcm-me2-vcm-20261001-wake-refresh-20261001-152052-043003.zip`, SHA `b9853db0f4c47a6c4096af7a71d7e32be46934d9ad96480bb1dcd8d1f42039ff`; status pending. Last verified deployed package SHA remains `83f1e59c710e4d124bf7f685af6f0b045ae4588f7e50cb9e84bf5278c0ed5ea7`. Current wake and intent weights are identical across both packages.
