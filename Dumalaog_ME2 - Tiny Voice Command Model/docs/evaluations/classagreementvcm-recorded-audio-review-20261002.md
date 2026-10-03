# classagreementvcm Recorded-Audio Review

**Date:** 2026-10-02 (Asia/Manila)<br>
**Status:** Active-model replay and phrase audit complete. No classagreementvcm model exists yet, so there is no candidate-vs-baseline comparison.

## Result

The classagreementvcm work currently in the repository is a Phase 1 dataset/provenance audit. It did not train an intent model. The current active intent model remains a 31-class model; the proposed classagreementvcm model would add an OOS output, but that fit is blocked on schema and rights decisions. Therefore, it is not possible to say that classagreementvcm is better or worse yet.

To answer the saved-audio part of the request, I replayed the current active INT8 intent model on the 115 mapped command clips in the existing personal test split. The split comes from the recorded human data and has been evaluated before. The command audio files are deduplicated in the saved split manifest. The frozen gold test and holdout were not read or scored, and the replay executed no device actions.

| Measure | Active 31-class model on personal command replay |
|---|---:|
| Clips / prediction failures | 115 / 0 |
| Correct / accuracy | 106 / 92.17% |
| Macro F1 | 92.44% |
| Command leaves represented | 31 / 31 |
| Local Windows preprocessing + inference latency | p50 2.047 ms; p95 2.496 ms |

This reproduces the prior 92.17% accuracy / 92.44% macro F1 result. It is not a new independent score: the personal test was previously reported, and the split is clip-disjoint but not speaker-disjoint. The latency is from this Windows host, not the Raspberry Pi.

### Personal classes below 95% F1

| Intent | Correct / support | F1 | Suggested next recording action |
|---|---:|---:|---|
| LIGHT_OFF | 3 / 5 | 66.67% | Highest priority; record more varied natural voices and contrast against MESSAGE and alarm requests. |
| VOLUME_UP | 1 / 1 | 66.67% | Highest priority; one test example is not enough to establish recall or generalization. |
| ALARM_6_00AM | 2 / 2 | 80.00% | Record all three exact Option B variants with several speakers. |
| ALARM_8_00AM | 2 / 2 | 80.00% | Record all three exact Option B variants with several speakers. |
| COLOR_RED | 2 / 3 | 80.00% | Add the other Option B phrasings and contrast color requests with volume commands. |
| PLAY_MUSIC | 6 / 7 | 80.00% | Record varied voices; include the exact three Option B variants. |
| LIST_REMINDERS | 2 / 2 | 80.00% | Record all three benchmark phrasings; include natural speech variants in training only if labeled. |
| PAUSE | 8 / 11 | 84.21% | High priority; add the other Option B phrasings and contrast PAUSE with PLAY_MUSIC and alarms. |
| BRIGHTNESS_100 | 3 / 4 | 85.71% | Add distinct voices and contrast with lower brightness values. |
| BRIGHTNESS_20 | 3 / 3 | 85.71% | Recall is 100% on three clips, but false positives lower F1; add contrastive brightness levels. |
| MESSAGE | 3 / 3 | 85.71% | Add more varied voices; distinguish from LIGHT_OFF. |
| NEXT | 5 / 6 | 90.91% | Record more voices and include all three Option B phrasings. |

Several apparently perfect classes still have only 2-4 test examples, so their 100% recall is weak evidence. The existing recording-priorities document also flags COLOR_BLUE and COLOR_GREEN as having only three personal held-out clips each, and TEMPERATURE_18, TEMPERATURE_22, and TIME as reference-set weaknesses.

### Leading confusions

- PAUSE -> PLAY_MUSIC: 2 clips
- PAUSE -> ALARM_6_00AM: 1 clip
- BRIGHTNESS_100 -> BRIGHTNESS_20: 1 clip
- COLOR_RED -> VOLUME_UP: 1 clip
- PLAY_MUSIC -> LIST_REMINDERS: 1 clip
- LIGHT_OFF -> MESSAGE: 1 clip
- LIGHT_OFF -> ALARM_8_00AM: 1 clip
- NEXT -> LIGHT_OFF: 1 clip

## Phrase Alignment

I compared the current canonical recorder suggestion for each of the 31 leaves with the corresponding Option B command/value and its three phrases in the pinned 93-row sheet. This compares displayed suggestions, not the words that were actually spoken in old WAV files.

- 22 of 31 current suggestions exactly match one Option B phrase.
- 6 differ only by capitalization; the spoken wording and slot are otherwise the same.
- 3 alarm suggestions omit the `:00` shown in the Option B phrase text. Their intended slot values remain the same, but the visible benchmark strings are not exact matches.

| Leaf(s) | Current suggestion | Option B text / difference |
|---|---|---|
| COLOR_BLUE | Change color to blue | Change color to Blue (capitalization only) |
| COLOR_GREEN | Change color to green | Change color to Green (capitalization only) |
| COLOR_RED | Set color to red | Set color to Red (capitalization only) |
| CREATE_REMINDER_DRINK_WATER | Create a reminder to drink water | Create a reminder to Drink water (capitalization only) |
| CREATE_REMINDER_EXERCISE | Remind me to exercise | Remind me to Exercise (capitalization only) |
| CREATE_REMINDER_STUDY | Create a reminder to study | Create a reminder to Study (capitalization only) |
| ALARM_6_00AM | Wake me up at 6 AM | Option B has “Wake me up at 6:00 AM” (also offers Alarm / Set an alarm phrasings) |
| ALARM_8_00AM | Wake me up at 8 AM | Option B has “Wake me up at 8:00 AM” (also offers Alarm / Set an alarm phrasings) |
| ALARM_9_00PM | Set an alarm for 9 PM | Option B has “Set an alarm for 9:00 PM” (also offers Alarm / Wake me up phrasings) |

The human manifest contains 89 non-empty `suggested_phrase` values, but all belong to `wake_word` (72), `_unknown_` (7), `_background_noise_` (7), or `_silence_` (3). It has zero prompt-text entries for command clips; older rows have no transcript. Therefore, the manifest cannot confirm whether the actual command audio used the displayed prompt, the exact Option B wording, or a natural paraphrase. Do not describe prompt metadata as an audio transcript.

For the class-agreement benchmark, update the visible alarm prompts to one of the exact Option B variants and preserve all three benchmark strings in the evaluation set. Case-only differences are harmless for speech, but can be normalized in documentation. Keep the recorded waveform and its agreed command/slot label as separate evidence.

## What To Record

1. Prioritize new, distinct LIGHT_OFF, VOLUME_UP, PAUSE, PLAY_MUSIC, COLOR_RED, BRIGHTNESS_100, BRIGHTNESS_20, ALARM_6_00AM, ALARM_8_00AM, LIST_REMINDERS, MESSAGE, and NEXT examples. These are the 12 supported leaves below 95% personal-test F1; several have only 1-5 examples.
2. Add coverage for COLOR_BLUE, COLOR_GREEN, TEMPERATURE_18, TEMPERATURE_22, and TIME because current evidence is sparse or the reference evaluation is weak.
3. For each priority leaf, record each of the three exact Option B phrase variations, with multiple real speakers, distances, and the approved quiet/fan conditions. The existing recording guidance recommends at least 30 distinct clips per weak class across at least three speakers, then reserving a new speaker/session for a genuinely independent evaluation. Do not create duplicate WAVs to inflate support.
4. Keep the agreed overall source mix near 1:1 real/synthetic per variation where licensed and consented data permit; within real clips, use classmate/group voices first, then open-source non-native English, then open-source native English. Do not violate source terms or rebalance fixed test/holdout files.
5. Capture prompt text and a verified transcript separately for every new command recording. Listen to the saved take and correct any wrong slot value before it is labeled. Old command audio cannot be assigned an exact spoken transcript from the current manifest.
6. For OOS rejection testing, collect only reviewed unrelated/near-miss/unsupported-slot speech with explicit labels. Keep wake calls, room noise, and silence separate unless the approved OOS label policy explicitly defines their use.

The 158 gold train rows flagged in-scope but missing a valid slot/variation are **not automatically a recording request**. First ask the dataset owner to inspect and adjudicate them. Correct metadata only when the audio clearly supports the allowed slot; label genuinely unsupported values OOS under the approved policy; re-record only where the audio is unusable or its ground truth cannot be resolved.

## Remaining Work

1. Confirm with the dataset owner that the 10 OOS rows in the fixed holdout are intentional and adjudicate the 158 invalid-slot train rows without silent relabeling or deletion.
2. Resolve the source-license and privacy gates in `docs/classagreementvcm-audit-20261002.md`: SLURP terms, Timers and Such license, Xela download/derivative tracking, classmate consent, and synthetic voice provenance.
3. Approve the 32nd `OUT_OF_SCOPE` model output and pre-register minimum per-leaf support, OOS false-action bound, metric thresholds, and latency target before frozen-test access.
4. Decide whether the three alarm recorder prompts should be changed to exact Option B strings. Preserve all three Option B phrases for the class benchmark.
5. Train an isolated classagreementvcm candidate only after the above gates pass, using gold train with speaker/synthetic-group-disjoint validation. The candidate must not write to the active deployment.
6. Run a paired baseline/candidate comparison on an approved, untouched evaluation set, plus a fresh speaker/session recording test. Current active-model replay alone cannot establish whether classagreementvcm is better.
7. Reconfirm Pi access, microphone, speaker/alarm output, and GPIO hardware. Test a candidate in a separate manual sibling install with rollback; never replace `/home/dalmacio/Desktop/dandan` or the Windows Desktop copy as part of this comparison.

## Artifacts and Protocol

- Derived test manifest: `runs/classagreementvcm-recorded-audio-eval-20261002/recorded_command_test.csv`
- Active-model replay JSON: `runs/classagreementvcm-recorded-audio-eval-20261002/active_intent_replay.json`
- Replay program: `scripts/evaluate_command_audio.py`; inference-only, no GPIO/device actions.
- Model: active `deployment/current_vcm/models/intent_int8.onnx`, SHA-256 `12984401fe312758dbd7ecd53ae9358b0a41b27ffcd024a9ae263614e2e7b977`.
- Dataset: the previously created `runs/me2-vcm-20261001-human936/personal_split_manifest.csv`, test split, mapped command rows only. The `115` clips were already used in the prior personal test report; the separate 27 unmapped/special rows were not treated as commands.
- Gold test/holdout, training, threshold selection, model export and deployment were not part of this replay.
