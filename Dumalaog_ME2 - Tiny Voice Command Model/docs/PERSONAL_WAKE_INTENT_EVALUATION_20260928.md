# ME2 - VCM on Raspberry Pi 5: personal wake and intent evaluation

**Run:** `runs/personalized-vcm-20260928-155109/`  
**Evaluated:** 2026-09-28, Asia/Manila  
**Status:** fresh-trained experimental candidate; not deployed to the Pi and not promoted over the canonical course baseline.

## Selected two-model setup

The evaluated design is two independent classifiers: a two-class `NON_WAKE / WAKE_WORD` model gates a 31-class intent model. Both networks were initialized from scratch (seed 231; no pretrained weights). Commands remain local function labels; the candidate does not transcribe speech or call a cloud speech model.

The broad-negative wake model is preferred over the hard-negative variant. Both gave 4/5 wake clips and zero accepts in the small held-out negative sets, while the broad model uses a less extreme threshold (`0.9992183447` vs. `0.9999421835`). The INT8 intent candidate is personalized with three exact-mapped person-01 examples per batch while retaining the Option B training corpus. Its 39,315-byte model paired with the 37,425-byte wake model totals **76,740 bytes (74.9 KiB)**.

## Data audit and protocol

The personal manifest had 428 WAV rows. Exact decoded-PCM deduplication collapsed 227 repeat rows into 201 unique clips, with no cross-label exact-hash collisions. There is one real speaker, `person-01`; conditions are mostly quiet-near (178 clips), with 18 quiet-far and 5 fan-near unique examples. A deterministic per-label 70/15/15 split holds out 5 unique wake recordings for test and 28 strictly mapped command recordings for test. Clip holdout from one speaker is not unseen-speaker validation.

Only exact mappings to the fixed intent classes were scored. Alarm time, resume, thermostat, and 5/10-minute timer recordings do not have corresponding labels in this 31-class intent model; they remain prediction-only and are excluded from accuracy. No target classes were inferred from the wording.

The 31-class Option B test partition contains 1,798 clips across its existing 100-speaker split. It has been evaluated in prior runs; here it is a regression comparison, **not a new untouched benchmark**.

## Offline results

| Candidate | Evaluation | Result |
|---|---|---:|
| Broad-negative INT8 wake | 5 held-out personal wake clips; five temporal windows per clip, clip accepted if any window crosses the threshold | 4/5 (80%) |
| Broad-negative INT8 wake | Option B commands | 0/1,798 false wakes |
| Broad-negative INT8 wake | Personal held-out non-wake | 0/42 false wakes |
| Personalized INT8 intent | Strict-mapped personal heldout | 24/28 accuracy (85.7%); macro-F1 86.0% |
| Canonical Option B INT8 intent | Same personal heldout | 12/28 accuracy (42.9%); macro-F1 35.4% |
| Personalized INT8 intent | Reused Option B test | 96.11% accuracy; 96.09% macro-F1 |
| Canonical Option B INT8 intent | Reused Option B test | 96.27% accuracy; 96.25% macro-F1 |

The per-intent personal heldout counts are small:

| Intent | Correct / support |
|---|---:|
| CALL | 2/2 |
| LIGHT_OFF | 4/5 |
| LIGHT_ON | 2/2 |
| NEXT | 1/2 |
| PAUSE | 1/2 |
| PLAY_MUSIC | 1/1 |
| TIME | 3/3 |
| WEATHER | 1/2 |
| LIST_REMINDERS | 2/2 |
| TIMER_1m | 3/3 |
| VOLUME_DOWN | 3/3 |
| VOLUME_UP | 1/1 |

The personal result is promising for this speaker and vocabulary, but support ranges from one to five examples per class. The near-perfect per-class rows with tiny support should not be read as stable accuracy estimates.

The following recorded intent labels have no exact target class in this model, so they are deliberately excluded from intent accuracy. These are the model's predictions for their heldout clips, not correct/incorrect scores:

| Recorded label | Clips | Candidate predictions |
|---|---:|---|
| alarm_set | 3 | `ALARM_8_00AM` ×2, `ALARM_9_00PM` ×1 |
| media_resume | 1 | `TIME` ×1 |
| temp_cooler | 2 | `VOLUME_UP` ×2 |
| temp_warmer | 1 | `TIMER_1m` ×1 |
| timer_10min | 2 | `TIMER_1m` ×2 |
| timer_5min | 2 | `TIMER_1m` ×2 |

Wake metrics use the streaming-style five-window maximum per recording. A separate one-centered-window check on the same five clips accepted 3/5; that difference demonstrates how strongly wake outcomes depend on audio-window placement. During live playback through Windows Stereo Mix, the heldout wake clip was accepted once at 0.999345. In the offline centered window it scored 0.998981, below the frozen threshold. That live run is useful integration evidence, not a controlled microphone result.

PC CPU model-only p95 was 0.516 ms (wake) and 0.492 ms (intent). Feature extraction and stream scheduling are excluded; these numbers do not establish the Raspberry Pi's full-path latency.

## Playback and runtime audit

Three personal heldout files were auditioned through the local speaker-to-Stereo-Mix loopback and submitted to the active two-stage VCM:

- `wake_word/ff41417469ab459dbea00ce7c16ae4b5.wav`: wake accepted during the rolling playback run, with the window sensitivity above.
- `lights_off/0a0f78164ebd404d953a63776b72a16a.wav`: wake-gated command returned `LIGHT_OFF` once at 87.8%.
- `question_time/9fd45b6793e94849bedc0bb857e148aa.wav`: wake-gated command returned `TIME` once at 97.7%.

Loopback is a digital audio path, not the physical microphone. A no-save capture probe on the selected Windows microphone-array endpoint showed an active stream but zero input samples/RMS, including while the test audio played. The Fifine endpoint was not present in the device enumeration during that probe. Therefore **the user's real microphone and live speech are not validated**.

The playback audit exposed an intent bug: while a command was still being spoken, the runtime classified an early rolling crop as `STOP` and then classified the completed phrase. The local runtime now waits for 0.35 seconds of speech-end silence before intent inference, and the heldout `LIGHT_OFF` and `TIME` playback tests each produced a single correct event with no `STOP` event. The 10-second post-wake inactivity timeout remains in place. Focused regression suites include endpoint gating and RGB lightbox checks.

The Studio contains a large dedicated RGB lightbox showing simulated color and brightness. Its state is a software device; RGB GPIO hardware is disabled and has not been tested on the Pi.

## Artifacts and integrity

- Executed training/evaluation notebook: `notebooks/ME2_Personalized_Wake_Intent_Retraining.ipynb` (5/5 code cells executed; no stored errors). It presents the stored training histories and final test report. Set `VCM_EXISTING_RUN_DIR` to an empty/unset value in its configuration cell before a deliberate fresh retraining run.
- Full metrics/provenance: `runs/personalized-vcm-20260928-155109/final_evaluation.json` and `dataset_audit.json`.
- Pi review staging area: `deployment/personalized_candidate/` (two INT8 ONNX models, portable wake `export_summary.json`, checksums and limitations in `metadata.json`). It is **not a complete application bundle and has not been deployed**.
- Selected ONNX SHA-256: wake `5e74fade5beefdd86267823ea0c895fcd3a512fc04ce87b06b5d9a4e4b5dad3d`; intent `c91c73860b24eea13b74620970feb18c1cf531629de727d23ce55361d8196a4`.
- The staging pair was loaded through `VCMPredictor`; a fresh heldout replay matched 24/28 mapped intent clips and 4/5 wake clips when using the same five-window maximum, with 0/39 personal command false wakes and 0/3 personal non-command false wakes in this smaller single-window staging probe.

## Remote audition

`remote_vcm_eval.py` is an authenticated, loopback-only prediction page. It captures a two-second wake sample and evaluates five placements (matching the offline threshold protocol), then a 2.5-second command sample after wake. Audio is accepted only after a button press, held in memory for inference, not persisted, and there is no appliance-action endpoint. `scripts/start_remote_vcm.py` prompts locally for a username and masked password (no password is saved), starts the evaluator on `127.0.0.1:7870`, and opens a temporary Cloudflare Quick Tunnel. The user has since put public/network deployment **on hold**; do not install `cloudflared` or run the tunnel until the user reopens that scope. No public URL is active. Quick Tunnels produce temporary random URLs intended for tests; they do not provide a stable production address. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/). Authentication and route behavior are covered by `tests/test_remote_vcm_eval.py`.

## Next data and checks

1. Reconnect/select a working physical microphone and run at least three deliberate wake trials. Confirm input RMS is nonzero before interpreting misses.
2. Add two real speakers minimum, with wake/non-wake near-misses, fan/room noise, far-field and ordinary command recordings. Keep people and recording sessions split across training and evaluation; report false accepts per hour in continuous audio.
3. Record additional personal examples for `PLAY_MUSIC`, `VOLUME_UP`, `LIGHT_ON`, `CALL`, and underperforming `NEXT`, `PAUSE`, `WEATHER`, and `LIGHT_OFF`. Gather every intended label before deciding whether to expand the vocabulary.
4. After model promotion criteria are met, integrate the pair into a fresh Pi application bundle; test full frontend+model latency, memory and capture on Raspberry Pi 5. The Pi is currently powered down and no deployment occurred in this run.
5. Public/network deployment is on hold by the user's request; resume only if the user explicitly reopens it.
