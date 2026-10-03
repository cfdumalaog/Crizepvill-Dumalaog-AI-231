# classagreementvcm Implementation Plan

**Version:** classagreementvcm<br>
**Plan date:** 2026-10-02 (Asia/Manila)<br>
**Status:** Planning only. No training, test execution, model promotion, Pi installation, or change to the active model was performed for this plan.

> **Dataset update - 2026-10-02:** The current snapshot is `data/ai231-me2-voice-commands-hf-a90b8d10`, revision `a90b8d106349b02c5570a1a258503386043f63b2`. Read `docs/classagreementvcm-repull-audit-20261002.md` first. It supersedes the initial row interpretation and holdout-confirmation gate below: the 158 rows are intentional in-scope intents with unsupported slot values, and the owner confirmed OOS additions to test/holdout. The latest 13-file snapshot and all 81,686 WAVs passed integrity checks. Train, test and numeral files are unchanged from the earlier linked revision; 59 holdout entries were replaced. Source permissions and the model's reject/unsupported-slot policy remain unresolved, so training is still gated. The older review iterations below are historical; read the updated decisions in Sections 1, 4, 6, 7, 10 and 13 as current.

**Purpose:** Implement the class-agreed command vocabulary and the collated gold dataset as a separately versioned candidate, validate the full Raspberry Pi behavior, and preserve the present working release until a later explicit promotion decision.

## 1. Executive Decision

Treat the latest verified Hugging Face snapshot of the ME2 Spoken Command Dataset as the source of truth for this candidate. Its schema follows the class-agreed command taxonomy: 19 top-level intents, 31 permitted command/slot combinations, and three benchmark phrases for each combination (93 phrase rows). Train only from the frozen train partition, with a speaker/synthetic-voice-disjoint validation subset carved from train. Keep the published test frozen for one final offline comparison and use holdout only for the approved live Raspberry Pi demonstration. Keep number-only clips separate.

The existing intent model has 31 outputs; a full exact crosswalk now confirms these correspond to the 31 permitted command/slot combinations. The dataset contains explicit out-of-scope (OOS) rows and a separate category of 158 in-scope commands with non-schema slot values. Benchmark scoring accepts either an explicit OOS result or no response as reject, so a 32nd output is a design choice rather than a requirement. Approve how the candidate handles true OOS and unsupported slot values before training. Keep the binary wake model byte-identical for the first candidate; it remains separate from command recognition.

Train, export, package, test, and if approved manually run classagreementvcm beside the active installation. Never let the ordinary training/sync path write to deployment/current_vcm, the Windows Desktop launcher package, or /home/dalmacio/Desktop/dandan during candidate work. No automatic promotion is in scope. Replacement of the active model is a distinct future decision requiring measured acceptance results and explicit user/team approval.

## 2. Scope and Guardrails

### Included

- Audit the pinned master dataset, all source/license metadata, schema mapping, audio integrity, deduplication, and fixed split boundaries.
- Train and evaluate a separately named candidate command model with stable class order, OOS handling, and an auditable run manifest.
- Preserve and reuse the current wake model initially; evaluate wake gating against the agreed gold evaluation material without fitting the wake model on test or holdout.
- Implement and test each of the 19 commands and their permitted values in the current Raspberry Pi application architecture.
- Add regression, model parity, action/API, physical-device, live-demo, packaging, rollback, and handoff evidence.
- Prepare a candidate-only Raspberry Pi installation in a sibling directory after local gates pass and Pi access is available.

### Explicitly excluded unless separately approved

- Replacing or overwriting the active intent model, wake model, Windows Desktop deployment package, or /home/dalmacio/Desktop/dandan installation.
- Boot autostart, public network binding, or Internet-exposed control endpoints.
- Using gold test, gold holdout, or any test result to fit weights, select thresholds, tune preprocessing, select epochs, or choose among checkpoints.
- Training on gold numerals as command examples. The numerals bucket remains separate until a separately scoped slot-extraction task is approved.
- Sending real phone calls or SMS/messages; controlling HVAC; creating timed reminders when the schema provides no reminder-time slot.
- Treating conversational phrases, open-source source files, or Drive archives as licensed for redistribution without verifying their terms.
- Claiming live speech, alarms, physical audio, or GPIO work from simulation-only checks.

### Non-negotiable invariants

1. Capture the current active model hashes, configuration, release receipt, and current user changes before any implementation work.
2. All classagreementvcm training and packaging outputs are written to a new run/release namespace; candidate-only mode is the default.
3. The current two-model active pair and its rollback archive remain preserved and usable throughout.
4. Never commit raw classmate audio, protected manifests, credentials, or restricted-source audio to a public repository.
5. A prepared package is not a deployment; a candidate running beside the active app is not a promotion.
6. Keep the current Windows launcher/package at <code>C:\Users\danda\Desktop\dandan</code> and the active Pi installation at <code>/home/dalmacio/Desktop/dandan</code>; keep their launch paths synchronized only when an explicitly approved active deployment changes location.
7. Treat <code>C:\Users\danda\Desktop\Kiko</code> and <code>/home/dalmacio/Desktop/kiko</code> as user-owned protected data. Do not inspect, modify, move, rename, or delete either path for this work.

## 3. Evidence Baseline and Current Alignment

The observations below come from the checked-in current source, tests, local gold snapshot and handoff records. They are planning evidence, not claims that this request reran those tests or rechecked the currently unreachable Pi.

| Area | Observed current state | classagreementvcm requirement |
|---|---|---|
| Intent model | TinyDSCNN-48, 31 fixed command/slot outputs; live confidence and margin gates are 0.68 and 0.15. | Preserve the architecture/frontend initially, map the 31 outputs exactly, and choose an OOS/unsupported-slot strategy with validation-only rejection calibration; an explicit OOS output is optional. |
| Wake model | Separate binary wake model, already deployed as part of the active pair. | Reuse its exact bytes and record its hash in the candidate manifest. Do not silently retrain it. |
| Frontend | 16 kHz mono, 2.5 second input, 40-band log-mel, feature shape 1x40x251. | Keep identical for a controlled comparison unless a validation-only experiment is separately recorded and frozen before test. |
| Training source | Current loader points to the older project command corpus and personal recordings. | The verified gold Parquet is not currently the loader source. Add a dedicated audited gold loader; do not silently blend old manifests. |
| Current partitions | Current legacy manifest has train/val/test partitions; reference test has been reused in earlier evaluations. | Do not call that old test a fresh final test. Use the fixed gold test for one pre-registered final comparison only. |
| Personal audio | Existing saved human rows are not speaker-independent across their partitions. | Treat as supplemental only after identity, exact source inclusion, consent, audio hash, transcript/label, and leakage audits. The first clean candidate should use gold train only. |
| LIGHT_ON | Current action resets white and brightness to 100 percent, restoring all RGB channels after off or a color. Existing RGB regression coverage and an earlier Pi pin check are recorded. | Preserve this behavior and add candidate-path API/UI and physical regression cases. |
| Music | The current MediaEngine has local demo tracks and optional external sources; action handling updates player state and returns track metadata. | Make the demo playlist self-contained/offline, and prove audible play, pause/resume position, next, stop, and volume on the target Pi. |
| Alarms | The three agreed alarm labels are recognized, but runtime stores one evaluated alarm pointer while also displaying a list. There is no comprehensive alarm suite; the buzzer is optional/unset. | Implement a real scheduler for multiple listed one-shot alarms, using Asia/Manila time, with deterministic clock tests and confirmed audible output. |
| Timers | 10-second, 30-second and 1-minute values set a monotonic deadline; new commands replace the single deadline. | Specify replacement feedback and test all durations, expiry, simultaneous timer/alarm events, and actual notification output. |
| Reminders | Create command returns an acknowledgement; list command returns a hardcoded list. | Implement a truthful local list limited to the three supported reminder items. No due-time alerts without a time slot. |
| Other actions | Calls/messages are demo responses; temperatures are a software simulation; weather is optional and has an offline fallback. | Keep these boundaries truthful and side-effect free; test all offline behavior. |
| Current release | Active pair is preserved in /home/dalmacio/Desktop/dandan; boot autostart and public access are off. Latest attempted reporting package was pending and Pi connectivity/capture were unverified in the latest recorded check. | Recheck access when implementation starts; candidate install is sibling/manual only. Do not use the standard auto-sync path without a candidate-only guard. |

## 4. Source-of-Truth Data and Access Requirements

### 4.1 Dataset selection

Primary source: the pinned local snapshot at:

<code>AI 231/Dumalaog_ME2 - Tiny Voice Command Model/data/ai231-me2-voice-commands-hf-a90b8d10</code>

Hugging Face dataset: [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)<br>
Pinned revision: <code>a90b8d106349b02c5570a1a258503386043f63b2</code><br>
Class gold Drive folder: [ai231-me2-gold-dataset](https://drive.google.com/drive/folders/1k2RklcsU8ull9-4fagl-8zTfecPaQWBZ)<br>
Dataset schema spreadsheet: [Dataset Schema](https://docs.google.com/spreadsheets/d/1HZ7Q58S-konS5XWUE6FUGC-S-C8qAoSPaVbIP4NYSU0/edit?usp=sharing)<br>
Meeting notes identify Ma'am Ailene's separate collected dataset at [AI231-ME2-Voice-Data](https://drive.google.com/drive/folders/1_GcDuvaRnGlkdtGdoogQI7FflgWf6gF8?usp=share_link); it is a source in the collation story, not itself the master gold folder.
Dataset documentation artifact: [ME2 Master Dataset Audit](https://claude.ai/artifact/PPNHMWd5rx9qcXTdcukV7s).

The complete local Hugging Face snapshot is the usable full corpus. The authenticated Drive ZIP is retained for provenance but was previously found incomplete in its audio coverage; do not mix it into the training source or use it to fill holes unless the manifest-to-audio audit proves exact content identity. Preserve the Drive discrepancy as an open data-owner question, not a reason to alter the Hugging Face splits.

### 4.2 Pinned snapshot inventory

| Dataset partition | Expected rows | Intended use |
|---|---:|---|
| train | 10,682 | Model fitting; validation group split is carved only from this partition. |
| test | 4,418 | Frozen offline final comparison after every model and threshold choice is locked. |
| holdout | 196 | Live Raspberry Pi demonstration/acceptance only; never fitting, early stopping, threshold selection, or repeated model selection. |
| numerals | 66,390 | Separate number-only clips, not a command class and not part of train/valid/test/holdout. |
| variations.csv | 93 | Authoritative phrase/label reference: 31 class/value pairs times 3 phrases. |

The latest snapshot contains 81,686 rows in total and 10 Parquet shards. Its audio is documented as 16 kHz, mono, 16-bit PCM WAV. All 13 files and all audio passed a full local/Hub integrity scan. `variations.csv` and the train, test, and numeral shards match the earlier revision byte-for-byte; the README and holdout shard changed. The new holdout has 186 in-scope clips and 10 true OOS clips; 59 paths changed (56 in-scope real-voice clips and 3 OOS clips). Keep the latest 196-row holdout frozen for live demonstration only. See the repull audit for hashes and exact comparison.

### 4.3 Dataset gating and requirements

Before feature extraction or fitting:

1. Verify the snapshot revision and every Parquet/shard SHA-256 against the pinned manifest/Hub metadata; verify variations.csv and README hashes too.
2. Verify every manifest path resolves inside the selected snapshot directory, every referenced file decodes, and each WAV has one channel, 16 kHz rate, PCM representation, finite nonempty samples, and plausible duration. Record corrupt, silent, truncated, and clipped rows; do not quietly drop them.
3. Reconcile row totals and labels against the tables above. Validate every in-scope transcript against its manifest command, permitted variation/value, and phrase annotation. Retain mismatches in an audit report and quarantine only with an explicit rule.
4. Preserve all provenance and applicable license/consent. Create a machine-readable source/license inventory with fields: source name, upstream URL, license/terms, attribution, research/education restriction, consent status for classmate recordings, redistribution status, and disposition. Block use of any source whose use for model training is not authorized.
5. Never infer an exact human transcript from a prompt field. The gold README says some training transcripts come from prompts/TTS labels and some checks are not human verified; carry transcript_source, variation_match and whisper_check into the audit.
6. Use speaker_id and synthetic voice ID as grouping keys. Test that no identity is shared across the published train/test/holdout partitions and none crosses the new train/validation split. Check raw-audio SHA-256 and normalized-PCM hashes across every split and all supplemental corpora.
7. Exact-duplicate audio may occur under different filenames/transcripts. Deduplicate only with an explicit deterministic rule and preserve a crosswalk to all original rows; any same-audio/different-label collision is a hard stop for that row pending adjudication.
8. Do not re-add any source files already present in the gold snapshot. For extras, match path, exact audio hash, speaker/source/license, and label. Add them only to gold train (never gold test or holdout), and document why the gold master did not already contain them. Recommended first run excludes all supplemental extras to preserve a clean, reproducible master-only baseline.
9. Preserve the agreed per-variation source mix: target approximately 1:1 synthetic-to-real audio where the gold snapshot has licensed/consented coverage. For the real-audio portion, prioritize classmate/group voices, then open-source non-native-English clips, then open-source native-English clips. The gold README implements this as real audio first up to half, with the group's synthetic set filling the rest. Audit actual ratios and exceptions by variation, split, source and accent group; do not move or rebalance fixed test/holdout clips to force a ratio.
10. Numerals remain excluded from command classification. For slotted labels, only the exact values in Section 5 are supported outputs. The 158 train examples with a valid top-level command and another slot value remain an explicit unsupported-slot category (`out_of_scope=0`); decide whether they inform safe rejection or another approved slot-handling design. Do not relabel them as true OOS, force them to a nearest slot, or silently discard them.
11. Keep bulk audio and generated features out of public Git. Use the existing root shared .venv; stream/chunk Parquet audio and cache only to an ignored local run directory. Plan for enough local disk to retain the immutable source, logs, feature cache, and candidate bundle without deleting user data.

### 4.4 Data-access checklist

- [ ] Local complete Hugging Face snapshot readable; pinned revision and hashes verified.
- [ ] Gold manifests, approved schema, and meeting notes readable; any schema discrepancy referred to the data owner.
- [ ] Source/license/consent recorded for every dataset contributor and external corpus.
- [ ] DGX path, access permissions, quota, and transfer integrity confirmed only if the team elects DGX; DGX is optional for the first PC training run.
- [ ] Raspberry Pi SSH/network authorization confirmed; never store passwords/tokens in this plan, handoff, source, or logs.
- [ ] Raspberry Pi 5 architecture, free storage, Python/runtime packages, GPIO backend, ALSA, and candidate port confirmed before install.
- [ ] 16 kHz microphone, speaker/alarm output, RGB GPIO circuit, and safe manual access physically available.
- [ ] Offline-only smoke tests possible even if weather networking and external music services are unavailable.

## 5. Exact Command Taxonomy and Phrases

The 19 commands below each have three schema phrases. Slot values are categorical and finite. The mapping is the intended target mapping based on the current class names, but implementation must verify the schema row-by-row rather than rely on equal class counts. The top-level command is the reporting hierarchy; the classifier target is one of 31 exact allowed command/value combinations.

### ALARM (slot: time)

- <code>ALARM_6_00AM</code>, value <code>6:00 AM</code>: "Alarm 6:00 AM"; "Wake me up at 6:00 AM"; "Set an alarm for 6:00 AM".
- <code>ALARM_8_00AM</code>, value <code>8:00 AM</code>: "Alarm 8:00 AM"; "Wake me up at 8:00 AM"; "Set an alarm for 8:00 AM".
- <code>ALARM_9_00PM</code>, value <code>9:00 PM</code>: "Alarm 9:00 PM"; "Wake me up at 9:00 PM"; "Set an alarm for 9:00 PM".

### BRIGHTNESS (slot: percent)

- <code>BRIGHTNESS_20</code>, value <code>20 percent</code>: "Brightness 20 percent"; "Adjust brightness to 20 percent"; "Brightness level 20 percent".
- <code>BRIGHTNESS_60</code>, value <code>60 percent</code>: "Brightness 60 percent"; "Adjust brightness to 60 percent"; "Brightness level 60 percent".
- <code>BRIGHTNESS_100</code>, value <code>100 percent</code>: "Brightness 100 percent"; "Adjust brightness to 100 percent"; "Brightness level 100 percent".

### CALL (no slot)

- <code>CALL</code>: "Call"; "Make a call"; "Make a phone call".

### COLOR (slot: color)

- <code>COLOR_BLUE</code>, value <code>Blue</code>: "Change color to Blue"; "Switch color to Blue"; "Set color to Blue".
- <code>COLOR_GREEN</code>, value <code>Green</code>: "Change color to Green"; "Switch color to Green"; "Set color to Green".
- <code>COLOR_RED</code>, value <code>Red</code>: "Change color to Red"; "Switch color to Red"; "Set color to Red".

### CREATE_REMINDER (slot: reminder item)

- <code>CREATE_REMINDER_DRINK_WATER</code>, value <code>Drink water</code>: "Reminder Drink water"; "Remind me to Drink water"; "Create a reminder to Drink water".
- <code>CREATE_REMINDER_EXERCISE</code>, value <code>Exercise</code>: "Reminder Exercise"; "Remind me to Exercise"; "Create a reminder to Exercise".
- <code>CREATE_REMINDER_STUDY</code>, value <code>Study</code>: "Reminder Study"; "Remind me to Study"; "Create a reminder to Study".

### LIGHT_OFF (no slot)

- <code>LIGHT_OFF</code>: "Lights out"; "Kill the lights"; "Shut off the lights".

### LIGHT_ON (no slot)

- <code>LIGHT_ON</code>: "Lights on"; "Power on the lights"; "Turn on the lights".

### LIST_REMINDERS (no slot)

- <code>LIST_REMINDERS</code>: "Reminders"; "Show my reminders"; "List my reminders".

### MESSAGE (no slot)

- <code>MESSAGE</code>: "Message"; "Send a message"; "Send my message".

### NEXT (no slot)

- <code>NEXT</code>: "Next song"; "Skip song"; "Play next song".

### PAUSE (no slot)

- <code>PAUSE</code>: "Pause"; "Pause audio"; "Pause song".

### PLAY_MUSIC (no slot)

- <code>PLAY_MUSIC</code>: "Play music"; "Start music"; "Play some music".

### STOP (no slot)

- <code>STOP</code>: "Stop"; "Stop playing"; "End playback".

### TEMPERATURE (slot: degrees Celsius)

- <code>TEMPERATURE_18</code>, value <code>18 degrees</code>: "Temperature 18 degrees"; "Change the temperature to 18 degrees"; "Set the temperature to 18 degrees".
- <code>TEMPERATURE_22</code>, value <code>22 degrees</code>: "Temperature 22 degrees"; "Change the temperature to 22 degrees"; "Set the temperature to 22 degrees".
- <code>TEMPERATURE_26</code>, value <code>26 degrees</code>: "Temperature 26 degrees"; "Change the temperature to 26 degrees"; "Set the temperature to 26 degrees".

### TIME (no slot)

- <code>TIME</code>: "Time"; "What time is it?"; "Tell me the time".

### TIMER (slot: duration)

- <code>TIMER_10s</code>, value <code>10 seconds</code>: "Timer 10 seconds"; "Countdown for 10 seconds"; "Start a timer for 10 seconds".
- <code>TIMER_30s</code>, value <code>30 seconds</code>: "Timer 30 seconds"; "Countdown for 30 seconds"; "Start a timer for 30 seconds".
- <code>TIMER_1m</code>, value <code>1 minute</code>: "Timer 1 minute"; "Countdown for 1 minute"; "Start a timer for 1 minute".

### VOLUME_DOWN (no slot)

- <code>VOLUME_DOWN</code>: "Volume down"; "Lower the volume"; "Turn the volume down".

### VOLUME_UP (no slot)

- <code>VOLUME_UP</code>: "Volume up"; "Increase the volume"; "Turn the volume up".

### WEATHER (no slot)

- <code>WEATHER</code>: "Weather"; "What's the weather?"; "Tell me the weather".

### Label policy

- Top-level intents: exactly the 19 names above.
- In-scope classifier leaves: exactly the 31 current command/slot labels above. No open-ended numbers, colors, alarm times, or reminder strings are valid leaves.
- Rejection: true OOS is distinct from an in-scope intent with a non-schema slot. Benchmark scoring treats explicit OOS or no response as reject. Select a candidate strategy for true OOS and unsupported slot values before training; no benchmark rule requires an explicit OOS output neuron.
- Stable order: write an immutable <code>class_order.json</code> in each run and release for the 31 approved command/value outputs. If the approved candidate adds an OOS output, version and test that contract explicitly. Assert exact label-to-index equality in training, ONNX export, app metadata, and UI/API. Never sort labels independently at load time.
- Unknown or unsupported predictions must produce no action. Calibrate confidence/margin or other reject logic only on validation. Do not assume the current 0.68/0.15 gates transfer to a new candidate.

## 6. Split, Leakage, and Reproducibility Protocol

### Fixed source partitions

- The gold test remains exactly the published 4,418 rows and the holdout exactly the published 196 rows. No resampling, moving rows, or balancing by copying clips.
- The gold training source is the 10,682-row published train partition. Derive candidate validation from this source only, not by relabeling or resampling published test/holdout.
- Build a deterministic speaker-group validation split, initially target 15-20 percent of eligible train speaker groups subject to all supported classes and source groups having useful validation coverage. Tune the exact fraction once, before training, based on a coverage report. The actual split is group-based, not random clip-based. Group real speakers and synthetic voices independently by stable voice ID, so no voice can occur in both train and validation.
- Write a frozen split manifest with original row IDs, partition, stable grouping ID hash, audio hashes, target, seed, builder version, and source snapshot hash. Re-running the same builder must produce identical bytes or a documented canonical JSON equivalent.
- If a command/slot leaf is missing from a partition after strict speaker grouping, stop and revise only the train/validation allocation. Never borrow from test/holdout to fill it. Report unsupported leaves honestly.
- Train only on gold train rows allocated to train. Validation may be used for checkpoint/epoch, hyperparameter, and reject-threshold selection. It may not be merged back into the fitting rows after model selection for this candidate.

### Holdout composition and freeze rule

The latest holdout has 196 rows: 93 supported variation groups with two clips each (186 in-scope) plus 10 true OOS examples. The group owner stated that OOS clips were added to test and holdout and asked members to re-pull; the linked benchmark README independently describes the 196-row holdout as 186 commands plus 10 OOS clips. The confirmation gate is closed. The latest 196-row inventory is now the frozen version for the approved live Raspberry Pi demonstration. Do not drop, move, relabel, use for fitting, select thresholds from, or repeatedly evaluate these clips. Report the 186 commands and 10 OOS rows separately.

### Test exposure policy

Before final test inference, freeze and hash: training code/config; frontend; class order; checkpoint selection; quantization; wake model hash and wake thresholds; command confidence/margin rejection; output/action mapping; and test harness. Perform exactly one registered inference pass for each selected active baseline/candidate model on the frozen gold test. Report every run and do not conceal a failed attempt. If a defect requires a code/model/threshold change after test access, mark the test exposed; obtain a new independent approved test before making a fresh confirmatory claim. Never move test examples into train.

## 7. Model and Training Plan

### Candidate architecture

1. Use the current 16 kHz mono, 2.5-second, 40-mel frontend and TinyDSCNN-48 intent architecture as the controlled first candidate, retaining the 31 supported command/value outputs unless a separate approved reject output is chosen. Record the exact frontend code hash and configuration.
2. Reuse the active binary wake model exactly (record SHA-256, input signature, threshold and provenance); do not retrain it in the first pass. Intent fitting is candidate-only. Gold test/OOS may be scored as wake negatives without updating the wake model or its threshold.
3. Choose a true-OOS rejection strategy before training. An explicit output and a validation-calibrated no-response/reject gate are both valid candidates under the benchmark; measure OOS false actions separately. Define unsupported-slot behavior separately and do not relabel those 158 rows as true OOS without approval.
4. Keep exact preprocessing, normalization, windowing, padding/truncation, label ordering, quantization, and confidence semantics in run metadata and exported model metadata. Ensure WAV transcript text is not fed into inference.

### Training sequence

1. Snapshot current source changes, active ONNX hashes, run metadata, thresholds, deployment receipts, and baseline action test results. Do not alter active files.
2. Create a new run directory named with classagreementvcm and timestamp/revision. The output path must be distinct from all prior runs and deployment/current_vcm.
3. Run the data audit, class-order crosswalk, license check, waveform integrity, dedupe and speaker/synthetic leakage checks. Fail closed on unmapped labels, unresolved hash collisions, or missing source authorization.
4. Produce and review the deterministic speaker-group train/validation manifest. Have a second reviewer approve split counts and class support before fitting.
5. Train from fresh initialization with the established local recipe/optimizer and documented seed. Use class weights only if imbalance is measured in gold train and the weighting formula is predeclared. Log all epochs, learning rates, losses, per-leaf validation metrics, confusion matrix, seed, package versions, hardware, and wall times.
6. Choose epoch/hyperparameters/checkpoint and validation-only rejection thresholds using a predeclared selection rule. Avoid repeated informal reruns against test/holdout. Store the winning checkpoint and decision report.
7. Export float model and INT8 model into the run directory. Validate input/output shapes, finite outputs, fixed class order, float/INT8 parity, representative validation quantization, and inference determinism. Use only train/validation data for quantization calibration.
8. Build a candidate pair containing the exact existing wake model plus the new intent model. Verify per-file hashes and exact load-time API metadata. Record that the wake file is inherited, not newly trained.
9. Benchmark Windows shared-environment CPU/GPU as available. Benchmark ARM64 Pi frontend+model and live pipeline separately once Pi access is restored. No PC timing may be called a Pi result.
10. Generate a signed/hashed classagreementvcm release manifest and test report. Keep all outputs separate from active deployment. End the first cycle at candidate-validated/not-promoted.

### Candidate comparison and measures

After every choice is frozen, run a paired, one-time inference on the latest gold test using the active 31-output model and the candidate with its approved reject strategy. For the active baseline, measure valid-command accuracy and false execution of OOS examples, without pretending the 31-way model had an explicit OOS output. Report:

- exact command+slot accuracy, per-leaf precision/recall/F1 and support;
- macro F1 over the 31 in-scope leaves, top-level 19-intent metrics after aggregating leaves, and OOS precision/recall/false-action rate;
- phrase-variation 1/2/3, speaker, real/synthetic, accent group, source, duration, transcript-source, and command-value slices when sample support permits;
- confusion matrix including OOS, confidence/reject coverage, wake false accepts on all allowed negative test examples, and wake detection on authorized positives if present;
- speaker-cluster bootstrap confidence intervals, not just clip-level standard errors;
- model size, frontend+model latency, peak memory, dropped audio, response/action success, and runtime crash/restart evidence.

The existing project goal of at least 95 percent F1 for each command/slot leaf is an acceptance goal, not a current result. Proposed gates for team approval: deterministic unit/action tests 100 percent; each adequately supported in-scope test leaf at least 95 percent F1 before promotion; OOS no-action correctness 100 percent in deterministic tests and a pre-agreed low false-execution rate in the fixed test; no class below the agreed minimum test support may be claimed as passing; ARM64 frontend+model p95 below 10 ms under repeated warmed inference; complete pair below 500 kB (the current pair is about 76.7 kB); and no regression in wake behavior beyond an explicitly accepted statistical margin. These are proposed acceptance thresholds and must be ratified before final test access. Do not adjust gates after seeing test scores.

## 8. Raspberry Pi Command and State Requirements

All recognized commands must use the existing wake gate, show a truthful result in the app/API, and produce a deterministic state transition. A rejection/OOS prediction must not call a device action. Any physical action failure must be returned as a failure, not a success acknowledgement.

| Intent/value | Required behavior in classagreementvcm | Current gap or boundary |
|---|---|---|
| LIGHT_ON | Turn all three RGB channels on as white at 100 percent. This must restore all channels after LIGHT_OFF, after any selected color, and after a restart if restart policy restores a known safe default. | Core execute behavior and RGB regression already exist; repeat candidate API, UI and physical checks. |
| LIGHT_OFF | Set all channels to zero; report off only after GPIO route accepts the change. | Existing behavior; confirm physical pins. |
| COLOR_RED/GREEN/BLUE | Select exactly the requested channel. If currently off, use the defined on behavior (current behavior turns it on at 100 percent); list this behavior in the UI result. | Existing simulated behavior; repeat each sequence on Pi. |
| BRIGHTNESS_20/60/100 | Set exact brightness while retaining selected color; when white, scale all three channels. Boundary is exactly 0-100 percent. | Existing values; test repeated idempotence and color interaction. |
| PLAY_MUSIC | Start/resume the current bundled local playlist. Offline use must work. If paused, continue the same track at preserved position; if stopped, start the current/first track. | Existing state/MediaEngine exists, but actual playback end-to-end coverage was not found. |
| PAUSE | Pause audible output without losing track or playback position; a subsequent PLAY_MUSIC resumes. | Add a real audio backend contract and device test. |
| NEXT | Advance one track and immediately play it; wrap deterministically at playlist end. Update the displayed current track. | Existing metadata next route; verify audible stream changes. |
| STOP | Stop playback and reset position according to the declared policy; next PLAY_MUSIC starts predictably. | Verify distinct semantics from PAUSE. |
| VOLUME_UP/DOWN | Change system/player volume by a fixed documented step (current route is 10 percentage points), clamp to 0-100, report actual level and handle unavailable device. | Existing ALSA route/test exists; add boundary and candidate media-output verification. |
| ALARM_6_00AM | Schedule one one-shot for the next local 6:00 AM in Asia/Manila. | Current label exists; system timezone is not explicit. |
| ALARM_8_00AM | Schedule one one-shot for the next local 8:00 AM in Asia/Manila. | Same. |
| ALARM_9_00PM | Schedule one one-shot for the next local 9:00 PM in Asia/Manila. | Same. |
| All ALARM values | Support multiple distinct active alarms, deduplicate the exact same due time, list each alarm in the app, fire once, sound a real alarm output, and allow UI/button dismiss and 5-minute snooze. Due alarms must not vanish because a later alarm replaced a single pointer. Recommend one-shot rather than daily repeat because the command schema does not express recurrence. | Current single evaluated alarm pointer conflicts with displayed list; optional buzzer is not proof of audible alarm. Persistence across restart is an approval question; default recommendation is persist scheduled alarms locally and restore safely. |
| TIMER_10s/30s/1m | Start the exact duration using monotonic time; display remaining time; on expiry notify visibly and audibly once. There is one timer. A new timer replaces it and the response says the old timer was replaced. | Existing one-deadline logic; add fake-clock tests and physical output checks. |
| CREATE_REMINDER_DRINK_WATER/EXERCISE/STUDY | Add the exact item to a local reminders list. If already present, behave idempotently or report duplicate; do not claim a timed alert. | Current acknowledgement does not store state. |
| LIST_REMINDERS | Return the current actual list in stable order, including empty-list behavior. | Current response is hardcoded rather than state-backed. |
| CALL | Return “demo only / no call placed.” Never place a call. | Current safe demo behavior. |
| MESSAGE | Return “demo only / no message sent.” Never send a message. | Current safe demo behavior. |
| TEMPERATURE_18/22/26 | Update a clearly labeled software demo target in Celsius; state that no HVAC is connected. | Current simulation only; never imply physical climate control. |
| TIME | Speak/display current Asia/Manila time using timezone-aware code. | Current system-local datetime use needs explicit timezone. |
| WEATHER | Try the existing optional live provider only when explicitly enabled and online; otherwise return a clear offline/unavailable message. No API key is required by the current Open-Meteo route. | Test network errors, malformed provider response and offline fallback. |
| OUT_OF_SCOPE / REJECT | Do not mutate lights, volume, music, timers, alarms, reminders, calls, messages or thermostat state. Show/say a concise retry message and log a non-sensitive reason. | New output/route required. |

### State and failure semantics to define before coding

- State changes should be idempotent where meaningful: LIGHT_ON twice leaves white/100; LIGHT_OFF twice stays off; PAUSE twice stays paused; STOP twice stays stopped; alarm duplicate produces one alarm.
- One active timer; multiple active alarms. A timer replacement is explicit in the response and UI. Alarm and timer expiry events can coincide without one suppressing the other.
- Alarms should have stable IDs, due timestamp with timezone, created timestamp, enabled/ringing/dismissed/snoozed state, and auditable schedule history. The app must surface which alarm is ringing. Snooze creates a new one-shot due time five minutes later; dismissal consumes only the ringing alarm.
- Before coding, choose whether alarm/reminder state survives a process restart, storage format and corruption handling. Recommended: alarms persist atomically in a local versioned JSON/SQLite store; in-memory reminders may remain session-only unless the team explicitly needs durable reminders. Do not persist test or user data outside the expected Pi home directory.
- Hardware/media errors must not update software state optimistically if the operation did not occur. Keep a simulated mode explicit and visually distinct from physical mode.

## 9. Test Plan and Evidence Matrix

Create a test log for every test ID with: date/time, tester, commit/run/release hash, device/hardware, exact command/input, expected result, observed result, PASS/FAIL/BLOCKED/NOT RUN, evidence file/link, and defect ID. Keep test reports versioned but keep private audio out of Git. Do not convert NOT RUN into PASS.

### A. Dataset and access tests

| ID | Test | Passing evidence |
|---|---|---|
| DATA-01 | Pin/revision and shard integrity | Revision equals the documented SHA; every expected shard and manifest hash matches; discrepancies stop the run. |
| DATA-02 | Row/schema/count audit | Exact 10,682/4,418/196/66,390 partition counts; schema columns/types; 93 variations; totals 81,686. |
| DATA-03 | Audio decode/rate/channel/duration | Every included row decodes; rate/channel/encoding checked; exceptions quarantined with row IDs, not silently skipped. |
| DATA-04 | Label and slot map | The 93 supported variation rows map to 31 leaves; true OOS and in-scope non-schema slot rows remain distinct and are counted correctly. |
| DATA-05 | Phrase coverage | variations.csv has 31 label/value groups and 3 exact phrases each; all 93 rows match the approved schema sheet/documentation. |
| DATA-06 | Split identity leakage | Speaker ID and synthetic voice groups are disjoint across official splits and carved train/validation. |
| DATA-07 | Content leakage | Raw and normalized PCM hashes do not cross model-fitting, validation, test, or holdout. Duplicate collisions have owner-reviewed disposition. |
| DATA-08 | Source/license/consent | Every source has training-use disposition and required attribution/retention; unauthorized material is excluded. |
| DATA-09 | Numeral exclusion | All 66,390 number-only records remain outside command fitting/evaluation; no number-only row appears in the class loader. |
| DATA-10 | Holdout reconciliation | Latest 196-row holdout is hash-pinned and reported as 186 supported commands plus 10 OOS; owner confirmation is recorded in the repull audit. |
| DATA-11 | Extra-data dedupe | Each supplemental item is proved absent from gold by content/source audit or excluded. |
| DATA-12 | Source balance and priority | Per-variation real/synthetic mix and source/accent counts are reported; the approximate 1:1 target and classmate -> non-native open source -> native open source real-audio priority are verified where data availability/licensing allow; fixed test/holdout contain eligible real and synthetic voices absent from training where available; fixed evaluation partitions are not altered. |
| ACCESS-01 | Required local/HF access | Full pinned snapshot is readable without altering source; insufficient permissions/storage create a clear blocker. |
| ACCESS-02 | Pi access/hardware | Authorized SSH, architecture, mic, speaker, GPIO and safe physical setup recorded; unavailable devices marked BLOCKED. |

### B. Split and model tests

| ID | Test | Passing evidence |
|---|---|---|
| SPLIT-01 | Deterministic grouped split | Same source+seed+builder yields identical split manifest/hash. |
| SPLIT-02 | Validation coverage | All supported leaves have predeclared minimum useful train and validation support; all gaps documented before fit. |
| SPLIT-03 | Frozen test/holdout | Test and holdout row IDs/hashes remain identical to upstream; scripts refuse them as fit inputs. |
| MODEL-01 | Class order | Exact approved command label names/indexes match training, export, app, reports and UI metadata; any optional reject output is explicitly versioned. |
| MODEL-02 | Fit isolation | Instrumented loader proves only training subset is optimized; validation used only for selection; test/holdout never backpropagated or quant-calibrated. |
| MODEL-03 | Seed/reproducibility | Seed, package versions, configuration, code hash and split hash recorded; deterministic components reproduce within stated tolerance. |
| MODEL-04 | OOS/rejection | Approved OOS/unsupported-slot strategy and validation-selected reject behavior cause no action; thresholds chosen without test/holdout. |
| MODEL-05 | Quantization parity | Float and INT8 class order, finite outputs, input shape, validation parity and task-level differences documented. |
| MODEL-06 | Export compatibility | Pi ONNX/runtime loads candidate and produces the approved output vector; no missing external files. |
| MODEL-07 | Candidate isolation | Training/export leaves every current active model/config/release hash unchanged. |
| MODEL-08 | Test freeze/replay | Signed/frozen evaluation config precedes exactly one final test inference; outputs include all rows and failures. |
| MODEL-09 | Runtime benchmark | Warmed ARM64 frontend+model p50/p95/p99, peak RSS and audio pipeline latency measured on Pi; target under 10 ms p95 for model path. |
| MODEL-10 | Metric acceptance | Per-leaf, macro, intent-level, OOS, confidence, source/accent/speaker slices meet pre-registered gates or fail honestly. |

### C. Intent/action tests

For every phrase row in Section 5, test the provided golden audio clips where their manifest marks the corresponding phrase/variation, then use the same fixed holdout only in the live acceptance phase. For each prediction, test both the model route and direct action/API route so classifier errors can be separated from action errors.

| ID | Test sequence | Passing evidence |
|---|---|---|
| ACT-01 | Exercise all 19 top-level intents and all 31 leaves | Correct stable label/value, correct response, no unrelated state mutation. |
| ACT-02 | In-scope low confidence and OOS utterances | Explicit reject; state snapshots before/after identical for all devices. |
| ACT-03 | Unsupported slot values (e.g., alarm time/color/temperature/timer outside allowed values) | Rejected/no side effect; not coerced to nearest allowed value. |
| ACT-04 | Wake gating | Same command audio without wake does not execute; wake followed by command executes once; timeout clears listening state. |
| ACT-05 | Phrase variation | Score each of the 3 approved phrases per leaf and report variation-specific results. |
| ACT-06 | UI/API response mapping | API, assistant screen, status snapshot and audible feedback agree on command, value, actual state and errors. |
| ACT-07 | Repeat/idempotent commands | Repeating lights, pause, stop, alarm duplicate and reminder produces the documented stable result. |

### D. Lights and hardware tests

| ID | Test sequence | Passing evidence |
|---|---|---|
| HW-LIGHT-01 | LIGHT_ON from boot/off | White 100 percent; all three expected GPIO channels asserted. |
| HW-LIGHT-02 | COLOR_RED -> LIGHT_OFF -> LIGHT_ON | Off has all channels low; subsequent on restores white at 100, not red-only. |
| HW-LIGHT-03 | Repeat green and blue -> off -> on | Same all-channel restoration for both colors. |
| HW-LIGHT-04 | Each color from on and off | Exactly selected channel (or defined equivalent wiring output); UI hex/RGB matches pins. |
| HW-LIGHT-05 | Brightness 20/60/100 for white and a selected color | Correct monotonic PWM/brightness levels and stable selection. |
| HW-LIGHT-06 | Missing GPIO backend / disconnected hardware | Clear error/simulated status; no false hardware-success response; app stays usable. |
| HW-LIGHT-07 | Candidate API and physical check | Repeat key sequence using classagreementvcm candidate, then record pin readings and visual observation separately. |

### E. Music, alarms, timers, reminders and remaining actions

| ID | Test sequence | Passing evidence |
|---|---|---|
| MEDIA-01 | Offline PLAY_MUSIC | Bundled local track is audible; UI/API show matching track and playing state with network disabled. |
| MEDIA-02 | PLAY -> PAUSE -> wait -> PLAY | Audio pauses; position is preserved; resumed track continues rather than restarts. |
| MEDIA-03 | NEXT at middle/end of playlist | Next bundled track audibly starts; order and wrap policy are correct. |
| MEDIA-04 | STOP -> PLAY | Playback stops distinctly from pause and later starts under declared reset policy. |
| MEDIA-05 | VOLUME_DOWN/UP and bounds | Actual output changes by documented step; clamps at 0/100; device-unavailable error is truthful. |
| ALARM-01 | Fake clock before/after 6:00 AM, 8:00 AM, 9:00 PM | Correct next Asia/Manila due timestamp, including exactly-at-time boundary and next day. |
| ALARM-02 | Schedule two or three different alarm labels | All persist in list and fire at their own deadlines; one does not overwrite another. |
| ALARM-03 | Duplicate, dismiss and snooze | Duplicate policy is idempotent; dismiss affects the ringing alarm; snooze is exactly five minutes; event rings once. |
| ALARM-04 | Timer and alarm due together | Both events are surfaced and neither is lost; outputs do not race/block app. |
| ALARM-05 | Process restart/state-store recovery | Matches the approved persistence policy; malformed store falls back safely and reports error. |
| ALARM-06 | Pi audible alarm | Verified sound from actual speaker/buzzer; UI indicates which alarm is ringing; physical stop/dismiss works. |
| TIMER-01 | 10s, 30s, 1m with fake monotonic clock | Exact duration, single expiry event, no wall-clock/DST dependence. |
| TIMER-02 | Start a second timer while one is active | Previous timer replacement is announced and reflected in UI; only new timer expires. |
| REM-01 | Create each of 3 supported reminders and list | List exactly matches stored items; no hardcoded phantom items; duplicate/empty behavior matches spec. |
| SAFE-01 | CALL and MESSAGE | Recognized demo feedback; network/telephony/SMS APIs are never invoked. |
| SAFE-02 | TEMPERATURE_18/22/26 | UI labels simulated Celsius target; no GPIO/HVAC/network action. |
| SAFE-03 | TIME | Correct timezone-aware Asia/Manila output around date boundary. |
| SAFE-04 | WEATHER online/offline/malformed/network failure | Live result only when enabled/available; clear fallback otherwise; no crash or secret/API dependency. |

### F. Deployment, coexistence and rollback tests

| ID | Test | Passing evidence |
|---|---|---|
| DEP-01 | Candidate artifact manifest | Every file hash, class order, config, wake provenance, licenses and expected runtime version are present. |
| DEP-02 | Windows active-state invariant | Before/after hashes for current models, app, launcher, deployment receipt are unchanged. |
| DEP-03 | Pi sibling install | Candidate installs only into explicitly named classagreementvcm sibling folder; current <code>/home/dalmacio/Desktop/dandan</code> files/PID/state and <code>C:\Users\danda\Desktop\dandan</code> launcher/package remain untouched. |
| DEP-04 | Manual start/stop | Candidate uses a documented free loopback port, foreground/manual start; no service/autostart/public bind. |
| DEP-05 | Side-by-side isolation | Candidate and active API/UI/model paths and ports are unambiguous; each reports its own version/hash. |
| DEP-06 | Rollback | Stop/remove only the candidate install; verify active dandan app/model hashes and launch path unchanged and active app returns. |
| DEP-07 | Release receipt | Prepared, candidate-installed, candidate-tested, and promoted statuses are distinct; no receipt falsely marks promotion. |
| DEP-08 | Reboot/manual operation | Candidate does not start automatically; active service behavior unchanged; user can start the known active version. |

## 10. End-to-End Implementation Phases and Gates

### Phase 0: Freeze current baseline

Capture Git status and preserve current uncommitted changes. Record active model SHA-256 values, class order, wake and intent thresholds, frontend config, app/API version, Windows Desktop launcher/package hashes, Pi release receipt, active Pi model hashes if reachable, and all currently passing relevant tests. Never clean or revert user files to create a baseline.

**Gate 0:** Baseline and preserved rollback path are documented; existing changes remain intact.

### Phase 1: Gold audit and owner reconciliation

Run DATA-01 through DATA-12 against the latest revision. Generate a source/data audit report and exact schema crosswalk. The 10 OOS holdout rows are confirmed by the dataset-owner update and linked benchmark README; keep all 196 latest holdout rows untouched. Obtain any missing usage permissions and a statement of approved schema version. Decide how to represent true OOS and unsupported slot values before model fitting.

**Gate 1:** Hashes, labels, licenses, audio, provenance and count differences are reviewed. No unresolved in-scope label ambiguity.

### Phase 2: Dataset loader and deterministic splits

Add a dedicated Hugging Face Parquet reader and tests. Keep existing legacy loader/manifest unchanged. Map source records into a canonical internal schema while retaining source fields. Deduplicate with a crosswalk. Produce fixed gold-test/holdout inventories and a seed-stable speaker-group train/validation split from gold train only. Export exact counts and SHA-256. Do not include validation in fit.

**Gate 2:** Split/leakage tests pass; per-leaf coverage is approved; gold test/holdout file hashes are unchanged.

### Phase 3: Baseline and candidate model training

Register the current active pair as baseline without modifying it. Train the candidate from scratch with the approved 31 command/value outputs and the separately approved reject strategy. Do not force the 158 non-schema slot rows into true OOS without an explicit decision. The current wake model is inherited byte-for-byte. Tune only against validation. Export float and INT8, record all artifacts/config/logs, and confirm separate namespace/no sync. Quantization calibration uses train/validation only.

**Gate 3:** Candidate deterministic label contract and model parity pass; candidate remains unpromoted; acceptance gates are ratified before test exposure.

### Phase 4: App/action implementation

Wire the approved command outputs and reject path to the stable action dispatcher; OOS and unsupported slots must have no side effects. Make alarm scheduling multi-entry and timezone-aware, create state-backed reminders, make offline bundled music genuinely audible with pause-position semantics, preserve RGB restoration, use timezone-aware time, and add deterministic fake-clock/device fakes. Keep calls/messages no-send, temperature simulation explicit, and weather optional.

**Gate 4:** Full unit/service/API tests pass; no unresolved side-effect or state contract. Simulated tests are labeled simulated.

### Phase 5: Frozen offline evaluation

Freeze model, wake thresholds, intent thresholds, input frontend, action mapping, and evaluation scripts. Run MODEL-08 paired baseline/candidate inference once on gold test, publish raw aggregate report and per-class results with caveats. Do not tune on test. Evaluate gold holdout only for authorized live demo later.

**Gate 5:** Pre-registered quality gates pass or candidate is reported as not meeting them. No gate is relaxed after seeing scores.

### Phase 6: Raspberry Pi qualification

Confirm Pi network/access, OS, architecture, packages and physical devices. Build isolated candidate bundle. Copy only to the sibling classagreementvcm path; manually launch on loopback and a recorded free port. Run GPIO, audio, alarms, timers, all commands, wake gating, concurrency, and restart checks. Do not enable autostart or replace dandan. Run the fixed holdout only for the agreed live demonstration; preserve raw per-clip verdicts and never use it to tune.

**Gate 6:** Hardware evidence, p95/resource evidence, quality reports and rollback all pass. If Pi access/hardware is unavailable, mark deployment phase BLOCKED/NOT RUN, not passed. The candidate may run only from a separate sibling path such as <code>/home/dalmacio/Desktop/classagreementvcm</code>; do not reuse or relocate <code>/home/dalmacio/Desktop/dandan</code>.

### Phase 7: Handoff and independent promotion decision

Update the project and workspace handoffs with exact current state, artifacts, verification and open issues. Review candidate report with the class/project owner. Promotion is outside this implementation and requires a later explicit approval, chosen maintenance window, active-pair backup, tested rollback, and a separately recorded acceptance. Until then, active remains the currently deployed release.

**Gate 7:** User/team signs off on candidate test evidence and explicitly requests promotion. No implicit replacement occurs by completing this plan.

## 11. Access, Tools and Environment Requirements

- Use only the existing shared workspace environment at <code>AI 222 231/.venv</code>. Follow workspace and project AGENTS.md, HANDOFF_INDEX.md and SETUP_NOTES.md before implementation.
- Required packages: existing PyTorch/audio/Parquet stack and ONNX export/runtime packages already used by the project. Inventory versions first; install nothing until a missing dependency is demonstrated. Do not create a new virtual environment.
- Training host: current PC with sufficient free disk/RAM; current notes identify RTX 3050, but measure actual availability when work starts. Use deterministic data-loader workers/seeds and CPU fallback where needed.
- Pi target: Raspberry Pi 5 ARM64 with existing runtime and GPIO dependencies. Confirm <code>gpiozero</code>/<code>lgpio</code> compatibility, ALSA output/capture devices, loopback routing, Python, ONNX Runtime, and safe wiring.
- DGX is optional; only use with an authorized shared path/account and quota. Do not transfer or print credentials. Verify rsync/checksum and access controls if selected.
- Network is optional for core speech/actions. Local bundled music, lights, time, alarms, timers and reminders must work offline. Weather is an optional integration with a tested fallback.
- Privacy: voice is identifiable. Use only classmate audio with consent and education/training rights; anonymize IDs in shared reports. Keep licensed data and candidate artifacts private if license terms require it. Do not publish private audio or raw transcripts without approval.

## 12. Five Review Iterations

The plan was reviewed in five separate passes. These reviews changed the plan and found gaps; they did not change the code, weights, or deployment.

### Iteration 1: Scope and meeting alignment

**Review question:** Does the plan reflect the meeting decisions and avoid expanding them into unapproved product features?

**Finding:** The meeting fixes the 19-intent/slot/three-phrase benchmark, asks for all collected/generated material to be used, and fixes a model-inference test set. The current gold master already collates datasets and defines train/test/holdout/numerals. “All data” must not mean duplicate copies, test leakage, or license-blind merging.

**Plan change:** Gold snapshot is canonical; verified missing supplements may enter only gold train after identity/license audit. Test and holdout are frozen. Numerals stay out of the command classifier. Calls/SMS/HVAC are not implied by command recognition.

### Iteration 2: Intent, slot and phrase completeness

**Review question:** Are all classes, values and benchmark phrases named exactly?

**Finding:** The master contains 19 top-level commands but 31 permitted command/value leaves and 93 phrases. Equal cardinality with the existing 31-class head is not proof that every index maps correctly. Existing intent classes do not include OOS.

**Plan change:** Listed every phrase triple and exact proposed leaf mapping. Added a row-by-row crosswalk and immutable class-order requirement, and recommended a 32nd OOS output plus validation-only reject gate. Clarified out-of-range values are not coerced to nearest class.

### Iteration 3: Dataset integrity and split leakage

**Review question:** Can the planned evaluation be trusted and reproduced?

**Finding:** Published gold test and holdout must not be regenerated or used for fitting. Gold has no validation split; current old test is reused. README says two holdout clips for each of 93 variations (186), while the actual pinned holdout has 196 rows: 186 allowed rows plus 10 OOS rows. All 10 OOS rows store the documented blank variation as an empty string, explaining the 94 distinct raw variation values; this is not an extra in-scope variation. The conversation also sets an approximate 1:1 real/synthetic balance and a priority for real voices: classmate/group, open-source non-native English, then open-source native English.

**Plan change:** Carve deterministic group-disjoint validation from gold train only; hash each split; check speaker/synthetic identity and content hashes. Added mandatory owner confirmation that the 10 OOS rows are intentionally in holdout, without dropping or moving any row. Added per-variation source-balance audit and one-time frozen test exposure/leakage stop rules.

### Iteration 4: RPi action semantics and missing tests

**Review question:** Does every intent have an observable, safe, testable result?

**Finding:** LIGHT_ON restoring white/all channels is already present and regression-covered. Music has state/track metadata but lacks direct audible lifecycle tests. Timer has a single deadline without dedicated tests. Alarm labels exist, but one alarm pointer is checked while an active list is displayed; timezone and audible buzzer are not assured. Reminders are currently hardcoded/acknowledgement-only.

**Plan change:** Added state transition semantics for each of the 19 commands and every slot; multi-alarm scheduler and Manila timezone; fake-clock tests; accurate reminders; genuine offline playback and pause-position checks; explicit simulation/no-send/OOS no-action behavior; physical Pi evidence requirements.

### Iteration 5: Deployment, access, privacy and rollback

**Review question:** Could following the plan accidentally replace the working model or expose data?

**Finding:** The standard training/launcher path can synchronize the latest model. The latest recorded Pi check was unreachable and lacked capture hardware. Classmate voice, source licenses, and public-Git exposure are real constraints.

**Plan change:** Candidate-only output and sibling install are mandatory; active hashes and PIDs must remain unchanged; no autostart/public bind; deployment/rollback have separate tests. Recheck Pi access and mic before claiming qualification. Added consent/license gates and classified inaccessible hardware as BLOCKED/NOT RUN. Promotion requires a future explicit decision.

### Five-pass conclusion

The five review iterations above are historical. The latest linked revision audit closes the 10-row holdout confirmation and corrects the 158-row interpretation. Before training, resolve source rights/consent and choose a safe candidate policy for true OOS and in-scope intents with unsupported slot values. An explicit OUT_OF_SCOPE model output is optional; the benchmark also scores silence/no response as reject. Alarm persistence/recurrence and measurable acceptance thresholds still need owner confirmation. None of these unresolved points justify touching the active release.

## 13. Open Decisions and Recommended Defaults

1. **Holdout 196-row composition - resolved:** The owner stated OOS data were added to test and holdout; the current benchmark README confirms 186 supported commands plus 10 OOS in holdout. Preserve the latest hash-pinned rows for live demo only.
2. **Reject and unsupported-slot policy - open:** Decide whether true OOS uses a learned output, a validation-calibrated no-response/reject gate, or both. Separately define what to do when the intent is valid but the slot is unsupported. Do not treat these two row types as interchangeable.
3. **Alarm recurrence:** Default one-shot for the next Asia/Manila occurrence. The schema says “set an alarm” at a time but does not express “daily.”
4. **Multiple alarms:** Support multiple distinct active alarms, with duplicate exact due times idempotent. A later command must not overwrite another alarm.
5. **Alarm/reminder persistence:** Default persist alarms locally and recover safely after restart. Keep reminders session-only unless durable reminders are explicitly wanted; there is no time slot or reminder scheduler in the schema.
6. **Timer replacement:** One timer; a new timer replaces it and tells the user. There is no timer-cancel label.
7. **Music sources:** Default to the bundled local demo playlist so all music tests work offline; external streams/search are out of scope for this candidate unless licensed and separately approved.
8. **CALL/MESSAGE:** Keep as no-send demonstrations. Real telephony or messaging requires separate API, consent, confirmation, privacy and abuse-prevention scope.
9. **Metric thresholds:** Ratify proposed gates before gold test access, especially per-leaf 95 percent F1, OOS false action rate, class minimum supports, and Pi latency.
10. **Pi access:** Reconfirm SSH, physical mic/speaker/RGB and safe wiring; last recorded unreachable/no-capture state is historical, not a current-state assertion.

## 14. Definition of Done

The candidate is ready for a later promotion decision only when all of the following are evidenced:

- The latest pinned dataset passes the complete audit; source-use rights are resolved; the owner-confirmed holdout stays frozen.
- Exact 19-intent, 31-leaf, 93-phrase mapping and OOS policy are approved and tested.
- Deterministic group-disjoint train/validation is frozen; gold test/holdout/numerals remain uncontaminated and hash-identical.
- Candidate training/export is reproducible and uses the approved reject/slot policy with a byte-identical inherited wake model.
- Frozen paired test report meets ratified quality gates or clearly marks which gates failed; no claims use holdout for tuning.
- Each RPi intent, slot and state transition has direct test evidence; physical lights, audio, timer and alarm have hardware evidence where required.
- Candidate runs only from an isolated sibling path; no active model, deployment, launcher or autostart was changed; rollback was exercised.
- The project and workspace handoffs accurately state status and remaining risks.
- A separate, explicit user/team approval is obtained before any active-model promotion. Completion of this plan does not itself authorize replacement.
