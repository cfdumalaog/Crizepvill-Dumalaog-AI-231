# Antigravity task: Tiny VCM wake repair and 10-second timeout

Paste this task into the existing Antigravity VCM session. Work in the current
AI 231 ME2 repository and use the shared handoff index as the source of truth.
Read the companion `ANTIGRAVITY_EVIDENCE_RESTRICTIONS.md` before acting.

## Role and working agreements

You are the VCM implementer and may suggest the next improvement. Codex is the
independent auditor. Do not treat a plan, notebook cell, test name, dashboard
response, or previous report as proof that work ran. Only report results you
actually observed in this run and preserve earlier user changes and artifacts.

Before touching files, read:

1. Workspace `AGENTS.md`, `HANDOFF_INDEX.md`, and `SETUP_NOTES.md`.
2. AI 231 repository `README.md` and its Git status.
3. This project's `HANDOFF_INDEX.md` and `deployment/ANTIGRAVITY_EVIDENCE_RESTRICTIONS.md`.

Use the one shared environment at
`C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\.venv`. Do not make
another environment. Do not discard, clean, reset, or overwrite unrelated dirty
files. Update the project and workspace handoffs as required by `AGENTS.md`.

The work queue is in the project `HANDOFF_INDEX.md`. Task C is READY; Task D is
QUEUED. Claim only Task C, record that you started it in the handoff, and do not
start Task D until Task C has been returned to Codex and the handoff marks Task D
READY. At every return, give Codex concrete evidence and wait for its audit.

## Current facts to preserve

- Raspberry Pi 5: `cfdfnjrpi5` at `192.168.254.106`; SSH account `dalmacio`.
  Key-based SSH is already verified. Use the existing key and batch mode; never
  request or type a password.
- Installed service: `tiny-vcm-antigrav.service`, enabled and active, bound to
  loopback `127.0.0.1:7860`, working directory `/home/dalmacio/tinyvcm-rpi5`.
- Installed wake model: `models/antigrav_wake32_int8.onnx`, 39,384 bytes,
  SHA-256
  `551836d42f33a34c9a6d13e37e963bcdf82470ba9ae7d71f080941306e351bf3`.
  Preserve that exact model for Task C. The 31-class command model is also to
  remain byte-for-byte unchanged during Task C.
- The current service timeout is 5 seconds. The live runtime uses a 32-class
  model, requires top-1 `WAKE_WORD` at confidence >= 0.60, checks a 2.5-second
  rolling buffer about every 0.35 seconds, and has a VAD RMS gate of 0.012.
- Codex compared the exact installed ONNX with the 16 unique PCM wake clips in
  `data/human/manifest.csv`: only 8/16 reached the live 0.60 trigger in centered
  offline windows, even though these are training examples. This is diagnostic
  evidence only; it is neither held-out performance nor a live streaming test.
- The current human wake set is from one speaker. Do not describe results as
  multi-speaker or robust to unseen speakers. No live ambient/noise negatives
  are currently established.
- Existing 31-class command model remains the canonical course model. The
  binary wake detector in Task D is an experiment and must not silently replace
  any canonical artifact or Pi model.

## Task C — change inactivity timeout from 5 to 10 seconds

Make this isolated fix first. Keep both installed ONNX files byte-identical.

1. In `antigrav_demo.py`, set the default and actual server configuration to
   10.0 seconds. Update the wake/timeout status text, countdown scaling, and
   related documentation so they display the configured 10-second value and
   cannot drift from the actual state deadline. The ten-second deadline must
   start at wake and reset after every accepted command, as it does today.
2. Add or update focused tests. Prove that the assistant stays in LISTENING
   before the ten-second deadline, returns to STANDBY after ten seconds of
   inactivity, and receives a fresh ten-second deadline after a recognized
   command. Test configured state/API and countdown inputs; do not simulate a
   voice wake and report it as voice recognition.
3. Run the relevant test suites, compile/lint checks that the repository
   already uses, and the release builder/verifier in the shared `.venv`. Record
   exact commands, exit codes, and outputs. Build a new application bundle
   without changing either model. Verify the ZIP, manifest and checksums, and
   explicitly compare both model SHA-256 values before and after packaging.
4. Update the project README and handoff with the actual implementation,
   current limitations, tests, package hash, and deployment status. Update the
   workspace handoff and run its required refresh script. Stop and return to
   Codex for audit. **Do not deploy Task C until Codex approves the exact
   source diff and bundle hash.**

After Codex approval, follow the deployment task it records in the handoff:

- Transfer and install only the approved bundle on the Pi using the existing
  SSH key. Preserve a rollback copy/snapshot; do not change GPIO wiring or
  enable physical GPIO actions.
- Record the bundle hash before/after transfer, both model hashes before/after,
  `systemctl` enabled/active status, actual process command, API `timeout_sec`,
  API page health, and startup errors. The service must remain bound to
  `127.0.0.1:7860`.
- Verify the inactivity transition using an explicitly manual wake only for
  testing the timer: LISTENING remains before ten seconds and returns to
  STANDBY after ten seconds. Label this as a timer test, not a spoken-wake test.
- Update the handoff with exact Pi-side evidence, then stop for Codex's
  independent hash/service audit.

## Task D — separate two-class wake detector (only after Task C is audited)

The objective is to compare a dedicated wake detector against the installed
32-class wake extension while leaving the 31-class command model unchanged.
Build and report an experimental candidate; do not deploy it in this task.

1. Create `notebooks/ME2_Wake2_Binary.ipynb` with genuine, visible epoch-by-
   epoch training progress. Train a new model from random initialization with
   exactly two output labels: `NON_WAKE` and `WAKE_WORD`. Do not copy, fine-tune,
   distill from, or initialize any part from an existing VCM checkpoint.
   Record initialization seed, initial-weight hash, parameter count, data
   provenance, and `pretrained_weights_used: false` in machine-readable output.
2. Keep source recordings grouped across train/validation/test. Deduplicate
   repeated rows by decoded PCM/source identity before splitting. Use only
   training data for fitting, augmentation and INT8 calibration; choose the
   operating threshold only on validation. Do not access the held-out set until
   both FP32 and INT8 candidates, thresholds, and inference behavior are fixed.
3. Positives may use the existing real wake recordings and clearly identified
   supplemental synthetic wake audio. Keep the one-speaker limitation explicit.
   Do not fabricate more people, recordings, or room conditions. To claim
   multi-speaker performance, wait for and use real recordings from additional
   speakers with speaker-disjoint evaluation.
4. Build `NON_WAKE` from diverse real command speech and hard negatives,
   including wake-like phrases, ordinary speech, silence, music, and noise where
   those recordings exist. Apply augmentations only to training clips and track
   source groups so derived copies cannot leak between partitions. Report any
   missing negative conditions as limitations; synthetic noise alone does not
   prove real-room robustness.
5. Match training windows to inference. Either retain the 2.5-second model
   input and create randomized offsets/context windows that reproduce the
   live rolling buffer, or propose a different window and implement/test its
   exact runtime preprocessing. Document window duration, stride/cadence,
   endpoint/debounce rule, VAD behavior, resampling and normalization. Do not
   evaluate only perfectly centered clips and claim streaming success.
6. Compare the binary model and current installed wake32 model on identical,
   properly partitioned evaluation windows. Report wake recall by speaker and
   recording condition, false accepts on speech/commands/noise, and false
   activations per hour for continuous negative audio when enough actual audio
   exists. Report threshold trade-offs; never lower a threshold based on test
   results. Keep prior test sets that have already been used clearly labelled;
   do not reuse them to tune the new model.
7. Export an experimental INT8 ONNX under a new run-specific path outside
   canonical `models/`. Verify two-class label order with tests, FP32/INT8
   agreement and accuracy/recall, ONNX size (<500 KB), hash, and desktop CPU
   latency. The target is <10 ms for full frontend plus model on Pi 5; report
   this only if measured on the Pi after Codex approves a separate deployment.
8. Keep the existing command classifier and runtime untouched until the
   candidate and integration pass Codex audit. Record tests, failures, notebook
   execution counts, artifacts and all limitations in the handoff. Stop for
   Codex review; no model deployment is authorized by this experiment prompt.

## Required return report

Use a short evidence table with columns `claim`, `artifact/log`, `exact command
or measurement`, `result/exit code`, and `limitation`. Include the changed-file
list and a concise suggestion for the next step. Distinguish clearly between:

- implementation completed locally;
- tests/builds that actually ran;
- static recorded-audio inference;
- live microphone/spoken wake;
- Pi service installation and measured hardware behavior.

If a command fails, leave the failure visible and report it as failed. If you
cannot verify a claim, label it `NOT VERIFIED` and state exactly what evidence is
missing. Do not call the system finished, perfect, production-ready, deployed,
multi-speaker, or real-time unless the corresponding acceptance check ran and
the measured evidence supports that exact claim.
