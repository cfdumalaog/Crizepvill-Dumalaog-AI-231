# Antigravity VCM evidence and safety restrictions

These rules apply to every VCM task, report, notebook and Pi deployment. They
are also embedded by reference in `ANTIGRAVITY_VCM_REPAIR_PROMPT.md`.

## Truthfulness and evidence

- Report only commands, tests, training runs, measurements and deployments that
  actually happened. Never infer success from a plan, code inspection, HTTP
  200, old report, static mock, or a tool being available.
- For each verification claim, preserve the exact command, exit code, concise
  output, and artifact path. Keep full logs when useful, but redact secrets and
  avoid logging spoken personal content.
- Keep failures, warnings, skipped checks, and unresolved mismatches visible.
  Do not replace a failed log with a hand-written success summary. Use
  `NOT VERIFIED` when evidence is absent.
- A notebook counts as executed only when its code cells have real execution
  counts and outputs from the run, with zero stored error outputs. Never type
  expected training curves, metric tables, execution counts or final answers
  into output cells by hand. Save real epoch logs and actual checkpoints.
- “Trained from scratch” requires a new random initialization, no loaded
  checkpoint/teacher weights, and recorded initialization provenance/hash.
- Separate evidence categories: desktop unit tests, saved-clip inference,
  continuous-stream test, live microphone wake, Pi deployment, and measured Pi
  performance. One category never proves another.
- A manual API/button wake tests only the state machine or inactivity timer.
  It does not prove the wake-word detector heard speech. An API health check
  does not prove that the microphone or model recognized a command.
- Never claim Pi deployment until the exact release hash is on the Pi, the
  running process/service is tied to that release, and post-install checks ran.
  A restart alone is not proof. Record model hashes separately from bundle
  hashes.
- Never claim voice recognition from a prepared test clip as live microphone
  recognition. A centered WAV evaluation is not a streaming evaluation.
- Do not call a model robust, multi-speaker, noise-robust, real-time, or
  production-ready without measured evidence matching that claim and its
  conditions. State speaker counts, supports, test conditions and uncertainty.

## Data and evaluation integrity

- Preserve speaker, source-recording and augmentation groups across splits.
  Deduplicate PCM/source variants before splitting. No derivative of a source
  clip may cross train/validation/test.
- Fit weights, augmentation policy, normalization statistics, INT8 calibration
  and threshold selection using training/validation data only. Freeze the model,
  threshold and inference logic before the final held-out evaluation.
- Do not reuse test data for training, calibration, checkpoint selection,
  threshold tuning or a second “final” evaluation. If a prior test set was
  already inspected, disclose that and reserve new data for a clean comparison.
- Include hard negatives: ordinary speech, commands, near-miss wake phrases,
  silence and representative environmental audio where available. Do not claim
  real-room false-accept performance based only on synthetic noise.
- The current self-recorded wake set contains one real speaker. Do not invent
  speaker IDs or claim the three-person collection requirement is met. Request
  additional real recordings when multi-speaker evidence is needed.
- Keep raw human audio local and ignored by Git; do not upload it to cloud
  services or include it in Pi release bundles. Preserve required attribution
  and licensing for third-party corpora, including SLURP CC BY-NC audio.

## Scope and operational safety

- Use the shared workspace `.venv`; do not create a second environment.
- Read `AGENTS.md`, root and project handoffs, setup notes, project README and
  Git status before work. Preserve all existing user changes and unrelated
  artifacts; do not reset, clean, or overwrite them.
- Follow the active owner and one-task-at-a-time state in the project
  `HANDOFF_INDEX.md`. Do not promote a queued task to active without the
  handoff authorizing it. Stop at Codex audit gates.
- During the timeout-only task, keep both current ONNX model files byte-for-
  byte identical. During wake-model experiments, put outputs under a new
  run-specific path; do not replace canonical artifacts or the Pi model.
- Do not change GPIO wiring or trigger real actuators. Keep GPIO simulated
  unless a separate, verified wiring task authorizes a specific physical test.
- Use existing key-based SSH access for the Pi. Do not request, type, echo,
  save, or paste a password, token, key, or credential into terminal commands,
  logs, chat, handoff, source, notebook or screenshots. Never place secrets in
  a task report.
- Keep the service bound to loopback. Do not expose port 7860 to the LAN or
  public Internet. Do not disable the firewall or weaken OS security.
- Do not commit, push, publish, or send user data to an external service unless
  the user separately asks for that exact action. The Pi deployment task in
  the handoff is a local device transfer over the already authorized SSH key.
- Preserve a rollback path before a Pi update. Never remove the current working
  deployment/model until the replacement is verified and Codex has audited it.

## Required handoff update

Before returning work, update the project handoff with current state, known
limitations, open items, actual verification, exact hashes and one newest-first
change-log entry. Update the root handoff and run
`& '.\scripts\Update-HandoffIndex.ps1'` as required by `AGENTS.md`. Never mark a
task complete pending Codex review. Include a small evidence table and wait for
Codex to audit before taking the next queued task.
