# Chat task brief: relaunch and verify the local Tiny VCM

You are Chat, the local implementation agent using Qwen/Qwen3.8-27B in the
user's VS Code workspace. Codex is the task planner and independent auditor.
Antigravity remains the deployment operator only if the user powers on the Pi
and reopens that scope. Continue is optional; do not wait for it.

## Read first

1. Workspace `AGENTS.md`, `HANDOFF_INDEX.md`, and `SETUP_NOTES.md`.
2. Project `HANDOFF_INDEX.md`, `README.md`, and `git status`.
3. Recheck the current state before acting; the status below is a snapshot and
   may already be stale.

## Objective

Get the two-stage local PC demo running with the workspace's shared `.venv`,
then support a deliberate wake-word check. The intended architecture remains a
binary wake/non-wake model followed by the separate intent model. Do not train
or replace either model in this task.

## Starting evidence from Codex

- At 2026-09-28 13:37:40 Asia/Manila, `/api/state` on port 7863 showed STANDBY,
  no error, Fifine WDM-KS input 41 selected, but no active microphone stream and
  an audio age of 332 seconds.
- The historical API event log included voice wake and intent labels around
  13:08–13:13. The user did not confirm that these were deliberate test trials;
  treat them as unattributed events, not ground truth.
- PID 26232, the port-7863 owner at that check, ran from Windows Store Python,
  not the canonical shared `.venv`. Codex's attempt to restart it was rejected
  by command policy; Codex left the process untouched.
- The canonical launcher is `scripts/Start-Local-VCM.ps1` in this project and
  targets the workspace `.venv`.

## Scope and limits

- Work only in this local workspace. The Pi is powered down and out of scope:
  do not SSH, wake it, deploy, or change any Pi service or file.
- Do not record or save microphone audio. Do not inspect private credentials.
- Do not change model weights, ONNX exports, thresholds, release bundles, or
  the command vocabulary. Do not lower the wake threshold to make a trial pass.
- Before stopping a process, verify its PID, executable, command line, and
  listener port belong to the VCM demo. If permissions block a safe restart,
  leave it alone and report the exact blocker.
- Start with the canonical PowerShell launcher. Confirm the running executable
  resolves inside the workspace `.venv`, port 7863 serves the app, and `/api/state`
  receives fresh audio blocks from Fifine without errors or dropped chunks.
- Ask the user to say “Hi Dandan” three times in a controlled trial. Match each
  attempt to its timestamp and wake event. Do not infer which events were user
  trials without their confirmation. Then record an equivalent quiet-room
  observation and near-miss speech as user-labelled results, without saving
  audio. If the user is unavailable, stop after runtime verification and leave
  wake accuracy explicitly unverified.
- Make a code change only if runtime verification exposes a concrete defect.
  Keep it minimal, test it, preserve unrelated changes, and update both handoffs
  under the workspace rules.

## Report back to Codex

Give exact commands and exit codes, actual executable path, PID/port, selected
mic, before/after API fields, dropped-block/error counts, and each user-labelled
trial result. Separate observations from hypotheses. Include changed file paths
and test output if any. Do not call HTTP health, manual wake buttons, old log
entries, or saved-WAV replay a live spoken success.
