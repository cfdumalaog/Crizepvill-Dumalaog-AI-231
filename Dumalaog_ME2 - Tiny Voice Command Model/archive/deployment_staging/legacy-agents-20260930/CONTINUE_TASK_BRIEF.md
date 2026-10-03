# Continue task brief: local VCM microphone and wake reliability

You are Continue, an optional read-only reviewer using the locally served Qwen
27B model. Codex plans the work and performs the final audit. Chat in VS Code
is the preferred local implementation agent. Antigravity handles Pi deployment
only if the user powers on the Pi and reopens that scope.

The workspace Continue configuration selects `Qwen/Qwen3.8-27B`. This brief is
prepared for review, but it has not been submitted to a Continue chat and there
is no Continue response yet. The current Codex computer-use surface exposes no
native VS Code/Continue chat control; after Codex audit, the user can paste this
file into Continue. Do not treat file creation as an agent handoff.

## Objective

Independently investigate the current local PC microphone and two-stage wake
path, identify why wake scores can activate without a controlled phrase, and
recommend the next measured experiment. The user wants the final design to use
one model for binary wake/non-wake, then a separate compact intent model after
wake. The Raspberry Pi is powered down; all work is local.

## Required reading

1. Workspace `AGENTS.md`, `HANDOFF_INDEX.md`, and `SETUP_NOTES.md`.
2. Project `HANDOFF_INDEX.md` and `docs/LOCAL_WAKE_SENSITIVITY_AUDIT_20260928.md`.
3. Inspect current source and current Git status before making any suggestion.

## Evidence already collected

- The old server selected silent Realtek WDM-KS input 36. Fifine WDM-KS inputs
  40 and 41 returned audio; input 41 had the higher short-window RMS. Fifine
  MME and WASAPI inputs tested here did not open successfully.
- Current local entry point: `http://127.0.0.1:7863/studio`; use
  `scripts/Start-Local-VCM.ps1` from the project folder to start it.
- Current local trial: Fifine selected by name, VAD RMS gate 0.02, binary wake
  threshold 0.90, inference every 0.12 seconds, and two positive wake windows
  within 0.45 seconds. Intent classification uses the separate 31-class ONNX.
- Studio is served by a Python backend that opens the Windows microphone; the
  browser mic permission is used by the dataset recorder, not this listener.
- At the trial settings, the saved 16 single-speaker wake takes produced
  15/16 two-window detections. A live wake event occurred without a controlled
  user phrase, so the wake path is not considered validated. A 30-second
  no-save audio probe had three RMS windows above 0.02 and no scores above 0.90.
- Latest `/api/state` snapshot at 2026-09-28 13:37:40 Asia/Manila was STANDBY,
  selected Fifine WDM-KS input 41, reported no runtime error, but the stream was
  inactive and the last audio was 332 seconds old. Its event log shows voice
  wake and intent events around 13:08–13:13, including `Hi Dandan`; the user has
  not identified those events as a controlled evaluation. They are not ground
  truth. The running PID 26232 used the Windows Store Python executable instead
  of the workspace `.venv`; do not claim this snapshot verifies the canonical
  launcher environment.
- No audio was saved in those live probes. The candidate model remains
  experimental and the Pi is not to be accessed or deployed to.

## Scope and restrictions

- Work only in the local workspace. Do not contact, wake, or deploy to the Pi.
- Do not collect or save microphone audio. Do not alter model weights, ONNX
  exports, runtime source, tests, or the handoff files during this review.
- Do not invent a successful speech test from HTTP responses, manual wake
  buttons, model replay, or unlabelled live wake events.
- Do not lower the wake threshold based on the existing training takes alone.
- Keep claims tied to exact commands and outputs. Mark inference and
  recommendation separately from observed facts.

## Deliverable

Return a concise review containing:

1. Root-cause hypotheses ranked by evidence, including the microphone routing
   fix already made and remaining explanations for the uncontrolled wake.
2. A bounded next experiment with acceptance criteria for wake recall and
   false activations, using speaker-separated positives plus room and near-miss
   negatives.
3. Any runtime or data concerns Codex should address before promoting the
   candidate.
4. Commands you actually ran and their unabridged results. If you did not run a
   check, say so. Do not edit the workspace; send recommendations to Codex for
   audit before implementation.

After reading the brief, inspect the current handoff state first; it may have
changed since this file was written.
