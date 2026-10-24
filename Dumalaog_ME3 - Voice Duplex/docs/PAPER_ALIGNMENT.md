# ME3 paper alignment and Raspberry Pi boundary

This file is the implementation boundary for the source papers placed in the
ME3 folder. It prevents the local engineering benchmark from being mistaken
for a completed FULL-DUPLEX-BENCH v1.5 experiment.

## Source documents

- `Building AI Voice Agents - Summary.pdf`: cascaded STT -> LLM -> TTS,
  streaming/full-duplex operation, VAD/endpointing, interruption flush,
  context alignment, and latency/telemetry targets.
- `FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf`: four overlap scenarios,
  exact counts, signal treatments, `RESPOND`/`RESUME`/`UNCERTAIN`/`UNKNOWN`
  categories, stop/response timing equations, and pre/post response features.
- `refs/FDB_paper_2503.04721.pdf` plus `refs/full-duplex-bench/v1_v1.5/`:
  the earlier FDB definitions and reference evaluator conventions.
- `plan/ME3_Voice_Duplex_Plan.pdf`: Raspberry Pi 4/5 deployment, owned local
  ASR/TTS/VAD MCP services, full-duplex audio, and systemd operation.

## What is implemented

- `me3/agent.py` and `me3/audio_io.py` provide the local full-duplex loop:
  microphone capture and speaker playback remain open together, and speech
  during TTS can stop playback and start a new turn.
- `me3/fdb_v15.py` is the paper-native v1.5 contract. It contains the exact
  four scenario counts, expected categories, response labels, timing
  equations, and deterministic signal treatments. It accepts the paper labels
  (`Respond`, `Resume`, `Uncertain`, `Unknown`, normalized to uppercase) and
  maps the checked-in reference evaluator's `C_*` aliases to the same four
  paper categories.
- `scripts/fdb_v15_eval.py` scores captured overlap artifacts. It preserves
  missing boundaries as missing and reports incomplete scenario counts as an
  incomplete paper run.
- `me3/bench_driver.py` remains the separate engineering latency driver. It
  is useful for local pipeline timing, but its five latency scenarios and
  legacy FDB-v1 summary are not a v1.5 result.
- `scripts/check_paper_alignment.py` checks the checked-in source contract,
  required papers, owned MCP servers, full-duplex configuration, and the Pi
  release surface.

## Deliberate evidence boundary

The live product loop currently uses VAD speech during agent playback as an
immediate barge-in signal. That is appropriate for interruption responsiveness,
but it does not by itself classify a short backchannel, off-axis speech, or
background speech as `RESUME`. A valid v1.5 claim therefore requires captured
overlap trials, an explicit behavior category for every trial, and the
paper-native evaluator; the static contract does not substitute for those
measurements.

The live v1.5 run is not yet complete. No claim should be made for the 499
required utterances, behavior accuracy, prosodic adaptation, or Pi latency
until those artifacts exist.

## Raspberry Pi path

The release is intended for 64-bit Raspberry Pi OS. `install.sh` selects
`base.en` on smaller-memory Pi systems and `small.en` on larger systems unless
`ME3_ASR_MODEL` is explicitly set, uses int8 ASR, prefetches the primary and
overlap Piper voices, writes a non-secret runtime environment file, and
installs four user systemd units. The normal agent uses the configured/default
PortAudio input and output devices, so it does not depend on Windows Stereo
Mix.

The speaker-stimulus engineering benchmark needs a capture path that hears
the played stimulus. On Windows this is normally Stereo Mix. On Raspberry Pi
pass `--input-device` or set `ME3_BENCH_INPUT` to an ALSA loopback or other
explicit capture device; a physical microphone is suitable for the live agent
but is not an exact internal-loopback benchmark.

The Pi deployment receipt (`deployment/last_pi_deployment.json`, the
source of truth for the current release ID and hash) now reports `deployed`
at
`/home/dalmacio/Desktop/kiko/me3_voice_duplex`. The installed `me3.env` is
mode 600 and the Qwen key is provisioned (presence verified only). Pi
`--check` passes and a live text-mode Qwen run built a stopwatch web app
through `write_file` plus `start_web_server`, served it over HTTP, and the
page was cross-verified from a second machine on the LAN. The Pi currently reports zero
PortAudio devices and its user service is disabled, so this receipt does not
prove live microphone/speaker hardware, systemd supervision, ALSA capture, or
Pi latency. A physical audio check remains required before claiming live Pi
voice operation or paper-grade benchmark evidence.
