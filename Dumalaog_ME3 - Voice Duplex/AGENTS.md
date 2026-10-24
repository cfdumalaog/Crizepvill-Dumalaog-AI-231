# ME3 - Voice Duplex: local-dev workflow

Follow the parent workspace and AI 231 instructions. Preserve both Kiko folders
and the Desktop `dandan` VCM copy; ME3 deploys to its own Pi path
(`/home/dalmacio/Desktop/kiko/me3_voice_duplex`) and its own Desktop staging
area (`~/Desktop/me3_voice_duplex`), so it never touches ME2's `dandan`.

## Local development before Pi deployment (the ME2 pattern)

ME3 is developed and validated on the Windows dev box FIRST, then shipped to
the Pi. Local success is not proof of Pi deployment. The pipeline:

1. **Check the environment** - `python scripts/dev_env_check.py`
   Confirms the shared `.venv` has every ME3 dependency (pipecat, mcp,
   faster-whisper, piper, torch/silero, sounddevice, ...).

2. **Smoke-test locally** - `python scripts/local_smoke_test.py [--with-audio]`
   Imports all four MCP servers (ASR, TTS, VAD, tasks), calls each `health()`
   tool, and constructs the latency benchmark - no Pi, no network, no hardware
   required. `--with-audio` additionally probes PortAudio devices.

3. **Package a release** - `python scripts/package_release.py`
   Builds `dist/me3-<stamp>-<fp>/me3_voice_duplex`, writes a sha256
   `bundle_manifest.json`, zips it, and records `deployment/latest_release.json`
   with `status="prepared"`. Idempotent: unchanged source reuses the release.

4. **Deploy to the Pi** - `python scripts/deploy_to_pi.py`
   Stages the zip + a `Deploy-ME3.bat` launcher to `~/Desktop/me3_voice_duplex`,
   then scp's the zip to the Pi and runs `install.sh` there. Records
   `deployment/last_pi_deployment.json`.

## Deployment state (read these to know where things stand)

- `deployment/latest_release.json` - the last PREPARED release (local only).
- `deployment/last_pi_deployment.json` - the last Pi sync outcome:
  `never_deployed` / `pending_sync` / `sync_failed` / `deployed` / `dry_run`.

A `pending_sync` means the Pi was unreachable; the prepared release is
preserved and the next run (or `Deploy-ME3.bat`) retries. Never report a
`prepared` release as deployed until `last_pi_deployment.json` says `deployed`.

## Rules

- Use the single shared workspace `.venv`. Do not create a course-local env.
- No secrets in this folder, in chat, or in the release. The Pi LAN IP in
  `config/config.yaml` is a placeholder until confirmed.
- The four MCP servers (`me3/mcp_asr.py`, `me3/mcp_tts.py`, `me3/mcp_vad.py`,
  `me3/mcp_tasks.py`) are OUR code and must run on the Pi (task requirement).
  The underlying models (whisper/piper/silero/LLM) may be commercial/pretrained.
- The latency benchmark (`me3/benchmark.py`) is CLASSWORK: it uses a fixed,
  group-shared protocol and stimulus set (`data/DATASET.md`) so every group
  member measures identically. Do not change the metric definitions without
  updating `data/DATASET.md` and the plan.
- Keep `config/config.yaml` the single source of truth; every service reads it.

## Class-group note

The whole class shares the same benchmark dataset and metric definitions so
results are comparable. `data/DATASET.md` is the contract; `me3/benchmark.py`
implements it. Record raw trial data under `runs/` (gitignored) and aggregate
with the benchmark's CSV/JSON writers.
