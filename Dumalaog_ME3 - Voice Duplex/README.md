# ME3 — Voice Duplex

Full-duplex, voice-enabled AI agent for Raspberry Pi 4/5 and local Windows
development. Interruptible, conversational speech with a separate engineering
latency benchmark and a paper-native FULL-DUPLEX-BENCH v1.5 overlap evaluator.

Built with **Pipecat** (harness) + **MCP** servers (ASR/TTS/VAD/tasks) that
we own and run on the Pi. Commercial APIs allowed for the LLM brain; the
MCP servers and VAD are ours.

## Run locally (like ME2)

Double-click `launchers\Start-ME3.bat`. It starts the full-duplex voice loop
in a console window using the shared workspace `.venv` — the mic stays live
while the agent speaks, and you can interrupt it mid-sentence (barge-in).
`Ctrl+C` or closing the window stops it. See [launchers/README.md](launchers/README.md).

```powershell
# or from PowerShell, with options
& '.\scripts\Start-Local-ME3.ps1' -Device 'Fifine' -Brain stub
# verification without the microphone
& '..\..\.venv\Scripts\python.exe' -m me3.agent --check --load-models
```

## Layout
```
Dumalaog_ME3 - Voice Duplex/
├── config/config.yaml      # single source of truth
├── requirements.txt        # pinned SOTA versions
├── install.sh              # idempotent Pi installer
├── deploy/                 # systemd user units (asr/tts/vad/agent)
├── data/DATASET.md         # class-group shared benchmark dataset spec
├── launchers/              # Windows quick launch (ME2 pattern)
├── scripts/                # LOCAL-DEV pipeline (ME2 pattern) - run on Windows
│   ├── Start-Local-ME3.ps1 # 0. launch the voice agent locally
│   ├── dev_env_check.py    # 1. verify shared .venv has all ME3 deps
│   ├── local_smoke_test.py # 2. import+health all 4 MCP servers, no Pi
│   ├── package_release.py  # 3. build dist/ release + sha256 manifest + zip
│   └── deploy_to_pi.py     # 4. stage Desktop + scp to Pi + run install.sh
├── deployment/             # deployment STATE (read these first)
│   ├── latest_release.json     # last PREPARED release (local only)
│   └── last_pi_deployment.json # last Pi sync: never/pending/failed/deployed
├── refs/                   # benchmark references (Full-Duplex-Bench paper + repo)
├── me3/
│   ├── config.py           # config loader
│   ├── mcp_asr.py          # ASR MCP server (faster-whisper)
│   ├── mcp_tts.py          # TTS MCP server (Piper 1.8.0)
│   ├── mcp_vad.py          # VAD MCP server (Pipecat-bundled Silero ONNX)
│   ├── mcp_tasks.py        # real-world task tools (+ ME2 bridge)
│   ├── agent.py            # full-duplex control + local voice loop + Pipecat
│   ├── audio_io.py         # full-duplex mic/speaker (barge-in)
│   ├── benchmark.py        # legacy FDB-v1 + latency metric helpers
│   ├── fdb_v15.py          # paper-native v1.5 contract/transforms
│   ├── record_dataset.py   # record the 8 canonical stimuli
│   └── verify_dataset.py   # validate dataset standard
├── docs/PAPER_ALIGNMENT.md # paper scope, evidence boundary, and Pi path
├── docs/ME3_MANUAL_ELI5_AND_DEEP.md # ELI5 + deep technical manual
├── docs/acm/ME3_Voice_Duplex_ACM.tex # ACM two-column LaTeX source
├── docs/acm/ME3_Voice_Duplex_ACM.pdf # ACM two-column paper
├── docs/presentation/ME3_Voice_Duplex_Presentation_v4.pptx # presentation deck
├── FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf # user source paper
├── Building AI Voice Agents - Summary.pdf           # user source paper
├── models/                 # downloaded models (piper voice); git-ignored
└── runs/                   # local run logs; git-ignored
```

## Local development before Pi deployment (the ME2 pattern)

ME3 is built and validated on the Windows dev box first, then shipped to the
Pi. **Local success is not proof of Pi deployment.** Four steps:

```powershell
cd "C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME3 - Voice Duplex"
$py = "..\..\.venv\Scripts\python.exe"

& $py scripts\dev_env_check.py            # 1. deps present in shared .venv?
& $py scripts\local_smoke_test.py --with-audio  # 2. all 4 MCP servers healthy?
& $py scripts\check_paper_alignment.py          # source contract + Pi release check
& $py scripts\package_release.py          # 3. -> deployment/latest_release.json (prepared)
& $py scripts\deploy_to_pi.py --dry-run   # 4a. preview the scp/ssh steps
& $py scripts\deploy_to_pi.py             # 4b. real sync (needs Pi reachable)
```

If the Pi is unreachable, step 4 records `pending_sync` in
`deployment/last_pi_deployment.json`, preserves the prepared release, and
returns failure. Retry later (or double-click `~/Desktop/me3_voice_duplex\Deploy-ME3.bat`).

## Quick start (on the Pi)
```bash
bash install.sh
source .venv/bin/activate
python -m me3.agent                 # run the voice agent (same local loop)
python -m me3.bench_driver --brain stub --smoke # engineering loopback smoke (input required)
# paper-native evaluation of captured overlap JSON:
python scripts/fdb_v15_eval.py runs/fdb_v15/trials.json
```

## Functional Requirements Checklist
| # | Requirement | Status | Where |
|---|---|---|---|
| FR1 | AI agent harness with MCP support on RPi | ✅ scaffolded | `me3/agent.py` (Pipecat) + 4 MCP servers |
| FR2 | MCP server for SOTA ASR | ✅ built | `me3/mcp_asr.py` (faster-whisper 1.2.1) |
| FR3 | MCP server for SOTA TTS | ✅ built | `me3/mcp_tts.py` (Piper 1.8.0) |
| FR4 | VAD installed on RPi for voice detection | ✅ built | `me3/mcp_vad.py` (Pipecat-bundled Silero ONNX model) |
| FR5 | Full-duplex agent (simultaneous in/out) | ✅ local loop built | `me3/agent.py` LocalVoiceLoop + `audio_io.py` |
| FR6 | Interruptible / barge-in | ✅ local loop built | barge-in in the local loop + `VoiceDuplexAgent` |
| FR7 | Conversational, natural speech | ✅ config | `config.yaml` system_prompt + TTS |
| FR8 | Real-world tasks | ✅ built | `me3/mcp_tasks.py` (timers/sysinfo/files/cmd/ME2) |
| FR9 | We build+run our MCP servers (not just call APIs) | ✅ | all 4 servers are our code |
| FR10 | Engineering latency benchmark (classwork) | ✅ built locally | `me3/bench_driver.py` (not a v1.5 result) |
| FR11 | Class-group shared dataset | ✅ spec+tools | `data/DATASET.md`, record/verify scripts |
| FR12 | Deploy/autostart on Pi | ✅ built | `deploy/*.service` + `install.sh` |
| FR13 | LLM brain (local or commercial API) | ✅ config | `config.yaml` llm.remote/local |
| FR14 | Local-dev-before-deploy pipeline (ME2 pattern) | ✅ built | `scripts/` + `deployment/` state files |
| FR15 | Local quick launch like ME2 | ✅ built | `launchers/Start-ME3.bat` + `scripts/Start-Local-ME3.ps1` |
| FR16 | Agentic tool calling (build real artifacts from voice) | ✅ verified | `me3/brain.py` (Qwen tool loop) + `me3/tools.py` (19 tools) + `me3/task_runner.py` |

## Tool brain (agentic capabilities)

The Qwen brain (`me3/brain.py`) is a full OpenAI-compatible function-calling
loop against the vLLM endpoint. Given a voice request it can plan, call tools,
inspect results, and iterate — no training involved, pure inference + local
tool execution. Verified end-to-end on the dev box (2026-10-25): Qwen created
`agent_workspace/voice_webapp/index.html`, and a later Qwen verification
pass listed and read `index.html`, `styles.css`, and `app.js`.

Available tools (19): file ops (`list_files`, `read_file`, `write_file`,
`search_text`), build/test (`run_command`, `start_background_task`,
`task_status`, `task_output`, `cancel_task`), web serving
(`start_web_server`, `stop_web_server`, `web_servers`), research
(`fetch_url`, `web_search`), system (`system_info`, `git_status`), and
timers (`set_timer`, `list_timers`, `cancel_timer`).

Safety: all file paths are confined to the ME3 project; commands run
tokenized (no shell) against an allowlist; secret-looking files are refused;
tool output never contains environment variables or key material.

Requirements: `VLLM_API_KEY` must be present in the runtime environment for
remote Qwen calls. The agent never writes or prints the key. Without it the
agent falls back to the stub brain and the voice pipeline still runs.

## Pi deployment (tool brain key)

The vLLM key lives on the Windows dev box as a DPAPI blob
(`~/.codex/vllm-api-key.dpapi`), which the Pi cannot decrypt. One-time setup:

```powershell
# on Windows: provision the key through the approved secure operator workflow.
# Do not print it, paste it into chat, or commit it.
```

```bash
# on the Pi: make the key available to install.sh through the approved
# operator-only secret channel, then run the installer. The installer writes
# ~/.config/me3/me3.env with mode 600 and does not echo the value.
bash install.sh
```

Then `systemctl --user enable --now me3-agent` runs the full tool-capable
agent. `scripts/deploy_to_pi.py` handles the scp/unzip/install flow and
records `deployment/last_pi_deployment.json` (pending_sync if the Pi is
unreachable; only `deployed` counts). The currently verified deployment is
`/home/dalmacio/Desktop/kiko/me3_voice_duplex`; the receipt records the
release ID, the verified transfer hash, and the verification list. The current
receipt is `deployed`: `--check` passes on the Pi and a live text-mode Qwen
run built a stopwatch web app through `write_file` + `start_web_server` and
served it over HTTP, cross-verified from a second machine on the LAN. The Pi currently reports zero
PortAudio devices, so the user service remains disabled pending hardware
audio validation.

## Deployment state (how to tell where things stand)
| File | Meaning |
|---|---|
| `deployment/latest_release.json` → `status: prepared` | A release is packaged + sha256-checked **locally**. Not on the Pi yet. |
| `deployment/last_pi_deployment.json` → `never_deployed` | No Pi sync attempted. |
| `…` → `pending_sync` | Pi was unreachable; release preserved, retry later. |
| `…` → `sync_failed` | A scp/ssh step failed; `stderr` captured. |
| `…` → `deployed` | Pi received the release and ran `install.sh`. **Only this counts as deployed.** |

## What's implemented vs. what needs the Pi
- **Implemented (runs locally, verified):** all 4 MCP servers, Pipecat-bundled
  CPU ONNX VAD,
  full-duplex local voice loop with barge-in (`python -m me3.agent`),
  benchmark protocol + reporting, v1.5 contract/evaluator, dataset
  record/verify, installer, systemd units, local launcher.
- **Needs the physical Pi to finish:** live Pi audio device indices,
  systemd supervision, Pi-side v1.5 overlap captures, and the Pipecat
  transport wiring (the local loop is the reference implementation;
  `VoiceDuplexAgent` keeps the harness-agnostic control logic for the Pipecat
  pipeline). These are environment-bound.

The exact source-to-code boundary is recorded in
[`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md). In particular, the
five-scenario engineering driver must not be reported as the paper's four
scenario, 499-utterance v1.5 evaluation.

## Known limitations (local run)
- No acoustic echo cancellation: on loud speakers the mic may pick up the
  agent's own voice (use headphones). AEC is a Pi-hardware task.
- The remote LLM endpoint (`config.yaml`) is plain HTTP; set `VLLM_API_KEY`
  in the environment for the Qwen brain, or run with the stub brain.
- The LLM turn is processed while the mic is open, but no barge-in happens
  *during* ASR/LLM (only during TTS playback) in the v1 local loop.

## Multi-agent ownership (codex / antigravity / onit)
- **OnIt (this agent):** scaffold, MCP servers, benchmark, plan, integration.
- **Codex:** local voice loop + Pipecat transport wiring + audio tuning.
- **Antigravity:** UI/observability (live latency dashboard) + dataset recording.
