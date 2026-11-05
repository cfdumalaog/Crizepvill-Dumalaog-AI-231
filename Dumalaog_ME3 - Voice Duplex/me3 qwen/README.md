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
while the agent speaks. Playback is protected by default: ordinary speech is
buffered as the newest prompt, while explicit voice controls (`stop
processing`, `skip`, or `steer`) control the active response. `Ctrl+C` or
closing the window stops it. See [launchers/README.md](launchers/README.md).

This is the primary ME3 interaction. There is no Send button: the loop listens
continuously, Silero VAD detects the start and end of a turn, Whisper transcribes
the captured utterance internally, Qwen or GPT-5.6 Luna may execute tools automatically, and
the reply is spoken. New speech while ME3 is speaking is captured continuously;
the newest ordinary prompt waits for a safe audio boundary, while explicit
voice controls stop or redirect playback immediately. The browser UI below is
a secondary text and diagnostic console for inspection; it is not the
definition of the voice agent.

### Two selectable audio variants

ME3 has one shared duplex/tool architecture and two audio-provider profiles:

* `current` (default): local faster-whisper ASR plus one fixed Piper voice,
  `en_GB-alba-medium`, for every spoken event.
* `qwen_instruct`: local Silero VAD plus the supplied DGX3 vLLM services,
  `Qwen2-Audio-7B-Instruct` for ASR and `Qwen3-TTS-12Hz-1.7B-CustomVoice`
  for TTS. The existing Qwen text ToolBrain, MCP tools, web-app creation,
  channel safety gates, and protected-playback control state machine are shared.

The second profile is plug-and-play: the protected handoff file is read only
by the launcher into process memory. If the VPN or DGX3 listeners are not
reachable, `current` remains usable and the Qwen profile reports a bounded
provider error rather than silently claiming that audio succeeded.

The live browser UI labels the selected project variant in the header and
runtime card: the `me3 own` copy displays **ME3 FASTER-WHISPER + PIPER**, while
the `me3 qwen` copy displays **ME3 QWEN INSTRUCT**. The internal folder and
launcher identifiers remain `me3 own` and `current` for compatibility; the
user-facing name now describes the actual voice stack: multilingual
faster-whisper `small` ASR plus Piper `en_GB-alba-medium` TTS. It is not a
separately trained ME3 model. The text/tool brain is shared Qwen3.8-27B by
default, with GPT-5.6 Luna selectable as the alternative.

### Protected playback and the runtime terminal

The default live profile uses `barge_in: false` and `command_control: true`.
This is deliberate: the microphone remains open while TTS speaks, but an
ordinary overlapping sentence does not cut off a response. ME3 transcribes it
on the control path and retains only the newest pending prompt for the next
safe audio boundary. Say **stop processing** to stop playback and suppress
queued announcements, **skip** to skip the current audio, or **steer** to stop
the current audio and prioritize the newest captured prompt. Accepted long
tool tasks cannot be force-killed from the audio thread; their side effects
continue unless the normal task-cancellation tool is used.

The live `web/assistant.html` page includes a **Runtime terminal** panel. It
shows timestamped, sanitized operational events: state transitions, ASR/TTS
activity, tool names/status, and selected model/voice metadata. It never shows
hidden chain-of-thought, tool arguments, credentials, or raw tool results.
The panel is fed by `/api/state` and remains useful over the Pi LAN URL even
when the browser is only the speaker sink.

```powershell
# Existing/current audio path
& '.\scripts\Start-Local-ME3.ps1' -Brain remote -Variant current

# Qwen Instruct ASR/TTS path; launcher auto-reads the protected handoff file
& '.\scripts\Start-Local-ME3.ps1' -Brain remote -Variant qwen_instruct

# Same variant in the localhost diagnostic UI
& '.\scripts\Start-ME3-UI.ps1' -Variant qwen_instruct -OpenBrowser

# Matched, safe provider comparison (no transcript/audio bytes are persisted)
& '..\..\.venv\Scripts\python.exe' scripts\compare_audio_variants.py `
  --secret-file '..\..\Secrets New ME3 Model Qwen Instruct.txt' --timeout 10
```

The live loop exposes the Vocalis-Pi contract in
[docs/VOCALIS_PROTOCOL.md](docs/VOCALIS_PROTOCOL.md): VAD and transcript
events continue while TTS plays, a barge-in emits queue-flush/fade controls,
long tool turns can use a backchannel, and sanitized tool progress is available
to a background harness adapter. The verified default remains the bounded local
Qwen tool loop; GPT-5.6 Luna is an optional OpenAI-compatible alternative, and
the external harness dispatch is an explicit
MCP/JSON-RPC seam rather than an unconfigured network dependency.

```powershell
# or from PowerShell, with options
& '.\scripts\Start-Local-ME3.ps1' -Device 'Fifine' -Brain stub
# Split-device acceptance test: Pi Fifine mic -> this computer's speakers
& $py scripts/run_me3_module.py me3.agent --brain remote --ssh-mic 'dalmacio@cfdfnjrpi5.local' `
    --remote-device 'hw:2,0'
# verification without the microphone
& '..\..\.venv\Scripts\python.exe' scripts/run_me3_module.py me3.agent --check --load-models
```

The `--ssh-mic` lane is a local acceptance path for the hardware arrangement
where the Fifine microphone is attached to the Pi but the operator wants to
hear TTS through this workstation's default speakers. It streams raw 16-kHz
PCM from the Pi's `arecord` process over the existing SSH connection and keeps
the normal VAD -> ASR -> Qwen/tool -> TTS state machine and local barge-in
behavior. It does not replace the production Pi transport: a Pi deployment
still expects its own output device, while the injection-only evaluator below
remains the speaker-free evidence lane.

## Pi web server and browser speaker (ME2-style)

The Pi release includes a combined HTTP/WebSocket server. The Pi keeps the
Fifine/USB microphone as the input source, runs VAD -> ASR -> tool reasoning ->
TTS on the Pi, and can stream the generated 16-kHz PCM to the browser that is
viewing the UI. This is the intended path when the operator wants the sound to
play through the PC while the microphone remains attached to the Pi.

```bash
# On the Pi, from the active /home/dalmacio/Desktop/dandan release:
python scripts/me3_pi_server.py --host 0.0.0.0 --port 8126 --audio-port 8127 \
  --brain remote --variant current
```

Open `http://<PI-IP>:8126/` from a PC on the same network. In the **Pi audio
routing** card, select the Pi microphone and either **This browser / connected
PC speaker** or a Pi output device, apply the route, and click **Connect
browser speaker** once. The browser speaker dropdown selects the PC's local
audio output when the browser supports `AudioContext.setSinkId`. The same
server is installed by `deploy/me3-agent.service`; the Windows ME2-style
launcher `launchers/Start-ME3-Pi-Web.bat` starts it and forwards both HTTP and
audio ports through SSH.

The browser PCM transport permits exactly one active audio tab. A new ME3 tab
replaces a stale one and long responses are sent as 30-ms PCM chunks so the
WebSocket keepalive remains responsive. This is intentional: multiple stale
tabs have independent Web Audio clocks, and their combined acoustic output
cannot be matched by one microphone echo reference. Browser-native
`speechSynthesis` is disabled in both the live assistant and diagnostic UI;
the selected profile TTS is the only response voice.

For the Qwen Instruct audio deployment, provision the protected audio endpoint
values once after the first sync (the script reads the Windows handoff only in
memory and never prints the key):

```powershell
& '.\scripts\Provision-Qwen-Pi.ps1' -UseReverseTunnel
```

The Pi launcher uses an SSH reverse tunnel for the Qwen Audio services because
the audio DGX address (`10.153.60.233`) is reachable from this workstation's
VPN but not directly from the Pi. The named Qwen launcher forwards remote Pi
ports 18000/18001 through this workstation to DGX 8000/8001. If the serving
address changes, pass it explicitly with `-DgxIp` in direct mode. Use
`-UsePiBrainCredential` only when the routed endpoint accepts the Pi's existing
Qwen brain credential.

The named `launchers\Start-ME3-Pi-Qwen.bat` launcher then loads those values
from the dedicated mode-600 `~/.config/me3/qwen-instruct.env` and starts Qwen
on its separate 8136/8137 ports. The Own deployment remains on 8126/8127.

If the Fifine appears in the UI but speech is not triggering, check the USB
capture control on the Pi. The current Fifine card is ALSA card 2; its capture
switch must be on and its capture level should normally be 100%:

```bash
amixer -c 2 get Mic
amixer -c 2 sset Mic 100% cap
```

The UI's VAD meter and RMS value are the authoritative live signal indicators.
An RMS value around the low hundreds is usually room noise or a distant/low
hardware-gain microphone; speak close to the capsule and increase the Fifine's
physical gain before lowering the Silero threshold. After changing routing or
the mixer, restart the ME3 audio loop and click **Connect browser speaker**
again because browsers close WebSocket audio on service restarts.

The LAN server is intentionally not public hosting. Use the SSH tunnel or a
firewall/VPN boundary for remote access, and do not expose the raw audio port
to the Internet.

## ME2 relationship

See [`docs/ME2_VS_ME3.md`](docs/ME2_VS_ME3.md) for the precise distinction.
ME2 is a trained, offline, finite-intent TinyDSCNN command model; ME3 is an
inference-first, open-ended full-duplex agent. ME3 preserves ME2's Pi/browser
operation pattern but adds continuous listening, barge-in, API-backed ASR/TTS,
provider-neutral tool calling, MCP servers, and benchmarked turn-taking.

## Layout
```
Dumalaog_ME3 - Voice Duplex/
├── config/config.yaml      # single source of truth
├── requirements.txt        # pinned SOTA versions
├── install.sh              # idempotent Pi installer
├── deploy/                 # systemd user units (asr/tts/vad/agent)
├── data/DATASET.md         # class-group shared benchmark dataset spec
├── launchers/              # Windows quick launch (ME2 pattern)
│   ├── Start-Pi-Injection-Demo.bat # one-click deployed Qwen/tool demo
│   └── Start-ME3-UI.bat        # localhost Qwen/tool test console
├── scripts/                # LOCAL-DEV pipeline (ME2 pattern) - run on Windows
│   ├── Start-Local-ME3.ps1 # 0. launch the voice agent locally
│   ├── Start-ME3-UI.ps1 # localhost interactive UI on port 8124
│   ├── Start-ME3-Web.ps1 # local browser preview on port 8123
│   ├── Start-ME3-Pi-Web.ps1 # ME2-style SSH browser tunnel to Pi port 8090
│   ├── dev_env_check.py    # 1. verify shared .venv has all ME3 deps
│   ├── local_smoke_test.py # 2. import+health all 4 MCP servers, no Pi
│   ├── compare_audio_variants.py # current vs Qwen Instruct ASR/TTS metrics
│   ├── me3_pi_server.py    # Pi LAN UI + browser PCM speaker server
│   ├── me3_fdbench_agent.py # direct-feed adapter for fd-benchmark
│   ├── run_fdbench.py      # shared-environment benchmark launcher
│   ├── pi_audio_preflight.py # Pi ALSA + PortAudio read-only gate
│   ├── workstation_acceptance.py # 3. local presentable-output gate
│   ├── package_release.py  # 4. build dist/ release + sha256 manifest + zip
│   └── deploy_to_pi.py     # 5. accepted bundle only: scp + install.sh
├── deployment/             # deployment STATE (read these first)
│   ├── latest_release.json     # last PREPARED release (local only)
│   └── last_pi_deployment.json # last Pi sync: never/pending/failed/deployed
├── refs/                   # benchmark references (Full-Duplex-Bench paper + repo)
├── me3/                    # this folder's importable variant package
├── me3_bootstrap.py        # selects this package before importing me3
├── benchmarking/fd-benchmark/ # vendored MIT FDB v1/v1.5 benchmark
├── docs/PAPER_ALIGNMENT.md # paper scope, evidence boundary, and Pi path
├── docs/VOCALIS_PROTOCOL.md # full-duplex input/output event contract
├── docs/ME3_MANUAL_ELI5_AND_DEEP.md # ELI5 + deep technical manual
├── docs/ME3_QWEN_INSTRUCT_SYSTEM_FLOW.svg # full layer-by-layer process infographic
├── docs/ELEVENLABS_INTEGRATION.md # Qwen voice backend and Realtime decision
├── docs/CHANNEL_INTEGRATIONS.md # call, SMS/text, email API requirements
├── docs/acm/ME3_Voice_Duplex_ACM.tex # ACM two-column LaTeX source
├── docs/acm/ME3_Voice_Duplex_ACM.pdf # ACM two-column paper
├── docs/presentation/ME3_Voice_Duplex_Presentation_v9_Luna.pptx # presentation deck
├── docs/PHILIPPINE_CALL_CENTER_PILOT.md # Philippines-market pilot and handoff manual
├── docs/DEMO_RUNBOOK.md # classroom demo sequence and evidence boundaries
├── FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf # user source paper
├── Building AI Voice Agents - Summary.pdf           # user source paper
├── models/                 # downloaded models (piper voice); git-ignored
└── runs/                   # local run logs; git-ignored
```

## Workstation-first finalization before Pi deployment

ME3 is finalized on the Windows dev box first, then the exact accepted bundle
is shipped to the Pi. **A package cannot be deployed unless the workstation
acceptance record matches its release ID, source fingerprint, and archive
hash.** Five steps:

```powershell
cd "C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME3 - Voice Duplex"
$py = "..\..\.venv\Scripts\python.exe"

& $py scripts\dev_env_check.py            # 1. deps present in shared .venv?
& $py scripts\local_smoke_test.py --with-audio  # 2. all 4 MCP servers healthy?
& $py scripts\check_paper_alignment.py          # source contract + Pi release check
& $py scripts\workstation_acceptance.py    # 3. tests + docs + package + acceptance record
& $py scripts\deploy_to_pi.py --dry-run    # 4. preview the accepted transfer
& $py scripts\deploy_to_pi.py              # 5. real sync (needs Pi reachable)
```

The acceptance gate is local-only; it does not contact the Pi or place live
channel side effects. If the Pi is unreachable, the final step records `pending_sync` in
`deployment/last_pi_deployment.json`, preserves the prepared release, and
returns failure. Retry later (or double-click `~/Desktop/me3_voice_duplex\Deploy-ME3.bat`).

The complete policy and post-deployment hardware gate are in
[`docs/WORKSTATION_FIRST_RELEASE.md`](docs/WORKSTATION_FIRST_RELEASE.md).

## Quick start (on the Pi)
```bash
bash install.sh
source .venv/bin/activate
python scripts/run_me3_module.py me3.agent                 # run the voice agent (same local loop)
python scripts/run_me3_module.py me3.bench_driver --brain stub --smoke # engineering loopback smoke (input required)
python scripts/pi_audio_preflight.py              # Pi ALSA + PortAudio gate
python scripts/injection_eval.py \
  --input-wav runs/injection_eval_inputs/tell_me_system_status.wav \
  --interrupt-wav runs/injection_eval_inputs/stop.wav \
  --brain remote                                  # virtual speaker sink
python scripts/run_me3_module.py me3.bench_driver --brain stub --preflight # calibration only, no trials
# paper-native evaluation of captured overlap JSON:
python scripts/fdb_v15_eval.py runs/fdb_v15/trials.json
```

Windows one-click equivalent for the remote Qwen/tool injection demo:
`launchers\Start-Pi-Injection-Demo.bat`.

### Injection-only evaluation when the speaker is unavailable

`scripts/injection_eval.py` feeds a mono 16-bit WAV into the same real-time
VAD -> ASR -> Qwen/tool -> TTS state machine used by the voice agent. It adds
the configured VAD silence tail, keeps wall-clock frame cadence, and uses a
virtual discard sink for TTS. The optional interrupt WAV is injected while
virtual TTS is active, so barge-in can be evaluated without opening a speaker
device:

```bash
python scripts/injection_eval.py \
  --input-wav runs/injection_eval_inputs/tell_me_system_status.wav \
  --interrupt-wav runs/injection_eval_inputs/stop.wav \
  --brain remote
```

Artifacts are written under `runs/injection_eval/<UTC-stamp>/`. They include
the injected-input mode, tool-call names, transcripts/replies, virtual-output
events, barge-in status, and safe PortAudio topology. The report explicitly
sets `physical_speaker_exercised=false` and does not count as physical-speaker
or paper-native FULL-DUPLEX-BENCH v1.5 evidence. The input WAVs are ordinary
test fixtures and must not contain credentials.

For a presenter-friendly sequence with a status scenario, web-app scenario,
and physical-audio gate, use [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md).

### Browser access like ME2

ME3 now has an explicit browser path. On Windows, double-click
`launchers\Start-ME3-Web.bat` to serve the tracked `web/` presenter surface at
`http://127.0.0.1:8123/`; pass `-Path agent_workspace\voice_webapp` when you
want to serve a generated Qwen web app. The launcher also
prints LAN URLs for another device on the same network. For the Pi, double-click
`launchers\Start-ME3-Pi-Qwen.bat` to start/check the named Qwen Instruct
deployment at `/home/dalmacio/Desktop/dandan-qwen` and forward it through SSH
to `http://127.0.0.1:8100/`, matching the ME2 tunnel experience. The Own
variant has its own `Start-ME3-Pi-Own.bat` at `/home/dalmacio/Desktop/dandan`
and `http://127.0.0.1:8090/`. The deployments are isolated, but only one
full-duplex listener should own the physical Fifine capture device at a time;
launch one voice variant at a time unless separate input devices are selected.
Both paths are local-network/tunnel access, not public Internet hosting.

### Interactive local UI (diagnostic console)

For local model and tool testing on this workstation, use
`launchers\Start-ME3-UI.bat` or:

```powershell
& '.\scripts\Start-ME3-UI.ps1' -OpenBrowser
```

This serves `web/ui.html` at `http://127.0.0.1:8124/` and calls the same
provider-neutral ToolBrain used by the agent. Select **Qwen + tools**,
**GPT-5.6 Luna + tools**, **Offline stub**, or **Auto**. Auto preserves Qwen as
the first choice and uses Luna only when Qwen is not configured. GPT-5.6 Luna
uses model ID `gpt-5.6-luna` with hidden reasoning disabled and direct response
streaming enabled when supported; it requires
`OPENAI_API_KEY` and is text-only, so ASR and TTS remain ME3's separate local
pipeline. The browser voice listener starts automatically after microphone
permission and submits detected utterances without a voice button. While a turn
is running, **Show task progress (not model reasoning)** displays the current task plus safe live
progress stages such as model inference, tool planning, tool execution, and
completion. It does not expose private chain-of-thought or credentials. The UI
also shows model timing, tool identities, channel readiness, and safe errors.
The authoritative physical
audio path remains `Start-ME3.bat`; calls, SMS, and email are hard-disabled in
this console, so the unavailable Twilio account is not a blocker for local model
evaluation.

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
| FR13 | LLM brain (local or commercial API) | ✅ config | `config.yaml` `llm.remote` Qwen + `llm.openai` GPT-5.6 Luna |
| FR14 | Local-dev-before-deploy pipeline (ME2 pattern) | ✅ built | `scripts/` + `deployment/` state files |
| FR15 | Local quick launch like ME2 | ✅ built | `launchers/Start-ME3.bat` + `scripts/Start-Local-ME3.ps1` |
| FR16 | Agentic tool calling (build real artifacts from voice) | ✅ verified | `me3/brain.py` provider-neutral Qwen/Luna tool loop + `me3/tools.py` (23 tools) + `me3/task_runner.py` |
| FR17 | Optional ElevenLabs adapter | ✅ implemented, inactive in the single-voice demo path | `me3/elevenlabs_tts.py`; active profiles stay on one fixed voice |
| FR18 | Outbound voice-call integration | ✅ implemented, credential-gated | `me3/channels.py` (`make_call`) + Twilio Calls API; requires reviewed HTTPS TwiML/media bridge |
| FR19 | SMS/text integration | ✅ implemented, credential-gated | `me3/channels.py` (`send_sms`) + Twilio Messages API |
| FR20 | Transactional email integration | ✅ implemented, credential-gated | `me3/channels.py` (`send_email`) + Resend Send Email API |
| FR21 | Optional Qwen Instruct audio variant | ✅ live ASR/TTS + shared tool loop verified over VPN | `me3/qwen_audio.py`, `--variant qwen_instruct`, `scripts/compare_audio_variants.py` |

## Tool brain (agentic capabilities)

The complete Qwen Instruct path is shown in the [ME3 Qwen Instruct system-flow
infographic](docs/ME3_QWEN_INSTRUCT_SYSTEM_FLOW.svg). It covers the Pi Fifine
capture path, echo guard, Silero endpointing, Qwen2-Audio transcription,
streaming Qwen ToolBrain, owned MCP servers, confirmation-gated channels,
Qwen3-TTS Aiden output, browser speaker transport, barge-in feedback,
deployment/VPN layers, safety controls, and observability. The Gmail session
opened in the operator's browser is intentionally documented as an authenticated
operator surface only; it is not yet a ME3 Gmail/Google Meet tool connector, so
future send/join actions must remain explicitly authorized and confirmation-gated.

The ToolBrain (`me3/brain.py`) is a provider-neutral OpenAI-compatible
function-calling loop. Qwen/vLLM remains the default; GPT-5.6 Luna can be
selected with `--brain luna` or from the browser UI. Given a voice request,
either backend can plan, call tools, inspect results, and iterate. No training
is involved: this is pure inference plus local tool execution. Verified
end-to-end on the dev box (2026-10-25): Qwen created
`agent_workspace/voice_webapp/index.html`, and a later Qwen verification
pass listed and read `index.html`, `styles.css`, and `app.js`.

Available tools (23): file ops (`list_files`, `read_file`, `write_file`,
`search_text`), build/test (`run_command`, `start_background_task`,
`task_status`, `task_output`, `cancel_task`), web serving
(`start_web_server`, `stop_web_server`, `web_servers`), research
(`fetch_url`, `web_search`), system (`system_info`, `git_status`), and
timers (`set_timer`, `list_timers`, `cancel_timer`). The channel tools are
`channel_health`, `send_sms`, `make_call`, and `send_email`; the latter three
require explicit confirmation and protected provider configuration.

Safety: all file paths are confined to the ME3 project; commands run
tokenized (no shell) against an allowlist; secret-looking files are refused;
tool output never contains environment variables or key material.

Requirements: `VLLM_API_KEY` must be present for Qwen calls, or
`OPENAI_API_KEY` must be present for GPT-5.6 Luna calls. The agent never writes
or prints either key. The `auto` selector prefers Qwen, then Luna, then the
stub brain. To force Luna:

```powershell
$env:OPENAI_API_KEY = '<protected runtime value>'
& '..\..\.venv\Scripts\python.exe' scripts/run_me3_module.py me3.agent --brain luna --text 'Use system_info and summarize the host.'
```

Keep the value in the protected environment; do not paste it into shell
history, the repository, or documentation. Luna's API model ID is
`gpt-5.6-luna`; ME3 does not send `reasoning_effort`, and the request uses
the direct response stream when the provider supports it. The voice loop speaks
only returned assistant content, never hidden reasoning.

## Qwen Instruct audio variant

`--variant qwen_instruct` preserves the same 23-tool ToolBrain and duplex
controller while replacing only the audio providers. Qwen2-Audio is served as
`Qwen/Qwen2-Audio-7B-Instruct` through vLLM's multimodal
`/v1/chat/completions` route: ME3 sends the WAV as a base64 `audio_url` data URL
with a transcript-only prompt. Qwen3-TTS is served as
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` through `/v1/audio/speech`.

The protected handoff is read in memory by the PowerShell launcher and mapped
to `QWEN_INSTRUCT_ASR_BASE_URL`, `QWEN_INSTRUCT_TTS_BASE_URL`, and
`QWEN_INSTRUCT_API_KEY`. The current local path remains available as
`--variant current`. On 31 October 2026 over the VPN, the matched fixture
passed with Qwen ASR at 431.9 ms and Qwen TTS at 851.6 ms; the same run passed
with faster-whisper ASR at 2732.0 ms and Piper TTS at 100.1 ms TTFB/176.7 ms
total. The full injected Qwen voice/tool turn passed with zero errors and
called five shared tools. The comparison harness records no credentials,
transcripts, endpoints, or audio bytes.

## Call, text, and email integrations

ME3 includes shared direct-provider and MCP tools for outbound Twilio calls,
Twilio SMS, and Resend transactional email. They are deliberately
credential-gated and confirmation-gated: the agent must identify the target
and content, receive explicit confirmation, and only then pass `confirm=true`.
The tools never return credential material or provider response bodies. The
complete account/API/webhook checklist is in
[`docs/CHANNEL_INTEGRATIONS.md`](docs/CHANNEL_INTEGRATIONS.md).

Current implementation status is “adapter tested, live side effect not yet
placed.” `channel_health` is network-free and reports only configured/missing
environment names. An outbound call starts a Twilio call using a reviewed
HTTPS TwiML URL; conversational inbound/outbound call-center behavior still
requires the operator's media-stream bridge, transfer workflow, consent,
recording policy, and CRM/identity integrations.

## Pi deployment (tool brain key)

## ElevenLabs voice backend

ME3 keeps Qwen as the reasoning/tool model and can render its responses with
ElevenLabs streaming PCM. The adapter is retained and documented for future
experiments, but the active demo profile deliberately does not auto-select it:
Own is locked to Piper `en_GB-alba-medium` and Qwen is locked to Qwen3-TTS
CustomVoice `aiden`. This preserves one voice for replies, backchannels, and
completion announcements while keeping the full-duplex microphone, VAD, ASR,
playback, and barge-in path. See
docs/ELEVENLABS_INTEGRATION.md.

The ElevenLabs CLI onboarding flow stores the local development key in the
gitignored project `.env`. ME3 loads that file locally without overriding
explicit process/service variables; the Pi continues to use its protected
`~/.config/me3/me3.env` runtime file. Do not commit or print either file.

The old auto/fallback behavior is kept only in the adapter tests; it is not
reachable from the active profile selector. `ME3_TTS_ENGINE`,
`ME3_TTS_VOICE`, and `ME3_ANNOUNCER_VOICE` cannot change the live profile voice.

The local provider path was exercised on 2026-10-26 using an operator-selected
test voice: ElevenLabs returned 16-kHz raw PCM and the Filipino-language test
phrase played through the workstation's default speaker. The runtime now uses
the multilingual Whisper model with automatic Filipino/English detection and
passes `language_code=fil` to ElevenLabs. A native Filipino/Philippine-English
accent still requires the operator to select and provision an approved voice
ID; that ID is never committed.

### Speaker feedback protection

The live audio transport keeps the microphone open during playback for true
barge-in, but it no longer forwards the raw speaker feedback directly to VAD.
`audio.echo_guard` in `config/config.yaml` searches a short playback-reference
delay window, subtracts the correlated speaker component, and keeps a short
room-reflection tail. On Windows `Stereo Mix`/digital loopback is recognized
and hard-muted during ME3 playback because it contains only system output;
the physical microphone path keeps correlation-based cancellation so unmatched
user speech can interrupt the agent. Headphones or a separated
speaker/microphone are still recommended for the cleanest physical test; this
software guard is not a substitute for hardware acoustic echo cancellation on
a Raspberry Pi. Browser-speaker output now registers its PCM with the same
capture-side echo guard, so the agent's own browser TTS is removed before VAD
and ASR instead of becoming a false new turn.

If the UI still shows the old **browser voice echo** checkbox, that tab loaded
an earlier build and can add a second operating-system voice (often a female
system voice) on top of Qwen Aiden. Reload the current page after deployment.
The corrected page has no checkbox, the runtime card shows the active TTS
backend/model/voice, and the audio card reports one active browser client.
Before the fix, a disconnected browser WebSocket also caused the transport to
skip reference registration; the Pi speaker then fed back into the Fifine and
opened repeated false turns. The current transport registers the reference
before publishing the first PCM chunk, chunks long responses, closes stale
clients, and clears queued browser audio on interruption.

### Barge-in does not cancel accepted work

When the operator starts speaking over ME3, the current TTS buffer is stopped
immediately so the new utterance can be heard. This is the expected
interruptible part of full duplex. It does not cancel a prompt that has already
entered the Qwen/tool worker: long-running work continues, and its completion
reply is placed in an announcement queue until the newer turn and its audio are
idle. An explicit `cancel_task`/`cancel_project_task` request is required to
cancel a background project task.

To verify the actual Windows loopback path, run:

```powershell
python scripts\loopback_selftest.py --input "Stereo Mix" --expect-echo-suppressed
```

This must report that Stereo Mix captured the agent output, the cleaned RMS is
near silence, and the output was suppressed before VAD/ASR. The raw ASR
loopback diagnostic remains available without the flag; it disables the guard
deliberately so that TTS-to-capture-to-ASR can be measured.

ElevenLabs Realtime v1 Mini/Max is a complete hosted voice-agent mode that also
manages the LLM and speech recognition. It is documented as an optional
alternative, but is not silently substituted because ME3's Qwen tool loop is
an explicit project requirement.

### Standardized Qwen voice output

The Qwen Instruct variant uses Qwen3-TTS CustomVoice `aiden` at 24 kHz as its
single standard spoken voice. The same configured `tts.announcer_voice` is
applied to ordinary replies, the immediate long-task backchannel, and the
completion announcement after a background tool task. Runtime logs label these
events as `role=response` or `role=announcer`; the voice itself remains
consistent. Voice environment overrides are intentionally ignored by the live
profile loader so a service restart cannot mix response and announcement voices.
The spoken policy is formal and professional: the fixed long-task
acknowledgement is `I am processing your request now.` and the assistant must
avoid flirtatious, intimate, suggestive, teasing, or overly familiar language.

The Qwen provider comparison is recorded in
[`runs/qwen_instruct_comparison_20261105_final.json`](runs/qwen_instruct_comparison_20261105_final.json): both Qwen ASR and Qwen TTS passed, with
60.8 ms ASR latency on the short fixture and 487.4 ms median TTS first-byte
latency across two runs. The full-duplex engineering smoke is recorded in
[`runs/fdb_benchmark/20261104-161639/summary.md`](runs/fdb_benchmark/20261104-161639/summary.md)
and its machine-readable results file. It produced 2/2 valid end-to-end trials
using Windows Stereo Mix loopback. The class benchmark driver sets
`ack_wait_s=0` by design, so its BC=0 row does not test the live long-task
backchannel; use a clean operator run with a deliberately long tool task for
that separate voice-role check. These artifacts are engineering/loopback
evidence, not physical Raspberry Pi audio or a paper-native FDB v1.5 result.

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
`/home/dalmacio/Desktop/dandan`; the receipt records the
release ID, the verified transfer hash, and the verification list. The current
receipt is `deployed`: `--check` passes on the Pi and a live text-mode Qwen
run built a stopwatch web app through `write_file` + `start_web_server` and
served it over HTTP, cross-verified from a second machine on the LAN. The
current Pi deployment has the Fifine USB capture card visible to ALSA and
PortAudio, with the browser-speaker loop verified at `loop=true`. The physical
microphone path has been exercised, but a clean human barge-in benchmark is
still a separate evidence gate. See [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md)
for the exact gate and evidence boundary.

### Pi audio preflight

Run this read-only gate before enabling the user service:

```bash
python scripts/pi_audio_preflight.py
python scripts/pi_audio_preflight.py --json
```

The report separates ALSA capture/playback enumeration from PortAudio
visibility. A Pi with HDMI playback but no capture device is not ready for
voice duplex; connect a USB/I2S microphone or configure an ALSA loopback and
rerun the gate. The Windows engineering driver also has a calibration-only
mode that must pass before any latency trials are started:

```bash
python scripts/run_me3_module.py me3.bench_driver --brain stub --preflight
```

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
  full-duplex local voice loop with barge-in (`python scripts/run_me3_module.py me3.agent`),
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
- Endpointing waits for **800 ms of continuous silence** before dispatching a
  voice turn. This is intentionally longer than the old 450 ms so a natural
  clause pause does not send the prompt prematurely; speaking while ME3 is
  talking still uses the barge-in path.
- Software echo suppression is implemented and the Windows Stereo Mix path is
  physically verified, but this is not hardware acoustic echo cancellation.
  A physical Pi microphone/speaker still needs a human barge-in check; use
  headphones or separated devices for the cleanest test.
- The remote LLM endpoint (`config.yaml`) is plain HTTP; set `VLLM_API_KEY`
  in the environment for the Qwen brain, or run with the stub brain.
- The LLM turn is processed while the mic is open, but no barge-in happens
  *during* ASR/LLM (only during TTS playback) in the v1 local loop.

## Multi-agent ownership (codex / antigravity / onit)
- **OnIt (this agent):** scaffold, MCP servers, benchmark, plan, integration.
- **Codex:** local voice loop + Pipecat transport wiring + audio tuning.
- **Antigravity:** UI/observability (live latency dashboard) + dataset recording.
