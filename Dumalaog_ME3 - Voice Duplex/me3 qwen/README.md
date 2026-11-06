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
while the agent speaks. The live Qwen profile uses automatic echo-protected
barge-in: overlap that matches the speaker reference is rejected as echo,
while genuine user speech stops the current TTS and starts the next turn;
explicit voice controls (`stop processing`, `skip`, or `steer`) remain
available. `Ctrl+C` or
closing the window stops it. See [launchers/README.md](launchers/README.md).

This is the primary ME3 interaction. There is no Send button: the loop listens
continuously, Silero VAD detects the start and end of a turn, Whisper transcribes
the captured utterance internally, Qwen or GPT-5.6 Luna may execute tools automatically, and
the reply is spoken. New speech while ME3 is speaking is captured continuously;
the echo guard rejects the agent's own output, while genuine overlap triggers
barge-in, flushes the current audio, and enters the new turn. Explicit voice
controls stop or redirect playback immediately.
The browser UI below is
a secondary text and diagnostic console for inspection; it is not the
definition of the voice agent.

### Single active audio profile: Qwen Instruct

The active ME3 release is **ME3 QWEN INSTRUCT**. It uses local Silero VAD plus
the supplied DGX3 vLLM services: `Qwen2-Audio-7B-Instruct` for ASR and
`Qwen3-TTS-12Hz-1.7B-CustomVoice` for TTS. The Qwen text ToolBrain, MCP tools,
web-app creation, channel safety gates, and protected-playback control state
machine are all part of this same release line.

The former local faster-whisper/Piper Own deployment has been archived and is
not an active launch target. The internal `current` profile remains only as a
legacy compatibility option for old comparison scripts; new launchers,
defaults, Pi service files, and the UI use `qwen_instruct`. It is not a
separately trained ME3 model. The text/tool brain is shared Qwen3.8-27B by
default, with GPT-5.6 Luna selectable as the alternative.

The Qwen handoff file is read only by the launcher into process memory. Qwen
ASR remains the primary provider. The Pi also has an explicit availability
failover: when the Qwen ASR listener is unavailable, cached multilingual
faster-whisper `base` transcribes locally; the Runtime card labels this as
`faster_whisper_fallback`. Qwen3-TTS remains the only spoken output voice. If
both ASR paths fail, the profile reports a bounded provider error rather than
silently claiming that audio succeeded. Fallback transcripts are functional
demo evidence, not Qwen2-Audio benchmark evidence.

### Full-duplex playback and the runtime terminal

The default live profile uses `barge_in: true` and `command_control: true`.
The microphone remains open while TTS speaks. Echo-protected VAD distinguishes
the agent's own output from genuine user speech; genuine overlap stops the
current TTS, flushes the playback queue, and starts the new turn. Say
**stop processing** to stop playback and suppress queued announcements,
**skip** to skip the current audio, or **steer** to stop the current audio and
prioritize the newest captured prompt. Accepted long tool tasks cannot be
force-killed from the audio thread; their side effects continue unless the
normal task-cancellation tool is used.

The live `web/assistant.html` page includes a **Runtime terminal** panel. It
shows timestamped, sanitized operational events: state transitions, ASR/TTS
activity, tool names/status, and selected model/voice metadata. It never shows
hidden chain-of-thought, tool arguments, credentials, or raw tool results.
The panel is fed by `/api/state` and remains useful over the Pi LAN URL even
when the browser is only the speaker sink.

Qwen Instruct uses direct streaming with `fast_mode: true` and
`thinking: false`. The request includes Qwen/vLLM's
`chat_template_kwargs.enable_thinking=false`, so hidden reasoning tokens are
not generated on the voice path. When a model/tool turn exceeds the immediate
acknowledgement window, the **Background processes** panel shows the live
agent thread and bounded project subprocesses (PID, elapsed time, stage,
command, and redacted output tails). ME3 speaks a formal progress update at
most once per minute while the task remains active, then announces the final
result at a safe playback boundary.

For the Pi microphone-to-laptop-speaker arrangement, the echo guard keeps the
exact browser PCM reference, retimes it using a browser render-start
acknowledgement, and fits a bounded multi-tap acoustic path to cover speaker
delay, room reflections, and USB-clock drift. A speaker-only frame is gated
when the residual collapses; unmatched user speech is retained for the
protected `stop processing`, `skip`, and `steer` control path. The guard
exposes only counters and correlation metrics through `/api/audio/devices`; it
does not claim a human acoustic acceptance test until one is run in the actual
room.

```powershell
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
# On the Pi, from the active /home/dalmacio/Desktop/dandan-qwen release:
python scripts/me3_pi_server.py --host 0.0.0.0 --port 8126 --audio-port 8127 \
  --brain remote --variant qwen_instruct
```

Open `http://<PI-IP>:8126/` from a PC on the same network. In the **Pi audio
routing** card, select the Pi microphone and either **This browser / connected
PC speaker** or a Pi output device, apply the route, and click **Connect
browser speaker** once. The browser speaker dropdown selects the PC's local
audio output when the browser supports `AudioContext.setSinkId`. The same
server is installed by `deploy/me3-agent.service`; the Windows ME2-style
launcher `launchers/Start-ME3-Pi-Web.bat` starts it and forwards both HTTP and
audio ports through SSH.

For the current deployment, use the Windows SSH alias:

```powershell
ssh me3-pi
```

It resolves to `dalmacio@192.168.254.105`. `localhost` is this PC, not the Pi.
The live Pi page serves `/`, `/roadmap`, and `/console` from the same origin;
navigation deliberately uses relative links so a browser opened through a
tunnel does not jump to the viewer's unavailable `127.0.0.1:8124`.
The `/console` path is retained only as a backward-compatible alias; the
authoritative operator page is the Assistant page, where Runtime terminal is
an in-page safe-events panel rather than a second terminal application. The
audio card has two deliberately different restart controls: **Restart ME3
loop** resets capture/TTS while keeping the UI and browser audio hub alive;
**Restart ME3 service** is Pi-only and restarts the canonical
`me3-agent.service` systemd user unit.

The browser PCM transport permits exactly one active audio tab. A new ME3 tab
replaces a stale one and long responses are sent as 30-ms PCM chunks so the
WebSocket keepalive remains responsive. This is intentional: multiple stale
tabs have independent Web Audio clocks, and their combined acoustic output
cannot be matched by one microphone echo reference. Browser-native
`speechSynthesis` is disabled in both the live assistant and diagnostic UI;
the selected profile TTS is the only response voice.

For the Qwen Instruct audio deployment, install the SHARC OpenVPN client on the
Pi first, as documented in `docs/SHARC_VPN_PI_SETUP.md`. Then provision the
protected audio endpoint values once after the first sync (the script reads the
Windows handoff only in memory and never prints the key):

```powershell
& '.\scripts\Provision-Qwen-Pi.ps1'
```

With the Pi SHARC VPN installed, the default endpoints are direct:
`10.153.60.233:8000` for ASR and `10.153.60.233:8001` for TTS. If the serving
address changes, pass it explicitly with `-DgxIp`. Use `-UseReverseTunnel` only
when the Pi VPN is intentionally unavailable and the laptop is connected to
the institutional VPN. Use `-UsePiBrainCredential` only when the routed
endpoint accepts the Pi's existing Qwen brain credential.

The canonical Qwen service is the Pi user unit on 8126/8127. Do not start a
second 8136/8137 listener while it is active. The named launcher is retained
for an explicit tunnel/start workflow, but `ssh me3-pi` is the direct
diagnostic route.

If ASR reports an empty transcript even though the Fifine signal is nonzero,
check the SHARC VPN route and the protected Qwen audio provider before changing
microphone settings. The Pi audio URLs should point directly to
`10.153.60.233:8000/8001` when the Pi VPN is active. The public
`202.92.159.240:8000/8001` endpoints expose the text Qwen3.8-27B service and
are not valid Qwen2-Audio/TTS endpoints.

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
├── docs/presentation/ME3_Voice_Duplex_Presentation_v10_Benchmark.pptx # presentation deck
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
The authoritative physical audio path remains `Start-ME3.bat`; Telegram text
and optional email remain confirmation-gated, while phone calls and SMS are not
part of the active project. No phone-provider account is required.

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
| FR16 | Agentic tool calling (build real artifacts from voice) | ✅ verified | `me3/brain.py` provider-neutral Qwen/Luna tool loop + `me3/tools.py` (project, web, task, ME2-parity, and channel tools) + `me3/task_runner.py` |
| FR17 | Optional ElevenLabs adapter | ✅ implemented, inactive in the single-voice demo path | `me3/elevenlabs_tts.py`; active profiles stay on one fixed voice |
| FR18 | Telegram text integration | ✅ implemented, credential-gated | `me3/channels.py` (`send_telegram_message`) + Telegram Bot API `sendMessage` |
| FR19 | Optional transactional email integration | ✅ implemented, credential-gated | `me3/channels.py` (`send_email`) + Resend Send Email API |
| FR20 | Telephone calls / SMS | ⛔ intentionally out of scope | Telegram Bot API text is the supported communication path; ordinary voice calls and SMS are not enabled |
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

Available tools include file ops (`list_files`, `read_file`, `write_file`,
`search_text`), build/test (`run_command`, `start_background_task`,
`task_status`, `task_output`, `cancel_task`), web serving
(`start_web_server`, `stop_web_server`, `web_servers`), research
(`fetch_url`, `web_search`), system (`system_info`, `git_status`), and
timers (`set_timer`, `list_timers`, `cancel_timer`). The channel tools are
`channel_health`, `send_telegram_message`, and the optional `send_email`; all
external side-effect tools require explicit confirmation and protected provider
configuration. Telegram Bot API text is supported. Ordinary Telegram
voice-call initiation is not exposed by the Bot API, so ME3 reports voice calls
as unsupported instead of routing them through another provider.

The ME2 command families are retained through explicit model-selected tools:
`play_music`, `media_control`, `get_weather`, `control_lights`,
`set_thermostat`, `set_alarm`, `create_reminder`, `call_contact`, and
`send_message`. See [ME2 command parity](docs/ME2_COMMAND_PARITY.md) for the
demo/live integration boundary. These adapters use inference and deterministic
tool execution; no ME2 model retraining is required. Clear short ME2 commands
also pass through a bounded voice-command routing guard before Qwen planning,
so repeated conversation history cannot turn an actionable command into a
generic readiness reply. The guard still emits the same named tool event and
uses the same dispatcher.

For music, the default Google route is **YouTube Music** because Google Play
Music was discontinued. `play_music` sends a visible browser handoff event to
the Pi UI; click **Open in browser** to use the already signed-in YouTube Music
or Spotify session on the operator's computer. ME3 never copies browser cookies
to the Pi. Fully automatic pause/skip/volume control would require a provider
player bridge and OAuth/SDK setup, so the current safe demo boundary is the
visible browser handoff.

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

## Active Qwen Instruct audio profile

`--variant qwen_instruct` preserves the same tool registry and duplex
controller while replacing only the audio providers. Qwen2-Audio is served as
`Qwen/Qwen2-Audio-7B-Instruct` through vLLM's multimodal
`/v1/chat/completions` route: ME3 sends the WAV as a base64 `audio_url` data URL
with a transcript-only prompt. Qwen3-TTS is served as
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` through `/v1/audio/speech`.

The protected handoff is read in memory by the PowerShell launcher and mapped
to `QWEN_INSTRUCT_ASR_BASE_URL`, `QWEN_INSTRUCT_TTS_BASE_URL`, and
`QWEN_INSTRUCT_API_KEY`. The former local audio deployment is archived. On
31 October 2026 over the VPN, the matched fixture
passed with Qwen ASR at 431.9 ms and Qwen TTS at 851.6 ms; the same run passed
with faster-whisper ASR at 2732.0 ms and Piper TTS at 100.1 ms TTFB/176.7 ms
total. The full injected Qwen voice/tool turn passed with zero errors and
called five shared tools. The comparison harness records no credentials,
transcripts, endpoints, or audio bytes.

## Telegram text and optional email integrations

ME3 uses Telegram Bot API text as its active outbound communication channel.
The optional Resend email adapter remains separate. Both are deliberately
credential-gated and confirmation-gated: the agent must identify the target and
content, receive explicit confirmation, and only then pass `confirm=true`. The
tools never return credential material or provider response bodies. The
Telegram setup checklist is in
[`docs/CHANNEL_INTEGRATIONS.md`](docs/CHANNEL_INTEGRATIONS.md).

Current implementation status is “adapter tested, live side effect not yet
placed.” `channel_health` is network-free and reports only configured/missing
environment names. The active Telegram adapter sends one Bot API
`sendMessage` request after confirmation. Ordinary Telegram voice calls are
not available through the Bot API; a future voice-call feature would require a
separately reviewed MTProto/TDLib user-account bridge.

Telegram uses the Bot API `sendMessage` method and requires
`TELEGRAM_BOT_TOKEN`; the operator must provide the approved `chat_id` and
confirm the message. No phone-provider account, phone number, or SMS provider
is required by the active project.

### Class benchmark protocol update

The class latency run is injection-only: it opens no physical microphone or
speaker, disables tool calls and external agent/harness calls, and disables
fillers/backchannels. The targets are E2E 4.0 s, ASR final 0.8 s, LLM TTFT
1.5 s, TTS TTFB 0.6 s, and barge-in 0.8 s. The direct-feed adapter keeps
live-agent tool behavior available separately through `--allow-tools`.

The first complete Pi stage run used 10 phrases × 3 runs and aggregated every
valid trial rather than choosing the best run. It produced E2E mean/p95 of
5.81/7.87 s, ASR final 3.38/4.92 s, LLM TTFT 0.21/0.21 s, TTS TTFB
1.60/2.98 s, and WER 29.3%. The Qwen ASR listener on DGX3 port 8000 was
unreachable during collection, so the runtime used its documented
faster-whisper fallback. Treat this as diagnostic deployed-pipeline evidence,
not a clean Qwen2-Audio ASR claim. The checked-in method and evidence boundary
are in [docs/ME3_QWEN_STAGE_BENCHMARK_20261107.md](docs/ME3_QWEN_STAGE_BENCHMARK_20261107.md).

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
capture-side echo guard. The browser sends a render-start acknowledgement
after scheduling the first Web Audio buffer, so the Pi retimes the reference
to the laptop's actual render schedule before correlation. This is the key
distributed-device step: a Pi send timestamp alone cannot describe when the
laptop speaker actually emitted the sound. The agent's own browser TTS is
therefore removed before VAD and ASR instead of becoming a false new turn.

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
