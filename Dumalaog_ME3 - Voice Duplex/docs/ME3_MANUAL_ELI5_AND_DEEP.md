# ME3 Voice Duplex: ELI5 Manual and Deep Technical Explanation

Version: 0.4, 25 October 2026

## What this project is

ME3 is a Raspberry Pi voice agent that listens and speaks at the same time.
It is designed to feel more like a conversation than a push-to-talk command
box: the user can interrupt the agent while it is speaking, and the agent can
use tools to do work rather than only generate an answer.

The language-model brain is Qwen served through the configured OpenAI-compatible
vLLM API. Speech input and output remain local to the device: our VAD detects
speech, our ASR server transcribes it, and our TTS server speaks the reply.
The four MCP servers are project-owned code. The ASR, TTS, and VAD models are
pretrained runtime models; ME3 does not require training a new model.

The most important new behavior is tool use. A spoken request such as:

> Build me a small web app that shows the Pi assistant status.

can cause Qwen to call the project file tools, write HTML/CSS/JavaScript,
start a background build or test when needed, read the result, and give a
short spoken summary. The voice loop and the text test mode use the same brain
and the same tool registry.

## ELI5: how it works

Imagine five helpers standing in a line:

1. **Ear helper (VAD):** notices when somebody is speaking.
2. **Word helper (ASR):** turns the sound into words.
3. **Thinking helper (Qwen):** decides what the words mean and whether a tool
   is needed.
4. **Hands helper (tools):** reads files, writes files, searches, runs a safe
   build command, or starts a longer task.
5. **Mouth helper (TTS):** turns the final answer back into speech.

The helpers pass messages to one another. Qwen is not allowed to pretend that
the hands helper completed a task. It receives the real tool result first.
That is why a web-app request can produce a real directory and real files.

### Why “duplex” matters

In a half-duplex assistant, the speaker may be disabled while the assistant is
talking. In ME3, the microphone remains open during playback. If the user
starts speaking while the agent is talking, VAD detects that speech and the
audio layer stops TTS playback. The new speech becomes the next turn.

### What “no training required” means

Training changes model weights. ME3 does not need that step for its agent
behavior. It uses:

- the Pipecat-bundled Silero CPU ONNX model for speech activity;
- a pretrained faster-whisper model for speech recognition;
- a pretrained Piper voice for speech synthesis; and
- the remote Qwen model for reasoning and tool selection.

The engineering work is orchestration, tool contracts, deployment, and
verification. If later measurements show that ASR or wake behavior needs a
custom model, that is a separate experiment and is not required for the
current tool-using duplex agent.

## What the agent can do

The direct Qwen registry contains these project-local capabilities:

| Capability | What it does | Typical voice request |
|---|---|---|
| `list_files` | Lists project files with sizes | “What files are in the web app?” |
| `read_file` | Reads a bounded text file | “Read the app configuration.” |
| `write_file` | Creates or replaces a text file | “Create an HTML page.” |
| `search_text` | Searches project text | “Find every reference to the Qwen endpoint.” |
| `run_command` | Runs one direct allowlisted command | “Run the web app test.” |
| `start_background_task` | Runs a longer build/test without blocking voice | “Start the frontend build and tell me when it finishes.” |
| `task_status` | Checks a background task | “Is the build done?” |
| `task_output` | Reads recent task output | “What did the test print?” |
| `cancel_task` | Stops a background task | “Stop that build.” |
| `system_info` | Reports non-secret host information | “How much memory is free?” |
| `fetch_url` | Fetches a public HTTP(S) page | “Read the project documentation page.” |
| `web_search` | Searches the public web | “Find the official Pipecat documentation.” |
| `git_status` | Shows repository status without mutation | “What changed?” |
| `set_timer` | Starts a one-shot timer | “Remind me in five minutes.” |
| `list_timers` | Lists pending timers | “What timers are running?” |
| `cancel_timer` | Cancels a pending timer | “Cancel that timer.” |
| `start_web_server` | Serves a project folder on the LAN and reports the URL | “Serve the web app and tell me the URL.” |
| `stop_web_server` | Stops a running preview server | “Stop the web server.” |
| `web_servers` | Lists active preview servers | “What is running on this machine?” |

The same registry is exposed through the owned tasks MCP server with explicit
MCP names such as `write_project_file`, `run_project_command`,
`start_project_task`, `search_public_web`, and `project_git_status`. The older
ME3 task tools remain available for timers, system information, and the ME2
bridge.

### Web-app example

The intended flow is:

1. User says what the app should do.
2. Qwen decides the app needs files.
3. Qwen calls `write_file` for each file under
   `agent_workspace/<app-name>/`.
4. Qwen calls `list_files` and `read_file` to inspect what it actually made.
5. If a build or test is needed, Qwen calls `run_command` or
   `start_background_task`.
6. Qwen receives tool results, fixes errors if possible, and then speaks a
   short factual summary.

The included demonstration at `agent_workspace/voice_webapp/` is a
dependency-free static HTML/CSS/JavaScript app with a command input,
microphone placeholder, status area, and tool-activity log.

## How to run it on Windows

All commands use the one shared workspace environment:

```powershell
Set-Location 'C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231'
Set-Location '.\AI 231\Dumalaog_ME3 - Voice Duplex'
$py = '..\..\.venv\Scripts\python.exe'

# Dependency and device check
& $py scripts\dev_env_check.py

# All four owned MCP servers, benchmark construction, and PortAudio
& $py scripts\local_smoke_test.py --with-audio

# Text-mode Qwen/tool test; the launcher supplies VLLM_API_KEY in memory
& $py -m me3.agent --brain remote --text 'Use system_info and summarize the host.'

# Live full-duplex microphone loop
& $py -m me3.agent --brain remote
```

The PowerShell launcher `scripts\Start-Local-ME3.ps1` resolves the configured
Qwen credential into the child process only. It reports only whether the
credential is configured. It never writes the key to the project.

For a no-network pipeline smoke test, use `--brain stub`. That exercises audio,
VAD, ASR, and TTS, but it cannot demonstrate Qwen tool calls.

## How to run it on the Pi

The ME3 deployment is separate from the ME2 `dandan` deployment. The intended
ME3 target is:

```text
Pi user: dalmacio
Host: cfdfnjrpi5.local
Path: /home/dalmacio/Desktop/kiko/me3_voice_duplex
```

The deployment workflow is:

```powershell
& $py scripts\package_release.py --force
& $py scripts\deploy_to_pi.py --dry-run
& $py scripts\deploy_to_pi.py
```

`deployment/latest_release.json` means only that a release was packaged and
verified locally. `deployment/last_pi_deployment.json` must say `deployed`
before the Pi is described as deployed. The verified deployment receipt is
release `me3-20261024-173332-36bb26b8` at
`/home/dalmacio/Desktop/kiko/me3_voice_duplex`; the transfer hash matched the
release manifest and the idempotent installer completed. On the Pi, `--check`
passes, and a live text-mode Qwen run built a stopwatch web app through
`write_file` plus `start_web_server` and served it over HTTP; the page was
cross-verified from a second machine on the LAN. The Pi currently exposes zero PortAudio
devices, so its user service remains disabled until hardware audio is checked.
The installer creates a Pi-local
virtual environment, installs the pinned runtime packages, prefetches an
RAM-appropriate Whisper model and Piper voices, and installs four user
systemd units. It does not enable autostart automatically; enable it on the Pi
only after the live audio and Qwen credential checks pass.

### Pi Qwen credential requirement

The Qwen API key is not bundled in the release. The Pi must have a secure
`VLLM_API_KEY` value in its runtime environment through the approved secure
operator channel and the service environment-file mechanism used by the
installer. The installed environment file is mode 600. Never print or copy
the value into project files, logs, handoff text, or chat.
The Windows DPAPI credential is not copied into Git, the release archive, the
handoff, or chat. If the Pi key is absent, the deployed agent can still start
in stub mode, but remote Qwen tool calls will not work until the Pi operator
provisions the key securely.

## Deep explanation of one turn

### 1. Audio capture and VAD

`FullDuplexAudio` keeps capture active and can play audio concurrently. Frames
are sampled at 16 kHz mono. The Pipecat-bundled Silero ONNX VAD receives
512-sample windows while the
audio device supplies the configured frame cadence. ME3 keeps a short pre-roll,
counts speech and trailing silence, and ends a turn after the configured
minimum speech and silence durations.

The state machine is:

```text
idle --speech--> listening --enough silence--> thinking
  ^                                           |
  |                                           v
  +--------- playback finished -------- speaking
                    ^
                    |
       speech while speaking = barge-in, stop playback,
       collect a new utterance, return to listening
```

The state machine is separate from the language model. This means it can be
unit-tested and can also be used by a Pipecat transport.

### 2. ASR

At the end of a turn, faster-whisper transcribes the PCM signal using the
configured model and int8 CPU inference. Empty transcripts are ignored. A
non-empty transcript becomes a user message for the Qwen brain.

### 3. Qwen tool-calling protocol

The brain sends an OpenAI-compatible chat request containing:

- a system message describing ME3 and its safety boundary;
- the recent conversation history;
- the current transcript; and
- JSON Schema function definitions from `me3.tools.tool_definitions()`.

Qwen can return either normal assistant text or one or more `tool_calls`. For
each tool call, ME3:

1. parses the JSON arguments;
2. dispatches the named local function;
3. appends a `role=tool` message with the real JSON result; and
4. asks Qwen for the next action or final answer.

The loop is bounded by `llm.remote.max_tool_steps` (eight by default; the checked-in configuration sets ten). This
prevents a malformed or indecisive model response from running forever.

The request/response shape is conceptually:

```json
{
  "messages": [
    {"role": "user", "content": "build a web app"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "write_file", "parameters": {}}}
  ]
}
```

Then Qwen may answer with a call such as `write_file(path, content)`. ME3
executes it and sends back:

```json
{
  "role": "tool",
  "tool_call_id": "...",
  "name": "write_file",
  "content": "{\"ok\": true, \"path\": \"agent_workspace/app/index.html\"}"
}
```

The tool result, rather than a model assumption, is the source of truth.

### 4. File and command safety

The tool registry is not a general unrestricted shell. It applies these rules:

- paths must resolve inside the ME3 project;
- credential-like paths, `.env`, `.git`, virtual environments, private keys,
  model caches, and release trees are protected;
- file reads and writes are bounded by size;
- direct commands are tokenized and run without a shell;
- only development and inspection executables are allowlisted;
- shell operators and destructive/external-access patterns are rejected; and
- long jobs write output under `agent_workspace/.tasks` and are capped by a
  timeout.

This is enough for a local static webapp, Python helper, or frontend build,
while avoiding accidental `rm -rf`, SSH, credential access, or repository
rewrites from an ambiguous voice transcript.

The boundary is intentionally narrower than unrestricted Onit. Onit can be
used for broad interactive coding with explicit operator oversight. ME3 is an
always-listening device, so its default voice tools must be bounded. Additional
capabilities should be added as explicit typed tools with tests, not by turning
voice into an arbitrary shell.

### 5. Background work

Voice must remain responsive while a build runs. `BackgroundTaskManager` starts
an allowlisted process with standard input closed, captures stdout/stderr to
project-local task logs, watches the timeout, and exposes status/output/cancel
operations. The assistant can say that a task is running, continue listening,
and answer a later “is it finished?” request from the stored status.

### 6. TTS and barge-in

After Qwen returns final text, Piper produces PCM audio. ME3 resamples that
audio to the configured device rate and starts non-blocking playback. VAD
continues to run. Speech detected during `speaking` stops playback and starts a
new turn. This is the practical duplex behavior; a future Pi deployment still
needs a physical audio check and echo-control tuning.

## MCP servers and ownership

ME3 ships four owned FastMCP servers:

- `mcp_vad.py`: frame and stream speech detection plus endpointing;
- `mcp_asr.py`: PCM/file transcription;
- `mcp_tts.py`: synthesis, streaming chunks, and voice inventory; and
- `mcp_tasks.py`: timers, system information, the ME2 bridge, and the project
  tool wrappers.

The direct Qwen loop calls the Python registry in-process for low overhead.
The MCP wrappers expose the same bounded operations to another harness. Both
paths therefore share the same path and command policy.

## Verification contract

The following claims are distinct:

| Claim | Evidence required |
|---|---|
| Code compiles | `py_compile` or `compileall` passes |
| MCP servers work locally | `scripts/local_smoke_test.py --with-audio` passes |
| Qwen tool loop works | live `--text` request logs a tool call and returns text |
| Webapp build capability works | real files exist and a Qwen request reads/verifies them |
| Release is prepared | `latest_release.json` says `prepared` and package hashes pass |
| Release is deployed | `last_pi_deployment.json` says `deployed` |
| Pi audio works | physical Pi capture/playback check passes |
| Paper benchmark is complete | captured v1.5 overlap artifacts and evaluator output exist |

The diagnostic stub benchmark is not remote-Qwen evidence and is not the
paper-native FULL-DUPLEX-BENCH v1.5 result. The four paper scenarios and the
engineering five-scenario latency driver remain separate.

## Known limitations and next experiments

1. The remote endpoint is currently HTTP, not HTTPS. Use it only on the trusted
   institutional network or VPN; migrate to administrator-provided HTTPS when
   available.
2. The Pi still needs a live audio self-test after deployment. The current
   check reports zero PortAudio devices; a connected mic,
   speaker behavior, ALSA device choice, CPU/RAM, and systemd supervision must
   be recorded separately from local Windows checks.
3. The Qwen API key is deliberately not included in the release. On the
   deployed Pi it is provisioned in a mode-600 runtime environment file;
   only its presence is verified, and the value is never printed.
4. The local loop handles barge-in during TTS, but it does not yet provide the
   full semantic RESUME/RESPOND classifier required by the paper’s overlap
   study.
5. A complete 499-trial v1.5 overlap run, judge transcripts, and before/after
   prosodic analysis remain future benchmark work.
6. The voice capability should be extended through reviewed typed tools for
   deployment, database, and external-service actions rather than unrestricted
   shell access.

## Quick troubleshooting

### Qwen says the key is missing

Check only the configured/not-configured status. Do not print the key. On
Windows, use `scripts\Start-Local-ME3.ps1 -Brain remote`; on the Pi, verify the
service environment was provisioned securely.

### The agent speaks but cannot build a webapp

Use text mode with the same request first. Inspect the tool log for a rejected
path, an existing file requiring `overwrite=true`, a command outside the
allowlist, or a background task that is still running. The default target is
`agent_workspace/<app-name>`.

### The Pi starts but does not hear speech

Run `arecord -l`, inspect the selected PortAudio/ALSA device, verify the user
has audio permissions, and run the Pi-local ASR/VAD health checks. Do not infer
Pi microphone behavior from a successful Windows PortAudio check.

### A build takes too long

Ask for `task_status` and `task_output`, or cancel it. The manager caps tasks at
30 minutes and keeps the voice loop separate from the worker process.

## Source map

- Runtime configuration: `config/config.yaml`
- Qwen brain: `me3/brain.py`
- Tool registry: `me3/tools.py`
- Background jobs: `me3/task_runner.py`
- Full-duplex loop: `me3/agent.py` and `me3/audio_io.py`
- Owned MCP servers: `me3/mcp_*.py`
- Local verification: `scripts/dev_env_check.py`,
  `scripts/local_smoke_test.py`, `scripts/check_paper_alignment.py`
- Release/deploy: `scripts/package_release.py`, `scripts/deploy_to_pi.py`,
  `deployment/latest_release.json`, `deployment/last_pi_deployment.json`
- Paper source and contract: `FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf`,
  `docs/PAPER_ALIGNMENT.md`, `me3/fdb_v15.py`

