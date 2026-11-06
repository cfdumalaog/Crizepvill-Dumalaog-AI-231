# ME3 Voice Duplex: ELI5 Manual and Deep Technical Explanation

Version: 0.8, 5 November 2026

## What this project is

ME3 is a Raspberry Pi voice agent that listens and speaks at the same time.
It is designed to feel more like a conversation than a push-to-talk command
box: the user can speak while the agent is speaking, and can use explicit
voice controls to stop, skip, or steer playback. The agent can use tools to do
work rather than only generate an answer.

The language-model brain is provider-selectable. Qwen served through the
configured OpenAI-compatible vLLM API remains the default, and GPT-5.6 Luna
(`gpt-5.6-luna`, reasoning effort `medium`) is an optional OpenAI backend.
Speech input remains on the device: our VAD detects speech and our ASR server
transcribes it. TTS can speak through a Pi output device or stream raw PCM to
the connected browser so the operator's PC plays the reply.
The four MCP servers are project-owned code. The ASR, TTS, and VAD models are
pretrained runtime models; ME3 does not require training a new model.

The most important new behavior is tool use. A spoken request such as:

> Build me a small web app that shows the Pi assistant status.

can cause Qwen or GPT-5.6 Luna to call the project file tools, write HTML/CSS/JavaScript,
start a background build or test when needed, read the result, and give a
short spoken summary. The voice loop and the text test mode use the same brain
and the same tool registry.

## ELI5: how it works

Imagine five helpers standing in a line:

1. **Ear helper (VAD):** notices when somebody is speaking.
2. **Word helper (ASR):** turns the sound into words.
3. **Thinking helper (Qwen or GPT-5.6 Luna):** decides what the words mean and whether a tool
   is needed.
4. **Hands helper (tools):** reads files, writes files, searches, runs a safe
   build command, or starts a longer task.
5. **Mouth helper (TTS):** turns the final answer back into speech.

The helpers pass messages to one another. Qwen is not allowed to pretend that
the hands helper completed a task. It receives the real tool result first.
That is why a web-app request can produce a real directory and real files.

### Why “duplex” matters

In a half-duplex assistant, the speaker may be disabled while the assistant is
talking. In ME3, the microphone remains open during playback. Ordinary speech
that overlaps a response is compared with the speaker reference: residual
output is rejected, while a genuine user utterance is buffered and processed
after the current answer. A short control path recognizes `stop processing`,
`skip`, and `steer`; those commands can stop the current audio without a
browser button. This is protected full-duplex playback rather than automatic
barge-in.

The live browser UI includes a Runtime terminal. It is an operator-facing
window into safe processing events: state changes, ASR/TTS lifecycle, tool
names/status, timing, and model/voice labels. It deliberately omits hidden
chain-of-thought, credentials, tool arguments, and raw tool results.

The adjacent Background processes panel answers the practical question “what
is it doing right now?” It shows the accepted voice turn, elapsed time,
current safe phase, heartbeat count, and bounded project subprocesses with PID
and redacted output tails. A long turn receives one formal spoken progress
update per minute at a safe audio boundary. This is a status summary, not a
display of private model chain-of-thought.

### What “no training required” means

Training changes model weights. ME3 does not need that step for its agent
behavior. It uses:

- the Pipecat-bundled Silero CPU ONNX model for speech activity;
- a pretrained faster-whisper model for speech recognition;
- one fixed profile voice for speech synthesis: Piper `en_GB-alba-medium` in
  Own or Qwen3-TTS `aiden` in Qwen; and
- the remote Qwen model or GPT-5.6 Luna for reasoning and tool selection.

The engineering work is orchestration, tool contracts, deployment, and
verification. If later measurements show that ASR or wake behavior needs a
custom model, that is a separate experiment and is not required for the
current tool-using duplex agent.

### Voice choice and provider lock

ME3 keeps Qwen as the default reasoning and tool-calling model. The active
profiles use exactly one voice model each: Own uses Piper
`en_GB-alba-medium`, while Qwen uses Qwen3-TTS CustomVoice `aiden`. The same
voice is used for replies, backchannels, and completion announcements, so a
turn cannot change speaker identity because of an environment variable.
The spoken style is also locked to formal customer service: the long-task
backchannel is exactly `I am processing your request now.`, and the assistant
must avoid flirtatious, intimate, suggestive, teasing, or overly familiar
language.

The ElevenLabs adapter remains available for a deliberate future experiment,
but the active selector does not auto-select it and does not use a second voice
as a fallback. This keeps the functional demo simple and repeatable. Eleven
Realtime Mini and Max remain an alternative hosted architecture, not a silent
replacement for the Qwen/tool loop.

ElevenLabs Realtime v1 Mini and Max are complete hosted voice-agent modes that
also manage the LLM and speech-to-text configuration. Using them would be an
alternative architecture, not simply a voice switch for Qwen. The exact
configuration and acceptance gate are documented in
docs/ELEVENLABS_INTEGRATION.md.

### Qwen fast mode and long-task visibility

The Qwen Instruct text brain is configured for direct streaming with
`fast_mode: true` and `thinking: false`. The OpenAI-compatible request carries
`chat_template_kwargs: {enable_thinking: false}`. This disables Qwen's hidden
reasoning-token generation on the voice path; the UI still shows only safe
phase labels such as “planning,” “running tool,” and “preparing the final
response.”

If the immediate model acknowledgement window expires, ME3 keeps the brain
call on a worker thread and enters `awaiting`. Every loop poll publishes the
worker's elapsed time and current phase. Project subprocesses started through
`start_background_task` are read from their bounded log tails and exposed as
redacted PID/status rows. At most once each 60 seconds, a formal heartbeat is
queued and spoken when the current audio boundary is safe; task completion is
then announced separately.

### ME2-style Pi web access and remote speaker

The Pi can run `scripts/me3_pi_server.py`, which combines the live assistant
dashboard with a raw PCM WebSocket speaker route. The microphone is opened by
the Pi's PortAudio stack, so a Fifine USB microphone attached to the Pi remains
the input source. When the output route is `browser`, the TTS samples are sent
to the browser over the LAN and scheduled through Web Audio on the operator's
PC. When the output route is a numeric/name Pi device, playback stays local to
the Pi. The UI's **Pi audio routing** card applies the new route by restarting
the audio loop; it does not change the LLM or tool state.

The browser must be connected with a user click once. This is a browser
autoplay security rule, not a manual voice-input trigger: ME3 continues to
listen on the Pi continuously, and the browser only receives the agent's
speaker stream. The SSH launcher forwards both the HTTP status port and the
WebSocket audio port, so the same experience works when the Pi is reachable
through the ME2-style tunnel.

For the conceptual difference from ME2, read
[`ME2_VS_ME3.md`](ME2_VS_ME3.md). ME2 trains a tiny fixed-label intent model;
ME3 uses inference APIs and project-owned orchestration/tools, so the agent can
interpret an open-ended request such as “build a web app” and execute the
resulting bounded file/build/web-server tools. The two systems should not be
compared using the same accuracy denominator.

### The final Qwen Instruct audio profile

The protected handoff defines the active Qwen audio path. It is easy to
misunderstand the audio services as a second general-purpose agent model. They
are not:
`Qwen/Qwen2-Audio-7B-Instruct` is used as the ASR service and
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` is used as the speech renderer. ME3 keeps
Silero VAD local, then sends the completed turn to the same text ToolBrain and
the same bounded MCP/tool registry. Therefore a request to build a web app,
inspect files, set a timer, or prepare a channel action uses the final Qwen
release's normal agent tools.

Select the path at runtime:

```powershell
& '.\scripts\Start-Local-ME3.ps1' -Brain remote -Variant qwen_instruct
& '.\scripts\Start-ME3-UI.ps1' -Variant qwen_instruct -OpenBrowser
```

The Qwen launcher reads `Secrets New ME3 Model Qwen Instruct.txt` only in
memory and maps the two protected vLLM listeners to
`QWEN_INSTRUCT_ASR_BASE_URL`, `QWEN_INSTRUCT_TTS_BASE_URL`, and
`QWEN_INSTRUCT_API_KEY`. On the Pi, set those three values in the protected
service environment instead. No key, endpoint value, transcript, or audio
bytes belong in Git or evaluation artifacts.

The Qwen2-Audio server is called through the OpenAI-compatible
`/v1/chat/completions` route with the WAV encoded as an `audio_url` data URL.
The Qwen3-TTS server is called through `/v1/audio/speech`. The matched
comparison harness is:

```powershell
& '..\..\.venv\Scripts\python.exe' scripts\compare_audio_variants.py `
  --secret-file '..\..\Secrets New ME3 Model Qwen Instruct.txt' --timeout 90
```

It reports only provider status, safe model labels, latency, byte counts, and
error classes. On 31 October 2026, both protected model IDs returned
authenticated `/v1/models` responses over the VPN. A matched fixture run
passed with Qwen2-Audio ASR at 431.9 ms and Qwen3-TTS at 851.6 ms; the same
run passed with faster-whisper ASR at 2732.0 ms and Piper TTS at 100.1 ms
TTFB/176.7 ms total. The full injected Qwen voice lane also completed one turn
with zero errors and called the shared `list_files`, `read_file`, `run_command`,
`start_web_server`, and `write_file` tools. If the VPN route becomes
unavailable, the active profile fails closed and reports the provider error.
The former local audio build is archived rather than presented as a second
production variant.

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
| `play_music` | Queues a browser/audio-sink music action | “Play Filipino jazz.” |
| `media_control` | Plays, pauses, stops, skips, or changes volume | “Pause.” / “Make it louder.” |
| `get_weather` | Reads current weather for a location | “What is the weather in Manila?” |
| `control_lights` | Turns lights on/off, dims, or sets a color | “Dim the living-room lights to 30 percent.” |
| `set_thermostat` | Sets a validated Celsius target | “Set the temperature to 24 degrees.” |
| `set_alarm` | Creates a next-occurrence alarm | “Set an alarm for 7 AM.” |
| `create_reminder` | Creates a delayed or timed reminder | “Remind me in ten minutes.” |
| `call_contact` | Resolves a protected contact and requests a call | “Call mom.” |
| `send_message` | Sends confirmed SMS or Telegram text | “Message mom on Telegram.” |

| `channel_health` | Reports safe call/SMS/email configuration status | "Are the communication channels configured?" |
| `send_sms` | Sends one confirmed Twilio SMS | "Text the customer this update." |
| `make_call` | Starts one confirmed outbound Twilio call | "Call the customer and use the support flow." |
| `send_email` | Sends one confirmed Resend email | "Email the customer the appointment details." |

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

### ELI5: calls, texts, and emails

ME3 now has three communication helpers, but they behave like a locked
outbox. `channel_health` checks whether the operator configured a provider; it
does not send anything. For a text, call, or email, ME3 must say who will
receive it and what it will contain, then wait for the user's clear yes. Only
then can Qwen call the matching tool with `confirm=true`.

Twilio handles the phone call and SMS. Resend handles transactional email. A
call starts through a reviewed HTTPS TwiML URL. That URL is the call-flow
instruction that Twilio fetches; a real call-center conversation additionally
needs an approved media-stream bridge, customer consent, AI disclosure,
recording policy, human transfer, and CRM/identity integrations. The full
provider checklist and required environment names are in
`docs/CHANNEL_INTEGRATIONS.md`.

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
& $py scripts/run_me3_module.py me3.agent --brain remote --text 'Use system_info and summarize the host.'

# Text-mode GPT-5.6 Luna/tool test; provide OPENAI_API_KEY securely
& $py scripts/run_me3_module.py me3.agent --brain luna --text 'Use system_info and summarize the host.'

# Live full-duplex microphone loop
& $py scripts/run_me3_module.py me3.agent --brain remote
```

The PowerShell launcher `scripts\Start-Local-ME3.ps1` resolves the configured
Qwen and OpenAI credentials into the child process only. It reports only
whether each credential is configured. It never writes either key to the
project. In `auto` mode it prefers Qwen, then Luna, then the stub.

For a no-network pipeline smoke test, use `--brain stub`. That exercises audio,
VAD, ASR, and TTS, but it cannot demonstrate Qwen tool calls.

## How to run it on the Pi

The ME3 deployment is separate from the ME2 `dandan` deployment. The intended
ME3 target is:

```text
Pi user: dalmacio
Host: cfdfnjrpi5.local
Path: /home/dalmacio/Desktop/dandan
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
release `me3-20261025-101632-b0238933` at
`/home/dalmacio/Desktop/kiko/me3_voice_duplex`; the transfer hash matched the
release manifest and the idempotent installer completed. On the Pi, `--check`
passes, and a live text-mode Qwen run built a stopwatch web app through
`write_file` plus `start_web_server` and served it over HTTP; the page was
cross-verified from a second machine on the LAN. The read-only Pi audio
preflight reports `pass_devices_visible` (exit 0) with the Fifine microphone at
`hw:2,0`. The user service remains disabled until a physical Pi speaker test
and human spoken turn are captured; the split-device acceptance path can route
Pi capture to the workstation speaker meanwhile.
The installer creates a Pi-local
virtual environment, installs the pinned runtime packages, prefetches an
RAM-appropriate Whisper model and the single fixed Piper voice, and installs four user
systemd units. It does not enable autostart automatically; enable it on the Pi
only after the live audio and Qwen credential checks pass.

### Pi inference credential requirement

The Qwen API key is not bundled in the release. The Pi must have a secure
`VLLM_API_KEY` value in its runtime environment through the approved secure
operator channel and the service environment-file mechanism used by the
installer. The installed environment file is mode 600. Never print or copy
the value into project files, logs, handoff text, or chat.
The Windows DPAPI credential is not copied into Git, the release archive, the
handoff, or chat. If the Pi key is absent, the deployed agent can still start
in stub mode, but remote Qwen tool calls will not work until the Pi operator
provisions the key securely.

To select GPT-5.6 Luna on the Pi, provision `OPENAI_API_KEY` through the same
mode-600 service environment mechanism, then launch the agent with
`--brain luna`. The release never contains either key. Luna uses the official
API model ID `gpt-5.6-luna`; hidden reasoning is disabled and direct response
streaming is enabled when supported. ASR and TTS remain the Pi-local ME3
services.

The ElevenLabs adapter remains documented and tested, but it is not active in
the single-voice demo path. Own is locked to Piper `en_GB-alba-medium` and
Qwen is locked to Qwen3-TTS `aiden` with a formal style instruction; environment overrides cannot switch the
voice at runtime. Re-enable ElevenLabs only as a deliberate future experiment
after choosing a replacement voice and updating the acceptance gate.

Before enabling the user service, run the read-only Pi audio gate:

```bash
python scripts/pi_audio_preflight.py
```

This separately reports ALSA capture/playback devices and PortAudio devices.
Playback-only HDMI output does not satisfy the duplex requirement. Connect a
USB/I2S microphone or configure an ALSA capture/loopback device, then rerun
the gate. On Windows, run `python scripts/run_me3_module.py me3.bench_driver --brain stub
--preflight` before any latency trials; a failed tone calibration invalidates
the run.

### Injection-only evaluation without a speaker

If the Pi microphone is present but the speaker is unavailable, use the
injection evaluator instead of forcing an output-device selection:

```bash
python scripts/injection_eval.py \
  --input-wav runs/injection_eval_inputs/tell_me_system_status.wav \
  --interrupt-wav runs/injection_eval_inputs/stop.wav \
  --brain remote
```

The evaluator uses `InjectedDuplexAudio`. It delivers 16-kHz PCM frames at
wall-clock cadence, appends a controlled silence tail so VAD can close the
turn, and sends the frames through the same VAD, faster-whisper, Qwen/tool,
Piper-or-ElevenLabs, and barge-in code. Synthesized TTS is discarded by a
virtual sink while its duration and stop events are recorded. An interrupt
WAV scheduled during virtual TTS tests whether the agent can listen and act
while it is speaking, without acoustic speaker feedback.

This produces an engineering artifact under `runs/injection_eval/`. It proves
the injected-input agent/tool path and virtual-output barge-in path only. It
does not prove the physical microphone signal, physical speaker behavior, or
the paper-native FULL-DUPLEX-BENCH v1.5 result; those lanes stay separate.

In the live loop, barge-in stops only the audio currently playing. If the
previous request has already entered the long-task worker, its tool work and
completion reply continue; the completion is queued until the newer turn is
idle. A background project task is canceled only through the explicit
`cancel_task` or `cancel_project_task` tool.

For a physical Windows speaker-feedback check, use the digital `Stereo Mix`
capture device:

```powershell
python scripts\loopback_selftest.py --input "Stereo Mix" --expect-echo-suppressed
```

This deliberately routes ME3's own TTS through `FullDuplexAudio.play()`, lets
Stereo Mix capture that output, and verifies that the cleaned frames are near
silence before VAD/ASR. The production guard uses a short playback reference
and correlation search for a real microphone; it uses a strict mute only for
digital loopback devices because those devices contain system output rather
than independent user speech. The engineering benchmark disables the guard in
its private configuration so its raw stimulus measurements remain valid.

For the Pi-to-laptop-speaker path, the guard additionally fits a bounded
multi-tap room response around the best reference match. It gates a frame only
when the cleaned residual falls below a configured fraction of the input, so
speaker delay, reflections, and USB-clock drift are handled without deleting
unmatched user speech. `/api/audio/devices` exposes only safe enabled/active,
cancelled-frame, and correlation counters. A real-room human acceptance test
is still a separate evidence gate.

### Split-device local acceptance: Pi microphone, workstation speakers

When the Fifine microphone is attached to the Pi but the operator wants to
hear the response through the workstation speakers, run the local agent with
the SSH microphone transport:

```powershell
& '..\..\.venv\Scripts\python.exe' scripts/run_me3_module.py me3.agent --brain remote `
  --ssh-mic 'dalmacio@cfdfnjrpi5.local' --remote-device 'hw:2,0'
```

The Pi runs `arecord` on the Fifine device and sends raw mono 16-kHz PCM over
SSH. The workstation runs the same VAD, ASR, Qwen/tool, and TTS state machine,
but resolves only its local output device for playback. Barge-in remains
active because the remote capture thread continues while local TTS is
playing. This is a local acceptance transport; production Pi mode remains
host-local and expects a Pi playback device. The command does not print or
copy the Qwen credential.

## Deep explanation of one turn

### 1. Audio capture and VAD

`FullDuplexAudio` keeps capture active and can play audio concurrently. Frames
are sampled at 16 kHz mono. The Pipecat-bundled Silero ONNX VAD receives
512-sample windows while the
audio device supplies the configured frame cadence. ME3 keeps a short pre-roll,
counts speech and trailing silence, and ends a turn after the configured
minimum speech and silence durations.

Before a frame reaches VAD, the audio transport checks whether ME3 is
speaking. For a physical microphone it searches a delay window against the
playback reference, estimates the correlated speaker component, and removes
it. A highly correlated frame is hard-gated; a lower-correlation frame keeps
the residual so a user can barge in. For `Stereo Mix` or another recognized
digital loopback device, ME3 hard-mutes frames during playback because the
device is a copy of the system output. This prevents the agent's own response
from being transcribed as a new turn while preserving the live microphone
path for normal duplex operation.

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

### 1a. Vocalis-Pi: the always-listening contract

The ELI5 version is: ME3 has a traffic controller for the conversation. It
keeps listening while it is talking. When it hears the user speak over the
reply, it tells the speaker to throw away queued audio, fades the current
audio, and listens to the new request. When a task takes a while, it can say a
short acknowledgement while the Qwen tools keep working.

The controller is implemented by `VocalisOrchestrator` in
`me3/vocalis.py`. Its stable states are:

| State | Plain meaning |
|---|---|
| `LISTENING` | The microphone is ready for a new turn. |
| `THINKING` | ASR produced a final transcript and Qwen/tools are working. |
| `SPEAKING` | Reply text has been sent to the TTS path. |
| `BACKCHANNELING` | A short acknowledgement is playing during a long task. |
| `INTERRUPTED` | Overlap was detected and TTS is being stopped. |

The controller publishes machine-readable events instead of hiding these
decisions in a UI. The input side carries VAD states and transcript deltas;
the output side carries assistant text, TTS controls, turn-state changes, and
an optional JSON-RPC `harness_dispatch` request. The current Qwen path still
executes the bounded ME3 tool registry locally, while tool start/finish
progress is mirrored as sanitized harness events. A future Onit/Pipecat
adapter can consume the same contract without changing the microphone or
barge-in logic. See [`VOCALIS_PROTOCOL.md`](VOCALIS_PROTOCOL.md) for the
complete schema and deployment boundary.

### 2. ASR

At the end of a turn, faster-whisper transcribes the PCM signal using the
configured model and int8 CPU inference. Empty transcripts are ignored. A
non-empty transcript becomes a user message for the selected Qwen or Luna brain.

### 3. Provider-neutral tool-calling protocol

The selected brain sends an OpenAI-compatible chat request containing:

- a system message describing ME3 and its safety boundary;
- the recent conversation history;
- the current transcript; and
- JSON Schema function definitions from `me3.tools.tool_definitions()`.

Qwen or GPT-5.6 Luna can return either normal assistant text or one or more
`tool_calls`. For
each tool call, ME3:

1. parses the JSON arguments;
2. dispatches the named local function;
3. appends a `role=tool` message with the real JSON result; and
4. asks the selected provider for the next action or final answer.

The loop allows up to 24 tool rounds for a demanding multi-file or web-app task.
If the model still requests another tool after the final permitted batch, ME3
executes that batch and makes one no-tool summarization request; it never speaks
an internal tool-limit error to the user. GPT-5.6 Luna does not receive
`reasoning_effort`; the implementation omits `temperature` for that provider
and requests the direct response stream when supported. This
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

Then the selected provider may answer with a call such as
`write_file(path, content)`. ME3
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

### 3a. External-channel tool protocol

The external tools follow the same selected-provider function-call loop, but their
side-effect boundary is stricter:

1. `channel_health` returns only configured/not-configured state and required
   environment names; it performs no network request.
2. `send_sms`, `make_call`, and `send_email` validate the target locally.
3. Without `confirm=true`, each returns `confirmation_required` and
   `network_called=false`.
4. With confirmation, the adapter sends one bounded REST request and returns
   only `ok`, provider, status code, channel, and a safe provider ID.
5. Provider response bodies and headers are discarded, because they can
   contain account details or customer data.

Phone targets must be E.164. Calls require a reviewed HTTPS webhook; email is
plain-text and bounded to ten recipients in the first implementation. The
adapters are shared by direct Qwen calls and the `me3-tasks` MCP wrappers, so
adding a second harness does not create a second permission policy.

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

After Qwen returns final text, the selected Piper or ElevenLabs backend
produces PCM audio. ME3 resamples that audio to the configured device rate and
starts non-blocking playback. VAD
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
2. The Pi capture path is visible and injection-only evaluation passes. The Pi
   still needs a physical speaker test and a human spoken turn; speaker
   behavior, ALSA output choice, CPU/RAM, and systemd supervision must be
   recorded separately from local Windows and split-device checks.
3. The Qwen API key is deliberately not included in the release. On the
   deployed Pi it is provisioned in a mode-600 runtime environment file;
   only its presence is verified, and the value is never printed.
4. The local loop handles barge-in during TTS, but it does not yet provide the
   full semantic RESUME/RESPOND classifier required by the paper’s overlap
   study.
5. A complete 499-trial v1.5 overlap run, judge transcripts, and before/after
   prosodic analysis remain future benchmark work.
6. Inbound call/SMS/email webhooks, provider signature verification,
   idempotency, queueing, CRM/identity actions, and human transfer remain
   production integration work. The outbound adapters are ready for
   operator-owned tests, not an unsupervised customer rollout.

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
- External channels: `me3/channels.py`, `docs/CHANNEL_INTEGRATIONS.md`
- Background jobs: `me3/task_runner.py`
- Full-duplex loop: `me3/agent.py` and `me3/audio_io.py`
- Owned MCP servers: `me3/mcp_*.py`
- Local verification: `scripts/dev_env_check.py`,
  `scripts/local_smoke_test.py`, `scripts/check_paper_alignment.py`
- Release/deploy: `scripts/package_release.py`, `scripts/deploy_to_pi.py`,
  `deployment/latest_release.json`, `deployment/last_pi_deployment.json`
- Paper source and contract: `FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf`,
  `docs/PAPER_ALIGNMENT.md`, `me3/fdb_v15.py`

