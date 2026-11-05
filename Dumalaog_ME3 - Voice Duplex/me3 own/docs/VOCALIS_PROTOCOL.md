# Vocalis-Pi full-duplex protocol

Vocalis-Pi is ME3's transport-independent contract for a voice agent that is
always listening, can speak while it continues to receive microphone frames,
and can be controlled without a browser button or a second manual submit.
The contract is implemented in [`me3/vocalis.py`](../me3/vocalis.py) and is
wired into the real [`LocalVoiceLoop`](../me3/agent.py), not just a mock UI.

## What is actually live

The workstation and Pi-compatible loop is:

```text
16 kHz PCM microphone
        |
        v
Silero VAD (20-30 ms updates) ---- while TTS plays
        |
        v
endpointed utterance -> faster-whisper ASR -> Qwen OpenAI-compatible API
                                                |
                                                +-> ME3 tools (MCP-style tool loop)
                                                +-> web/file/system task result
        ^                                       |
        |                                       v
        +-- protected playback / voice control -- TTS (ElevenLabs or Piper fallback)
```

The Qwen path is text-compatible internally because the configured Qwen
endpoint is a chat-completions model. That does not make ME3 a manual
voice-to-text application: the user speaks continuously, VAD automatically
ends the turn, Qwen tools execute without a Send button, and the reply is
spoken automatically. While audio is playing, new microphone speech triggers
the default protected-playback path keeps ordinary overlapping speech from
flushing the current reply. The newest ordinary utterance is queued for the
next safe audio boundary. Explicit voice commands (`stop processing`, `skip`,
or `steer`) stop current playback; `stop processing` also suppresses queued
announcements. If a long tool request has already been accepted, its worker
and side effects continue; use the normal task-cancellation tool when actual
task cancellation is required.

The default runtime still performs the bounded local ME3 tool loop in
`me3/brain.py`. This is the verified path for building a webapp, writing
files, starting a local server, reading the result, and reporting completion.
Vocalis also exposes an explicit JSON-RPC MCP harness dispatch seam for an
Onit/Pipecat/OpenCode-style background harness. It is deliberately an adapter
boundary: the protocol module does not make an unapproved network call or
assume a particular harness transport.

## State machine

| State | Meaning | Entered by | Leaves when |
|---|---|---|---|
| `LISTENING` | Microphone is open and the agent is ready for speech. | startup, completed TTS, empty/invalid turn | VAD captures a turn and ASR returns final text |
| `THINKING` | A final transcript is being reasoned about or a harness/tool task is pending. | final ASR delta, silence endpoint, MCP dispatch | assistant text or a backchannel is ready |
| `SPEAKING` | Reply audio is queued or playing while the mic remains open. | assistant text delta | playback completes or an explicit voice control is recognized |
| `BACKCHANNELING` | A short acknowledgement is playing while a long Qwen/tool task continues. | `ack_wait_s` exceeded | task result is announced or an explicit voice control is recognized |
| `INTERRUPTED` | A configured interrupt/control path is stopping TTS. | explicit control or legacy barge-in mode | the next speech frame returns to `LISTENING` |

The local audio loop has implementation states `idle`, `listening`,
`thinking`, `speaking`, and `awaiting`; the Vocalis state is the stable
external contract. In particular, local `awaiting` maps to Vocalis
`BACKCHANNELING` while a long-running tool task continues.

## Input channels

Each event is represented by `VocalisEvent`:

```json
{
  "channel": "user_vad_state",
  "payload": {"state": "ACTIVE", "probability": 0.91},
  "timestamp": 1730000000.123,
  "sequence": 42
}
```

| Channel | Payload emitted by ME3 |
|---|---|
| `user_vad_state` | `state` (`ACTIVE`/`INACTIVE`), frame timestamp, optional probability |
| `user_transcript_delta` | `text`, `final`, timestamp |
| `barge_in_event` | overlap timestamp |
| `harness_event_stream` | sanitized `tool_started`/`tool_finished` progress or an adapter result |

The local loop publishes a final transcript after faster-whisper endpointing.
Tool arguments and tool results are intentionally excluded from the progress
event stream because they can contain user data or credentials.

## Protected-playback voice controls

With the default `barge_in: false` profile, VAD continues to feed a short
control buffer while TTS is playing. The buffer is ASR-transcribed after a
short silence and classified before it can become a normal prompt:

| Spoken command | Effect |
|---|---|
| `stop processing` | Stop current playback, clear deferred prompt/announcements, and suppress pending spoken completions. |
| `skip` | Stop current playback and continue with the newest buffered prompt if one exists. |
| `steer` | Stop current playback and prioritize the newest buffered prompt if one exists. |
| Any other utterance | Retain it as the newest prompt and dispatch it only after the protected audio boundary. |

This command path is intentionally narrow so that talking to another person,
backchanneling, or ordinary overlap does not accidentally cancel a response.
The UI's Runtime terminal mirrors the resulting state, ASR, TTS, and sanitized
tool lifecycle events from `/api/state`; it does not expose hidden reasoning.

## Output channels

| Channel | Payload emitted by ME3 |
|---|---|
| `assistant_text_delta` | `text`, `final`, optional `kind=backchannel` |
| `tts_control_signal` | `action` (`PLAY`, `PAUSE`, `FLUSH_QUEUE`, `FADE_OUT`) and reason |
| `turn_state` | stable state plus transition reason |
| `harness_mcp_dispatch` | JSON-RPC 2.0 `tools/call` request for a configured harness adapter |

On barge-in, the ordering is:

```text
barge_in_event -> turn_state: INTERRUPTED
                -> tts_control_signal: FLUSH_QUEUE
                -> tts_control_signal: FADE_OUT
                -> turn_state: LISTENING
```

The flush applies to the active TTS buffer, not to the accepted task. A
background task is only canceled through its explicit task-cancellation tool;
otherwise its final announcement waits until the newer turn is no longer
speaking.

This ordering is unit-tested in `tests/test_vocalis_protocol.py`.

## MCP harness dispatch

The adapter seam produces a normal JSON-RPC request without coupling the audio
thread to network I/O:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "harness_dispatch",
    "arguments": {
      "task": "build a web app",
      "arguments": {"path": "agent_workspace/demo"}
    }
  }
}
```

Call `VocalisOrchestrator.dispatch_harness(...)` from a transport adapter and
provide a fast `harness_dispatch` callback if the deployment uses a separate
Onit/Pipecat/OpenCode process. The callback may return an acknowledgement;
completion is then published as `harness_event_stream`. For the current ME3
release, Qwen's local bounded tool loop is the primary verified execution
path, while Qwen tool callbacks are mirrored as sanitized harness events.

## Running and checking it

From the ME3 directory, use the shared workspace environment:

```powershell
$py = "..\..\.venv\Scripts\python.exe"
& $py -m pytest tests/test_vocalis_protocol.py -q
& $py -m compileall -q me3 scripts
& $py scripts/workstation_acceptance.py
```

For a live workstation run, start `launchers\Start-ME3.bat` or:

```powershell
& $py scripts/Start-Local-ME3.ps1 -Brain remote
```

The agent opens the configured microphone and speaker, says nothing until a
turn is detected, and handles the turn automatically. The browser UI is a
secondary diagnostic surface; it is not required to activate the microphone.

## Raspberry Pi deployment boundary

The release packager includes this protocol and the live loop. The Pi must
still pass the separate ALSA/PortAudio preflight and a human audio check after
transfer. A local acceptance pass proves workstation readiness only; it does
not prove that the Pi's microphone, speaker, network route, Qwen endpoint, or
provider credentials are available.
