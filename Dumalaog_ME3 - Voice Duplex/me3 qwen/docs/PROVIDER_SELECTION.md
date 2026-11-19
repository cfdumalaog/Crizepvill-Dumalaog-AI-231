# Choosing the ME3 voice provider

The active ME3 codebase now has two complete voice stacks. Both use the Pi's
microphone, Silero VAD, the same ME3 tool harness, and the selected Pi or
browser speaker route. No model training is required.

| Selection | Speech recognition | Tool-using text model | Speech output |
|---|---|---|---|
| Qwen | DGX3 Qwen2-Audio (with existing Deepgram/local ASR fallback) | Qwen3.8-27B | DGX3 Qwen3-TTS Aiden (with Piper fallback) |
| OpenAI | `gpt-transcribe` | `gpt-4.1-mini` | `gpt-4o-mini-tts`, Onyx, 24 kHz PCM |
| Gemini | Qwen2-Audio DGX3 by default | Gemini OpenAI-compatible `gemini-2.5-flash` | Qwen3-TTS Aiden by default |

The OpenAI profile uses the current file-transcription endpoint for completed
VAD turns. It is still the ME3 continuous microphone loop, not a browser
speech-to-text button. Its TTS endpoint is a compatibility path and is slated
for shutdown in January 2027; replace it with an OpenAI Realtime speech path
before that date. Provider choice alone does not prove live API access or a
new full-duplex acoustic benchmark result.

## Conversation-first setup

The Assistant now has one conversation surface. Type or say `set up OpenAI`,
`set up Spotify`, `set up Google Calendar`, `set up Telegram`, `set up GitHub`,
or `set up Deepgram`. ME3 inserts a service-specific prompt in the conversation
with only the required fields and actions. The normal chat endpoint rejects
credential-shaped text, so a pasted secret cannot accidentally become a model
turn.

1. Obtain an OpenAI API key from the OpenAI developer platform with access to
   transcription, chat completions, and speech. An active billing account may
   be required by the provider.
2. Start the Pi ME3 service and open the Assistant through the existing laptop
   SSH tunnel at `http://127.0.0.1:8100/`. LAN pages can view status, while
   private setup writes remain loopback-only.
3. Type or say `set up OpenAI`, enter the key in the contextual password prompt,
   and choose **Save privately on Pi**. ME3 stores it in the Pi user's private
   config directory (`~/.config/me3/credentials.json`, mode 0600), never in
   source, release ZIP, `/api/chat`, UI status responses, or browser storage.
4. Choose **Launch OpenAI** in that same prompt. ME3 saves the
   selection and restarts its Pi service, loading ASR, brain, and TTS together.
   The selected provider persists across Pi restarts. To switch back, type or
   say `set up Qwen` and choose **Launch Qwen**.
5. Check the Runtime card for `gpt-transcribe`, `gpt-4.1-mini-tools`, and
   `gpt-4o-mini-tts` before trying one short spoken turn. The status endpoint
   only reports that a key is stored; a spoken turn verifies authorization,
   network access, model availability, and audio playback.
6. To use Gemini as the text/tool brain, type or say `set up Gemini`, enter the
   Gemini API key privately, and choose **Launch Gemini brain**. Gemini uses
   Google's OpenAI-compatible chat-completions endpoint with function calling;
   Qwen ASR, Qwen TTS Aiden, and Silero remain the audio defaults.

If the Pi service is unreachable, first reconnect the Pi and SSH tunnel. The
provider can be selected only on the Pi service; this project does not start a
second ME3 service on the laptop. The Qwen profile can be restored even without
an OpenAI key. Existing Qwen, Deepgram, and channel credentials are preserved.

The same contextual setup flow handles Spotify's client ID and OAuth action,
Google Calendar's OAuth client fields, the Telegram BotFather token and target
mapping, the Deepgram fallback key, and the GitHub publishing token. These
values use their dedicated private endpoints; they never enter the model's
conversation history. Ordinary approvals such as `Yes, proceed` remain chat
messages and are bound to the preceding assistant request.

### Open-ended vLLM/API setup

The named service buttons are convenience shortcuts. They are not the provider
allowlist. Type or say a request such as `use this OpenAI-compatible vLLM
endpoint for the text tool model` or choose **Other vLLM / API**. ME3 asks the
setup planner for a secret-free description, infers the requested role, and
renders a form for the URL, model, private API key, protocol, voice, and
fallback. The form is generated at runtime; it is not a hard-coded field set
for only the seven buttons in the screenshot.

Use **Save privately on Pi** to store the profile, or **Save and activate** to
persist it, select the matching layer/provider, and restart ME3. The save and
activation writes are loopback-only, so open the Assistant through the local
SSH tunnel when configuring a secret. A custom text profile must expose
OpenAI-compatible `/v1/chat/completions`; custom ASR/TTS profiles must expose
the selected audio protocol. ME3 validates this role/protocol pairing before
activation and never returns the API key in status or conversation data.

The supplied DGX3 configuration is intentionally separate from this custom
text-provider form: port 8000 is Qwen2-Audio ASR and port 8001 is Qwen3-TTS.
Those two live audio services do not provide the Qwen text/tool brain. A
reachable text/vLLM or OpenAI brain is still required for a transcript to turn
into a tool-using agent response.

The OpenAI speech voice is AI-generated. Tell listeners that the voice is AI.

## Independent model-layer selectors

The Assistant includes a **Model layers** card below Pi audio routing. It reads
and writes `/api/layers` and offers a selector for the tool brain as well as
the audio layers:

- Tool brain: Qwen, OpenAI, Gemini, or a configured custom LLM profile;
- ASR: Qwen2-Audio DGX3, Deepgram Nova-3, faster-whisper local, OpenAI
  transcription, or a configured custom ASR profile;
- TTS: Qwen3-TTS Aiden, Piper local, ElevenLabs Filipino, OpenAI Onyx, or a
  configured custom TTS profile;
- VAD: local Silero or the lightweight energy-gate fallback.

The selector writes only backend/profile references to a private `layers.json`
beside the Pi credentials file. It never stores a key or puts a secret in the
conversation. The generic `custom_vllm` label is an internal compatibility
marker and is not offered in the dropdown; configured profiles are listed by
their saved names and model IDs.
Applying a selection schedules the canonical Pi service to restart, because
ASR/TTS engines are cached at process start. A direct service restart is still
available in the compact bottom control row. The complete Qwen/OpenAI provider
selector remains the quickest way to choose a consistent end-to-end stack;
independent selectors are for deliberate fallback/testing combinations.

## Non-destructive provider smoke test

The Pi release includes `scripts/provider_matrix_smoke.py`. It exercises the
configured ASR/TTS/VAD adapters with a local WAV fixture without opening the
physical speaker, changing the selected layer, printing transcripts, or
emitting credentials:

```bash
cd /home/dalmacio/Desktop/dandan-qwen
set -a
[ -f "$HOME/.config/me3/me3.env" ] && . "$HOME/.config/me3/me3.env"
[ -f "$HOME/.config/me3/qwen-instruct.env" ] && . "$HOME/.config/me3/qwen-instruct.env"
set +a
PYTHONPATH=. .venv/bin/python scripts/provider_matrix_smoke.py \
  --wav /tmp/me3-hello.wav
```

The 2026-11-19 Pi run verified the following after selecting the Qwen audio
layers: Qwen ASR completed directly through DGX3 in about 1.0 s, Qwen3-TTS
completed directly through DGX3 in about 1.4 s at 24 kHz using `aiden`, local
faster-whisper and Piper also completed, and both Silero and energy VAD probes
completed. OpenAI ASR and ElevenLabs TTS were correctly reported as not
configured on the Pi. The fixture test is adapter evidence only; it is not a
human microphone or physical speaker/echo-gate result. The text/tool brain
remains a separate reachability gate.

## Action permission policy

Provider selection and action authorization are separate controls. The
Assistant's **Action permissions** card exposes the same three policies as the
Codex-style reference UI:

1. **Ask for approval** — pause before internet, shell, external-file, and
   outbound side-effect tools.
2. **Approve for me** — run ordinary requested work, but pause for potentially
   unsafe external side effects such as GitHub publishing, channel messages,
   calls, calendar writes, and external file changes.
3. **Full access** — no ME3 approval pause. Existing OS-user and protected
   credential-path safeguards still apply; this is not root access and does
   not reveal provider keys.

The selected mode is persisted privately on the Pi. A blocked action returns a
credential-free summary and creates one in-memory pending action. The operator
can click **Approve action**/**Cancel action** or say “approve”/“cancel”. The
central dispatcher enforces this for both deterministic voice commands and
model-generated tool calls, including `deploy_github_pages`, so GitHub actions
no longer depend on a hidden, route-specific confirmation flag.
