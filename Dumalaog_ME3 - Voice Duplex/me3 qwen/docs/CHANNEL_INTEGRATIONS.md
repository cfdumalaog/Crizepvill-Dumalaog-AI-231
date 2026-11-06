# ME3 external-channel API requirements

This document is the implementation and provisioning checklist for connecting
ME3 to telephone calls, SMS/text messages, Telegram text, and email. The adapters are shared
by the direct provider-neutral tool loop and the `me3-tasks` MCP server. They do
not train a model: Qwen or GPT-5.6 Luna performs inference and chooses a bounded
tool, while the Pi runs
the adapter and records only safe provider metadata.

## What is implemented

| Channel | Provider/API | ME3 entry point | Side effect | Current evidence |
|---|---|---|---|---|
| Outbound voice call | Twilio Voice Calls resource | `make_call` | Starts a phone call | Implemented; fake-request tested, no live call placed |
| SMS/text | Twilio Messages resource | `send_sms` | Sends one SMS | Implemented; fake-request tested, no live SMS sent |
| Transactional email | Resend Send Email API | `send_email` | Sends an email | Implemented; fake-request tested, no live email sent |
| Telegram text | Telegram Bot API `sendMessage` | `send_telegram_message` | Sends one bot text message | Implemented; fake-request tested, no live message sent |
| Telegram voice call | Telegram Bot API | n/a | Ordinary bot voice-call initiation | Not available in the Bot API; use Twilio or a separately reviewed user-authenticated MTProto/TDLib bridge |
| Configuration status | Local, network-free | `channel_health` | None | Implemented and tested |

Every external side effect requires `confirm=true`. The agent must first state
the recipient and the content/action in the spoken conversation and receive an
explicit confirmation. A missing credential, invalid target, rejected request,
or timeout is reported as a failure; ME3 never claims delivery from a tool
call that did not return success.

## Required APIs and accounts

### 1. Qwen/vLLM: reasoning and tool selection

This is the default model API for the agentic path. At least one model provider
must be configured; the optional Luna provider below can be used instead.

- Endpoint: the OpenAI-compatible vLLM `/v1` endpoint in
  `config/config.yaml`.
- Model: `Qwen/Qwen3.8-27B` in the checked-in configuration.
- Runtime secret: `VLLM_API_KEY` in the protected Pi service environment.
- Function: transcribe-to-text context is sent to Qwen; Qwen returns a reply
  or a bounded function call such as `make_call` or `send_email`.
- Transport note: the current configured endpoint is plain HTTP. Use it only
  on the trusted institutional network/VPN; an administrator-provided HTTPS
  endpoint is the production requirement.

No Qwen training or fine-tuning is required for ME3.

### 1b. Active Qwen Instruct audio services

The supplied Qwen handoff is an audio-provider pair, not a replacement
for the text tool-calling brain. The active ME3 release uses the
OpenAI-compatible vLLM audio endpoints for `Qwen/Qwen2-Audio-7B-Instruct` ASR
and `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` TTS. Local Silero VAD, the text ToolBrain,
bounded tools, and the duplex interruption controller are in the same release.
Qwen2-Audio uses vLLM's multimodal `/v1/chat/completions`
surface with an `audio_url` data URL and transcript-only prompt; Qwen3-TTS uses
`/v1/audio/speech`. ME3 wraps those routes as its ASR/TTS MCP servers. The
protected services were live-verified over VPN on 31 October 2026.

Runtime-only values:

```text
QWEN_INSTRUCT_ASR_BASE_URL
QWEN_INSTRUCT_TTS_BASE_URL
QWEN_INSTRUCT_API_KEY
```

The workstation launcher can parse the protected handoff file in memory. A Pi
deployment must provide the same values through its protected service
environment. If the VPN route is unavailable, the active Qwen profile fails
closed and reports a provider error. The former local audio deployment is
archived.

### 2. GPT-5.6 Luna: optional OpenAI reasoning and tool selection

ME3 can use the official OpenAI-compatible `gpt-5.6-luna` model as a second
tool-calling brain. It uses the same `ToolBrain`, bounded project tools, voice
state machine, and TTS path as Qwen; this is an inference-provider choice, not
a second voice pipeline.

- Endpoint: `https://api.openai.com/v1`.
- Model ID: `gpt-5.6-luna`.
- Reasoning: disabled for the functional voice path; direct response streaming
  is enabled when supported by the provider.
- Runtime secret: `OPENAI_API_KEY`, kept only in the protected workstation or
  Pi service environment.
- Selection: `python scripts/run_me3_module.py me3.agent --brain luna`, or **GPT-5.6 Luna + tools** in
  the local UI.
- Audio boundary: the model accepts text, not audio in this ME3 path; local
  multilingual Whisper still performs ASR and Piper/ElevenLabs still performs
  TTS.

The current account has not supplied an OpenAI API key in this workspace, so
the Luna path is configured and unit-testable but not live-provider-verified.
OpenAI documents Chat Completions and function calling for this model and lists
`medium` as a supported reasoning effort.

### 3. Twilio Voice: outbound calls and call-flow webhook

ME3 uses Twilio's REST API to create an outbound call. The request is a
form-encoded `POST` to the Calls resource with `To`, `From`, and `Url`. Twilio
then fetches the configured `Url`, which must return valid TwiML. The official
resource is [Twilio Call resource](https://www.twilio.com/docs/voice/api/call-resource)
and the general HTTPS API base is documented in the
[Twilio Voice API overview](https://www.twilio.com/docs/voice/api).

Required protected environment values:

```text
TWILIO_ACCOUNT_SID       # account identifier
TWILIO_AUTH_TOKEN        # credential; never commit or print
TWILIO_FROM_NUMBER       # E.164 Twilio number used as caller ID
TWILIO_VOICE_URL         # reviewed public HTTPS endpoint returning TwiML
```

The implementation accepts a Twilio Messaging Service SID as an SMS sender,
but an outbound call still needs `TWILIO_FROM_NUMBER`. `TWILIO_VOICE_URL` is a
call-flow webhook, not a local Pi URL. A full conversational call-center
deployment additionally needs a reviewed webhook/media-stream bridge that
connects the caller's audio to the ME3 duplex loop, handles call status and
hangup events, and enforces recording/consent policy. The current adapter
initiates a call; it does not pretend that a static TwiML URL is already a
complete production media bridge.

Provisioning and safety requirements:

- buy or verify a Twilio voice-capable number and its Philippines routing;
- use E.164 numbers (`+639...`) in ME3;
- deploy the TwiML endpoint over HTTPS with an operator-controlled domain;
- configure status callbacks, rate limits, retry policy, consent/AI
  disclosure, recording policy, and human transfer before real customers;
- test with an operator-owned destination first; never run an unsolicited call
  from a demo voice turn.

### 4. Twilio Messaging: SMS/text

ME3 uses Twilio's Messages resource with a form-encoded `POST` to
`/Accounts/{AccountSid}/Messages.json`, sending `To`, `Body`, and either
`From` or `MessagingServiceSid`. See the official
[Twilio Message resource](https://www.twilio.com/docs/messaging/api/message-resource)
and [Twilio Messaging API overview](https://www.twilio.com/docs/messaging/api).

Required protected environment values:

```text
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_FROM_NUMBER             # or use TWILIO_MESSAGING_SERVICE_SID
TWILIO_MESSAGING_SERVICE_SID   # optional alternative sender
```

For a trial account, the destination may need verification. Production SMS
also requires sender registration, opt-in/opt-out handling, message templates
where applicable, rate limits, and a privacy/retention decision. WhatsApp is
not silently enabled; if the project later needs it, the provider channel and
approved sender must be added and tested separately.

### 5. Resend: transactional email

ME3 uses Resend's `POST https://api.resend.com/emails` endpoint with a Bearer
API key and JSON containing `from`, `to`, `subject`, and plain-text `text`.
The request shape is documented by the official
[Resend Send Email API](https://resend.com/docs/api-reference/emails/send-email).

Required protected environment values:

```text
RESEND_API_KEY
RESEND_FROM_EMAIL             # verified sender/domain address
```

Before live use, verify the sender domain, configure SPF/DKIM/DMARC as the
operator requires, set recipient/rate limits, define bounce/complaint
handling, and decide whether email content may contain customer PII. ME3
starts with plain-text transactional email; HTML templates, attachments,
threading, and inbound mailbox processing are intentionally not enabled by
the first adapter.

### 6. Telegram Bot API: text messages

ME3 uses the official Telegram Bot API `sendMessage` method with a JSON `POST`
to `https://api.telegram.org/bot<TOKEN>/sendMessage`. The implementation is
available through both the direct tool registry and the `me3-tasks` MCP server
as `send_telegram_message(chat_id, text, confirm=true)`. The bot token is read
from the protected runtime environment variable `TELEGRAM_BOT_TOKEN`; the
adapter reports only configured/not-configured state and never returns the
token or Telegram response body.

Required protected environment value:

```text
TELEGRAM_BOT_TOKEN
```

The agent must identify the chat and message and receive explicit confirmation
before setting `confirm=true`. The operator must first start the bot and send
it a message so Telegram can provide the correct numeric `chat_id` (or use a
reviewed `@channelusername`).

The Telegram Bot API does not provide ordinary one-to-one voice-call initiation.
ME3 therefore does not expose a fake Telegram call tool. `make_call` remains
the Twilio phone-call tool. A future Telegram voice-call feature would require
a separately reviewed user-account bridge using MTProto/TDLib, with its own
session security, consent, rate limits, and human-transfer policy; a bot token
alone is insufficient.

## Local model and infrastructure APIs

These are required by the complete voice agent even when no external channel
is provisioned:

| API/component | Purpose | Where it runs |
|---|---|---|
| Pipecat 1.13 | Full-duplex orchestration and barge-in control | Raspberry Pi |
| MCP/ FastMCP | Owned ASR, TTS, VAD, and task-server interface | Raspberry Pi |
| Silero ONNX via Pipecat | Voice activity detection | Raspberry Pi CPU |
| faster-whisper | Local speech-to-text | Raspberry Pi CPU |
| Piper | Offline speech fallback | Raspberry Pi CPU |
| ElevenLabs TTS | Optional Filipino-capable hosted voice | Pi or dev machine |
| PortAudio/ALSA | Microphone and speaker I/O | Raspberry Pi |
| SSH/scp + systemd user service | Deployment and supervision | Dev machine/Pi |

ElevenLabs is not the reasoning model. It renders the selected provider's response after the
tool loop. Its protected values are `ELEVENLABS_API_KEY` and the operator-
selected `ELEVENLABS_VOICE_ID`; the selected voice still needs Filipino or
Philippine-English approval. See `docs/ELEVENLABS_INTEGRATION.md`.

## Runtime setup

The Pi service reads a mode-600 file at `~/.config/me3/me3.env`. Do not put
these values in `config/config.yaml`, a commit, a tool argument, a handoff, or
a transcript. Example variable names only:

```bash
VLLM_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+639...
TWILIO_VOICE_URL=https://support.example.ph/twilio/answer
RESEND_API_KEY=...
RESEND_FROM_EMAIL=ME3 <support@example.ph>
```

The values above are placeholders and must not be copied literally into a
production service file. The installer preserves the environment file and
the tool registry reports only configured/not-configured state.

Read-only checks:

```bash
python scripts/run_me3_module.py me3.agent --check
python -c "from me3.channels import channel_health; print(channel_health())"
```

The first command checks the local model and audio prerequisites. The second
does not contact Twilio, Resend, or Telegram. A live channel test requires an
operator-owned destination and an explicit run outside the ordinary smoke
test; no live message or call is part of the automated test suite.

## Remaining production work

- provision and securely inject the provider values;
- for Telegram text, create the bot, obtain the operator-approved chat ID, and
  perform one confirmed test message;
- supply an HTTPS Twilio webhook/media-stream bridge for conversational calls;
- if Telegram voice calls are required, design and review a separate
  user-authenticated MTProto/TDLib bridge; do not treat the Bot API as a call
  transport;
- connect approved CRM/ticket/identity systems through separately reviewed
  MCP tools rather than giving the agent unrestricted HTTP access;
- add inbound call/SMS/email webhooks, signature verification, idempotency,
  queueing, audit logs, deletion/retention controls, and human transfer;
- perform operator-owned live tests and record provider IDs/status only;
- run a supervised Philippines pilot with AI disclosure, consent, safety,
  escalation, quality, latency, and cost gates.

Until those gates pass, ME3 is a demo-ready, credential-gated voice agent and
not a claim that a production call center can be removed.
