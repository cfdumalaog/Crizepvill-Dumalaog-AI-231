# ME3 Philippines Call-Center Pilot

## What changed

ME3 now has a Philippines-market runtime profile:

- Filipino (`fil`), English (`en`), and natural Taglish are in scope.
- Qwen is still the reasoning and tool-calling brain; no model training is
  required for this profile.
- ElevenLabs Flash v2.5 is configured as the low-latency Filipino-capable TTS
  path when an operator provisions an approved API key and a matching voice
  ID. `ME3_ELEVENLABS_LANGUAGE_CODE=fil` can override the configured language.
- `auto` remains safe: it selects ElevenLabs only when both secrets are
  present, and falls back to Piper otherwise.
- The first external-channel layer is implemented: explicit-confirmation
  tools can start an outbound Twilio call, send Twilio SMS, and send a Resend
  transactional email. Provider account, webhook, consent, and production
  gates are documented in `docs/CHANNEL_INTEGRATIONS.md`.
- The voice agent is positioned as a first-line support pilot with explicit
  human escalation, not as an unconditional replacement for a call center.

ElevenLabs documents Filipino as `fil` (an ISO 639-3 code) and lists it among
the languages supported by Flash v2.5. A voice selected from the ElevenLabs
library should match the target accent and the operator's usage rights. The
voice ID is intentionally not committed to this repository.

## ELI5 operating model

1. A customer speaks Filipino, English, or Taglish.
2. The microphone audio is split into small frames. VAD detects when the
   customer starts and stops talking.
3. ASR turns the speech into text.
4. Qwen decides whether it can answer, needs a bounded tool, or should ask for
   a human.
5. ElevenLabs speaks the answer in Filipino when configured; Piper is the
   offline fallback.
6. If the customer speaks while ME3 is talking, ME3 stops the current playback
   and listens to the new turn.

The important boundary is that language and voice localization do not grant
new permissions. The same bounded tools, path checks, confirmation behavior,
and evidence rules remain in force.

## Suggested pilot queues

Start with low-risk, high-volume tasks that have deterministic answers:

- store hours, service coverage, and branch directions;
- order or appointment status after a controlled identity check;
- basic product/service FAQs from an approved knowledge source;
- ticket creation and status lookup through a reviewed backend tool;
- lead qualification and callback scheduling with explicit consent;
- routing to a human queue when confidence, policy, or customer emotion is
  outside the approved scope.

Do not allow the pilot to make irreversible financial, medical, legal, account,
employment, or identity decisions without a reviewed human workflow. A
production telephony connector, CRM connector, recording policy, and identity
provider are not included in this repository and must be separately reviewed.

## Runtime configuration

The checked-in profile is in `config/config.yaml`:

```yaml
market:
  region: PH
  locale: fil-PH
  default_language: fil
  supported_languages: [fil, en]
  code_switching: true

tts:
  engine: auto
  elevenlabs:
    model_id: eleven_flash_v2_5
    language_code: fil
```

Provision secrets only through the operator-controlled runtime environment:

```bash
export ELEVENLABS_API_KEY='<operator-only secret>'
export ELEVENLABS_VOICE_ID='<approved Filipino voice id>'
export ME3_TTS_ENGINE=elevenlabs
export ME3_ELEVENLABS_LANGUAGE_CODE=fil
```

Never put those values in Git, a handoff, a screenshot, a tool argument, or a
customer transcript. The integration reports only configured/not-configured
status and safe provider event metadata.

## Human handoff contract

The pilot should expose a clear handoff action in the surrounding call-center
system. At minimum, preserve:

- the caller's explicit request for a human;
- the reason for escalation and the last safe transcript summary;
- consent and contact-channel state required by the approved workflow;
- a transfer/failure state if no human is immediately available.

ME3 must say that it is an AI assistant when asked and must not claim that a
human has taken over until the external system confirms the transfer. If a
customer disputes an answer, reports harm, or asks for a regulated decision,
the safe response is to stop improvising and escalate.

## Acceptance checklist

### Language and voice

- [ ] Approved Filipino/Philippine-English voice ID is provisioned through the
  protected operator channel.
- [ ] `health()` reports `engine=elevenlabs`, `language_code=fil`, and only
  safe credential-presence metadata.
- [ ] A short Filipino response is heard on the selected output device.
- [ ] A Taglish response is intelligible and does not silently switch the
  customer to an unintended language.
- [ ] Piper fallback is verified after a provider timeout or when credentials
  are absent.

### Duplex behavior

- [ ] The customer can interrupt speech and the playback stops promptly.
- [ ] Injected evaluation is recorded separately from physical-microphone and
  physical-speaker evidence.
- [ ] The Pi Fifine capture path is preflighted before enabling a service.
- [ ] No paper-native FULL-DUPLEX-BENCH v1.5 claim is made from a pilot demo.

### Call-center safety

- [ ] The caller is told that the first-line agent is AI where required by the
  approved product/legal policy.
- [ ] PII retention, recording consent, deletion, and access controls are
  defined by the operator before real customers are connected.
- [ ] Human escalation is tested, including transfer failure and customer
  request for a human.
- [ ] The pilot has a kill switch, rate limit, audit log policy, and an owner
  responsible for reviewing failure reports.

## What this project can claim today

The current evidence shows a Qwen/tool-capable duplex control path, a
Filipino-ready ElevenLabs configuration, a provider-live local TTS call played
through the Windows speaker, an injection-only Pi evaluation, and a
split-device local test path. The provider-live test used an operator-selected
account voice whose catalog label was English; it therefore validates the
integration and Filipino language-code request, not approval of a native
Filipino accent. The remote Qwen endpoint was unavailable during the latest
local voice test, and the Pi is offline. This does not establish that ME3 can
replace a staffed call center at production scale. That conclusion requires a
supervised pilot with real queue metrics, human handoff, privacy review, safety
testing, and cost/quality comparison.
