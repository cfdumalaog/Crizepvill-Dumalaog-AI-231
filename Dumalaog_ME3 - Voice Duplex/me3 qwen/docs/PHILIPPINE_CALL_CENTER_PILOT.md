# ME3 Philippines call-center pilot

This pilot profile targets Filipino (`fil`), English (`en`), and natural
Taglish conversations on a Raspberry Pi. The active Qwen release uses Qwen as
the reasoning and tool-calling brain, Qwen2-Audio for remote ASR, and
Qwen3-TTS CustomVoice `aiden` for replies, backchannels, and completion
announcements. English is the default spoken language; Filipino or Taglish is
mirrored when the caller uses it or explicitly requests it.

## Current voice and safety boundary

The active runtime deliberately uses one fixed voice and a formal,
professional customer-service style. Piper is the local TTS fallback. The
former ElevenLabs adapter was retired after the Pi-side account and TTS
probes returned HTTP 401/404; it is not a provider option in this release.
See `docs/ELEVENLABS_INTEGRATION.md` for the short historical decision record.

Voice localization must not be confused with authorization. The same bounded
tools, path checks, approval policy, consent behavior, and evidence rules apply
to Filipino, English, and Taglish turns.

## ELI5 operating model

1. A customer speaks Filipino, English, or Taglish.
2. Local Silero VAD detects speech boundaries while the microphone remains
   live during playback.
3. Qwen2-Audio ASR or the configured fallback turns speech into text.
4. Qwen decides whether to answer, call a bounded tool, or request a human.
5. Qwen3-TTS Aiden speaks the answer; Piper is used if the remote TTS service
   is unavailable.
6. If unmatched customer speech overlaps playback, automatic barge-in stops
   the current audio. The echo guard rejects ME3's own browser/Pi output.

The browser can optionally provide the microphone: select **This browser /
connected PC microphone** and click **Connect browser microphone**. The browser
transmits 16-kHz mono PCM over the existing WebSocket while the selected
browser speaker remains the output. The default demo input is still the Pi
Fifine microphone.

## Suggested pilot queues

Start with low-risk, high-volume tasks with deterministic answers:

- store hours, service coverage, and branch directions;
- order or appointment status after a controlled identity check;
- approved product/service FAQs;
- ticket creation and status lookup through a reviewed backend tool;
- lead qualification and callback scheduling with explicit consent;
- routing to a human queue when confidence, policy, or customer emotion is
  outside the approved scope.

Do not allow the pilot to make irreversible financial, medical, legal, account,
employment, or identity decisions without a reviewed human workflow. A
production telephony connector, CRM connector, recording policy, and identity
provider require separate review.

## Acceptance checklist

### Language and voice

- [ ] English-default reply is verified.
- [ ] Filipino and Taglish turns are intelligible and do not silently switch
      language.
- [ ] Qwen3-TTS Aiden is the same voice for reply, backchannel, and completion.
- [ ] Piper fallback is verified after a remote TTS timeout.

### Duplex behavior

- [ ] Automatic barge-in stops current playback for unmatched human speech.
- [ ] Echo guard rejects the agent's own output while retaining human speech.
- [ ] Browser microphone input is separately recorded as browser-input evidence.
- [ ] Pi Fifine capture is preflighted before a hardware demo.
- [ ] Injected evaluation is not presented as physical microphone/speaker proof.

### Call-center safety

- [ ] The caller is told that the first-line agent is AI where required by
      product/legal policy.
- [ ] PII retention, recording consent, deletion, and access controls are
      defined before real customers are connected.
- [ ] Human escalation and transfer failure are tested.
- [ ] The pilot has a kill switch, rate limit, audit policy, and an owner for
      reviewing failures.

## What the project can claim today

The current evidence supports a Qwen/tool-capable duplex control path,
provider-selectable ASR/TTS/VAD/tool layers, a browser microphone transport,
an injection benchmark path, and a Pi deployment path. It does not establish
that ME3 can replace a staffed call center at production scale. That requires
a supervised pilot with queue metrics, human handoff, privacy review, safety
testing, and cost/quality comparison.
