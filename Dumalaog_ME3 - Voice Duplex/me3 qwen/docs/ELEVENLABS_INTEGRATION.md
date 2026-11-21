# ElevenLabs integration — retired from active ME3

This file is retained as a short decision record, not as an installation guide.

On 2026-11-21 the active Pi deployment was checked using its protected
runtime configuration without printing or copying any credential. The account
probe returned HTTP 401, and the configured text-to-speech request returned
HTTP 404. Because the provider was not usable, the active ME3 Qwen release
removed its ElevenLabs adapter, selector entry, setup form, dependency, and
private runtime entries.

The active voice path is now deliberately single-voice:

* replies, backchannels, and completion announcements use Qwen3-TTS
  CustomVoice `aiden`;
* Piper is the local fallback when Qwen TTS is unavailable;
* the browser-native speech synthesizer remains disabled so it cannot add a
  second, inconsistent voice;
* no ElevenLabs key or voice ID is required by the service.

If a future course iteration evaluates ElevenLabs again, it must be treated as
a separate provider experiment. It must first pass an account/auth check, a
short TTS request, voice-consistency checks, and the full-duplex echo/barge-in
tests before it can be added back to the active selector. Credentials must be
entered through the protected runtime setup path and must never be committed,
printed, or placed in the browser bundle.
