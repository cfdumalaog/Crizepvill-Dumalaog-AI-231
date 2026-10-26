# ElevenLabs voice integration for ME3

Version: 0.2, 26 October 2026

## Decision in one paragraph

ME3 keeps Qwen/Qwen3.8-27B as the reasoning and tool-calling model. ElevenLabs
is added as an optional streaming voice renderer, using raw 16 kHz PCM so the
existing microphone, VAD, ASR, playback, and barge-in path remains intact.
This is the correct integration for the requirement “use ElevenLabs to voice
our model.”

ElevenLabs also has Eleven Realtime v1 Mini and Eleven Realtime v1 Max models
for complete low-latency voice agents. Those models manage the agent's LLM,
speech-to-text, and text-to-speech configuration inside ElevenLabs. Selecting
that mode would replace the current Qwen brain unless we built a separate
ElevenLabs agent and tool bridge. It is therefore documented as an alternative,
not silently substituted for the Qwen architecture.

Official references:

- https://elevenlabs.io/docs/eleven-agents/customization/realtime-models
- https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets
- https://elevenlabs.io/docs/api-reference/text-to-speech/stream
- https://elevenlabs.io/docs/help-center/other/what-languages-do-you-support

## Runtime behavior

tts.engine is now auto:

1. If ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are configured, ME3 uses
   ElevenLabs streaming TTS.
2. If either value is absent, ME3 uses the existing local Piper voice.
3. ME3_TTS_ENGINE=piper forces the offline path.
4. ME3_TTS_ENGINE=elevenlabs requires both ElevenLabs values and fails clearly
   if either is missing.

When `tts.engine=auto` selects ElevenLabs, the adapter uses one bounded retry
for timeout, rate-limit, and provider-server errors. If the provider fails
before producing its first PCM chunk, ME3 loads Piper and continues the turn.
A failure after partial audio is surfaced instead of replaying the sentence
from the beginning, which prevents duplicated speech. Explicit
`ME3_TTS_ENGINE=elevenlabs` remains strict and does not silently switch
providers.

The backend calls the official streaming endpoint with:

- model_id=eleven_flash_v2_5, selected for low-latency voice-agent use;
- language_code=fil for the Philippines-market profile; Filipino is supported
  by Flash v2.5, and the voice should be selected for the intended local
  accent from the operator's approved ElevenLabs library;
- output_format=pcm_16000, avoiding MP3 decoding on the Pi;
- enable_logging=false, requesting the provider's zero-retention request mode;
- a voice ID supplied by the runtime environment.

The API key is never stored in config.yaml, Git, release archives, logs, or
Qwen tool results. The Pi installer accepts the key from the runtime
environment or from the operator-managed file
~/.config/me3/elevenlabs-api-key.txt, writes the service environment file
with mode 600, and reports only configured/not-configured status.

The network-free `health()` tool reports the selected backend,
`elevenlabs_configured`, whether automatic fallback is active, sample rate,
and the last safe backend event. It does not make a provider request. The
agent logs backend, attempt count, latency, sample rate, and byte count; it
never logs the key or the spoken text as part of the provider event.

## Configure the real voice

Do not paste the key into chat or commit it. On the Pi, use the operator-only
secret channel to create the protected key file, set the voice ID, and rerun
the installer:

    mkdir -p ~/.config/me3
    chmod 700 ~/.config/me3
    # Place the ElevenLabs key in ~/.config/me3/elevenlabs-api-key.txt
    chmod 600 ~/.config/me3/elevenlabs-api-key.txt
    export ELEVENLABS_VOICE_ID=<voice-id>
    export ME3_TTS_ENGINE=elevenlabs
    export ME3_ELEVENLABS_LANGUAGE_CODE=fil
    bash install.sh

For automatic fallback behavior, omit ME3_TTS_ENGINE; auto selects ElevenLabs
only when both the key and voice ID are present.

The Windows development checkout now has the ElevenLabs key available through
the CLI-created, gitignored `.env`; the key remains process-local and is never
printed or committed. The voice ID is intentionally not persisted by this
project and must be supplied by the operator in the runtime environment. A
local provider-live acceptance call was completed on 2026-10-26 with an
operator-selected test voice: the API returned raw PCM at 16 kHz and the
Filipino-language test phrase played through the Windows default speaker.
The selected account voices were labelled English, so this proves the
integration and `language_code=fil` request path, not a native Filipino-accent
voice approval. The Pi remains offline and has not received this local-only
credential state.

The streaming adapter reassembles 16-bit PCM when an HTTP chunk ends halfway
through a sample, and classifies/retries `requests` chunked-encoding failures
through the correct exception namespace. This is covered by the unit tests in
`tests/test_elevenlabs_tts.py`.

## Readiness boundary

This integration does not claim that Eleven Realtime is already active. It
claims that ME3 can use ElevenLabs to voice the existing Qwen/tool agent. The
local provider and speaker portion is now verified, but a complete Qwen-backed
duplex acceptance test must still record:

- ElevenLabs HTTP status and first-audio-byte latency without printing the key;
- a spoken Qwen response from a reachable remote brain;
- barge-in while ElevenLabs audio is playing;
- a Qwen tool call that creates/verifies a real artifact;
- fallback to Piper after an ElevenLabs timeout or missing credential;
- a `health()` record showing which backend actually produced the audio.

The current local Qwen attempt timed out before inference at the configured
remote endpoint, so the provider-live result must not be presented as a full
Qwen voice-agent turn. The earlier Qwen/tool and physical Piper barge-in
results remain separate evidence lanes.

## Pi audio gate

Before enabling the user service or running a voice benchmark, run the
read-only preflight from the deployed project:

    python scripts/pi_audio_preflight.py

It reports ALSA capture/playback enumeration separately from PortAudio device
enumeration. The current Pi reports the connected Fifine USB capture device;
the physical Pi speaker remains a separate unverified gate. For the Windows
engineering driver, use the calibration-only gate before trials:

    python -m me3.bench_driver --brain stub --preflight

The command must report a calibrated loopback round trip. If it reports that
the tone was not detected, no latency trials should be run or published.

When the Pi speaker is unavailable, run the injection-only evaluator instead:

    python scripts/injection_eval.py \
      --input-wav runs/injection_eval_inputs/tell_me_system_status.wav \
      --interrupt-wav runs/injection_eval_inputs/stop.wav \
      --brain remote

This uses a virtual discard sink for TTS and injects the input WAV at real-time
frame cadence. It can verify Qwen/tool calls and barge-in state transitions,
but the resulting artifact is explicitly not physical-speaker or paper-native
v1.5 evidence.
