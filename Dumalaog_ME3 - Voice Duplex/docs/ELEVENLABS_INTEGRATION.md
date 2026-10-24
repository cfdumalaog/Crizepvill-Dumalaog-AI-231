# ElevenLabs voice integration for ME3

Version: 0.1, 25 October 2026

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

## Runtime behavior

tts.engine is now auto:

1. If ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are configured, ME3 uses
   ElevenLabs streaming TTS.
2. If either value is absent, ME3 uses the existing local Piper voice.
3. ME3_TTS_ENGINE=piper forces the offline path.
4. ME3_TTS_ENGINE=elevenlabs requires both ElevenLabs values and fails clearly
   if either is missing.

The backend calls the official streaming endpoint with:

- model_id=eleven_flash_v2_5, selected for low-latency voice-agent use;
- output_format=pcm_16000, avoiding MP3 decoding on the Pi;
- enable_logging=false, requesting the provider's zero-retention request mode;
- a voice ID supplied by the runtime environment.

The API key is never stored in config.yaml, Git, release archives, logs, or
Qwen tool results. The Pi installer accepts the key from the runtime
environment or from the operator-managed file
~/.config/me3/elevenlabs-api-key.txt, writes the service environment file
with mode 600, and reports only configured/not-configured status.

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
    bash install.sh

For automatic fallback behavior, omit ME3_TTS_ENGINE; auto selects ElevenLabs
only when both the key and voice ID are present.

The current checkout has no ElevenLabs key or voice ID configured. Therefore
the implementation and mocked stream tests are verified, while a provider-live
audio call remains pending secure credential provisioning. Until then, the
deployed Pi continues to use Piper.

## Readiness boundary

This integration does not claim that Eleven Realtime is already active. It
claims that ME3 can use ElevenLabs to voice the existing Qwen/tool agent when
the runtime credentials are provisioned. A complete live acceptance test must
record:

- ElevenLabs HTTP status and first-audio-byte latency without printing the key;
- a spoken Qwen response;
- barge-in while ElevenLabs audio is playing;
- a Qwen tool call that creates/verifies a real artifact;
- fallback to Piper after an ElevenLabs timeout or missing credential.
