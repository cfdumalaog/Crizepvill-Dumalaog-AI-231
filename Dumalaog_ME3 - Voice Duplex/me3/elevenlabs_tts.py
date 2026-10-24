"""Optional ElevenLabs streaming TTS backend for the ME3 voice loop.

ME3 keeps Qwen as the reasoning/tool-calling model and uses ElevenLabs only
as the voice renderer. The backend requests raw 16 kHz PCM so the existing
full-duplex audio path can play it and stop it immediately on barge-in.

The API key and voice ID are read from environment variables only. They are
never logged, written to project files, or included in tool results.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from urllib.parse import quote

import requests


class AudioChunk:
    """Small Piper-compatible wrapper consumed by LocalVoiceLoop."""

    __slots__ = ("audio_int16_bytes",)

    def __init__(self, audio_int16_bytes: bytes):
        self.audio_int16_bytes = audio_int16_bytes


class ElevenLabsTTS:
    """Streaming ElevenLabs TTS adapter with a Piper-compatible interface."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        base_url: str = "https://api.elevenlabs.io/v1",
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_16000",
        timeout_s: float = 30.0,
        request_fn=None,
    ):
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")
        if not voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID is not configured")
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.output_format = output_format
        self.timeout_s = float(timeout_s)
        self.request_fn = request_fn or requests.post
        self.config = SimpleNamespace(sample_rate=self._sample_rate(output_format))
        self._load_ms = 0.0

    @staticmethod
    def _sample_rate(output_format: str) -> int:
        try:
            return int(output_format.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return 16000

    @classmethod
    def from_config(cls, cfg: dict) -> "ElevenLabsTTS":
        options = cfg.get("elevenlabs", {})
        key_env = options.get("api_key_env", "ELEVENLABS_API_KEY")
        voice_env = options.get("voice_id_env", "ELEVENLABS_VOICE_ID")
        return cls(
            api_key=os.environ.get(key_env, ""),
            voice_id=os.environ.get(voice_env, options.get("voice_id", "")),
            base_url=options.get("base_url", "https://api.elevenlabs.io/v1"),
            model_id=options.get("model_id", "eleven_flash_v2_5"),
            output_format=options.get("output_format", "pcm_16000"),
            timeout_s=options.get("timeout_s", 30),
        )

    def synthesize(self, text: str):
        """Yield raw PCM chunks as objects compatible with PiperVoice."""
        text = str(text or "").strip()
        if not text:
            return
        url = f"{self.base_url}/text-to-speech/{quote(self.voice_id, safe='')}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/pcm",
            "content-type": "application/json",
        }
        params = {"output_format": self.output_format, "enable_logging": "false"}
        body = {"text": text, "model_id": self.model_id}
        response = self.request_fn(
            url,
            headers=headers,
            params=params,
            json=body,
            stream=True,
            timeout=self.timeout_s,
        )
        try:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    yield AudioChunk(bytes(chunk))
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
