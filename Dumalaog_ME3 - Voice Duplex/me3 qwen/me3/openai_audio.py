"""OpenAI file-transcription and PCM speech adapters for the Pi MCP servers."""
from __future__ import annotations

import io
import mimetypes
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

from .provider_choice import openai_key

BASE_URL = "https://api.openai.com/v1"


def _key() -> str:
    key = openai_key()
    if not key:
        raise RuntimeError("OpenAI key is not configured; use the private ME3 setup page")
    return key


class OpenAIASR:
    last_backend = "openai"

    def __init__(self, model: str = "gpt-transcribe"):
        self.model = model
        self._load_ms = 0.0

    def transcribe(self, audio, *, language=None, sample_rate=16000, **_kwargs):
        if isinstance(audio, (str, Path)):
            source = Path(audio)
            data = source.read_bytes()
            name = source.name
            duration = 0.0
            if source.suffix.lower() == ".wav":
                with wave.open(io.BytesIO(data), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
        else:
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
            duration = len(pcm) / sample_rate
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
            data = buffer.getvalue()
            name = "turn.wav"
        started = time.perf_counter()
        response = requests.post(
            f"{BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {_key()}"},
            data={"model": self.model},
            files={"file": (name, data, mimetypes.guess_type(name)[0] or "application/octet-stream")},
            timeout=(5, 45),
        )
        response.raise_for_status()
        text = str(response.json().get("text") or "").strip()
        return ([SimpleNamespace(text=text)] if text else []), SimpleNamespace(
            duration=duration, latency_ms=(time.perf_counter() - started) * 1000)


class OpenAITTS:
    provider = "openai-tts"
    model = "gpt-4o-mini-tts"
    voice = "onyx"

    def __init__(self):
        self.config = SimpleNamespace(sample_rate=24000)
        self.last_event = None
        self._load_ms = 0.0

    def synthesize(self, text: str):
        if not str(text or "").strip():
            return
        started = time.perf_counter()
        response = requests.post(
            f"{BASE_URL}/audio/speech",
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "input": str(text), "voice": self.voice,
                  "instructions": "Speak in a steady, formal, professional tone. Use English by default; speak Filipino only when the user requests it.",
                  "response_format": "pcm"},
            timeout=(5, 60),
        )
        response.raise_for_status()
        pcm = response.content
        if not pcm or len(pcm) % 2:
            raise RuntimeError("OpenAI TTS returned empty or invalid PCM audio")
        self.last_event = {"backend": self.provider, "ok": True,
                           "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                           "bytes": len(pcm), "model": self.model,
                           "voice_requested": self.voice}
        yield SimpleNamespace(audio_int16_bytes=pcm)
