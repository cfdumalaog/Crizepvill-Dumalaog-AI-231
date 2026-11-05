"""Remote Qwen audio adapters used by the optional ME3 variant.

The existing ME3 release keeps VAD local and uses faster-whisper plus Piper or
ElevenLabs.  The ``qwen_instruct`` variant swaps only the ASR/TTS engines for
the protected DGX3 OpenAI-compatible services:

* Qwen2-Audio-7B-Instruct through vLLM's multimodal
  ``/v1/chat/completions`` audio input;
* Qwen3-TTS through ``/v1/audio/speech``.

The text ToolBrain and bounded ME3 tool registry remain unchanged.  This is
deliberate: the supplied Qwen2-Audio service is an audio-to-text model, not a
drop-in replacement for the current text tool-calling brain.
"""
from __future__ import annotations

import base64
import io
import os
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _require_endpoint(base_url: str, api_key: str, label: str) -> None:
    missing = []
    if not base_url:
        missing.append("base URL")
    if not api_key:
        missing.append("API key")
    if missing:
        raise RuntimeError(
            f"Qwen {label} is not configured; missing {', '.join(missing)}"
        )


def _wav_bytes(audio, sample_rate: int) -> tuple[bytes, float]:
    if isinstance(audio, (str, Path)):
        raw = Path(audio).read_bytes()
        try:
            with wave.open(io.BytesIO(raw), "rb") as handle:
                return raw, handle.getnframes() / float(handle.getframerate())
        except wave.Error as exc:
            raise ValueError("remote Qwen ASR requires a WAV path or PCM array") from exc
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return buf.getvalue(), len(pcm16) / float(sample_rate)


def _json_text(payload: dict) -> str:
    text = payload.get("text") or payload.get("transcript")
    if isinstance(text, str):
        return text.strip()
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content
                if isinstance(item, dict)
            )
        if isinstance(content, str):
            return content.strip()
    return ""


class QwenRemoteASR:
    """faster-whisper-compatible wrapper around vLLM's audio chat API.

    Qwen2-Audio is an audio-language model rather than a transcription-only
    model. The supplied vLLM instance exposes it through multimodal chat
    completions, so the WAV is sent as an OpenAI-compatible ``audio_url`` data
    URL and the prompt asks for transcript-only output. A legacy
    ``transcriptions`` protocol remains available for deployments that expose
    the dedicated endpoint instead.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.model = str(cfg.get("model", ""))
        self.api_key = str(cfg.get("api_key", ""))
        self.protocol = str(cfg.get("protocol", "chat_completions"))
        self.prompt = str(cfg.get(
            "prompt", "Transcribe the spoken audio exactly. Return only the transcript."
        ))
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.timeout_s = float(cfg.get("timeout_s", 60))
        self._load_ms = 0.0

    def transcribe(self, audio, beam_size: int = 1, language: str | None = None,
                   vad_filter: bool = False, sample_rate: int | None = None):
        del beam_size, vad_filter
        _require_endpoint(self.base_url, self.api_key, "ASR")
        raw, duration = _wav_bytes(
            audio, int(sample_rate or self.cfg.get("sample_rate", 16000)))
        started = time.perf_counter()
        if self.protocol in {"transcriptions", "audio_transcriptions"}:
            data = {
                "model": self.model,
                "response_format": "json",
                "prompt": self.prompt,
            }
            if language:
                data["language"] = language
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=_headers(self.api_key),
                files={"file": ("me3_turn.wav", raw, "audio/wav")},
                data=data,
                timeout=self.timeout_s,
            )
        else:
            audio_url = "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")
            prompt = self.prompt
            if language:
                prompt = f"{prompt} The spoken language is {language}."
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={**_headers(self.api_key), "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_s,
            )
        response.raise_for_status()
        payload = response.json()
        text = _json_text(payload if isinstance(payload, dict) else {})
        info = SimpleNamespace(
            duration=duration,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return [SimpleNamespace(text=text)], info


class QwenRemoteTTS:
    """Piper-compatible wrapper around vLLM-Omni's speech API."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.model = str(cfg.get("model", ""))
        self.api_key = str(cfg.get("api_key", ""))
        self.voice = str(cfg.get("voice", "aiden"))
        self.language = str(cfg.get("language", "English"))
        self.instruct = str(cfg.get(
            "instruct",
            "Use a formal, neutral, professional customer-service delivery. "
            "Avoid flirtatious, intimate, playful, or suggestive expression.",
        ))
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.timeout_s = float(cfg.get("timeout_s", 60))
        self.config = SimpleNamespace(sample_rate=self.sample_rate)
        self.provider = "qwen3-tts"
        self.last_event = None
        self._load_ms = 0.0

    def _pcm_payload(self, content: bytes, content_type: str) -> tuple[bytes, int]:
        if content.startswith(b"RIFF") or "wav" in content_type.lower():
            with wave.open(io.BytesIO(content), "rb") as handle:
                return handle.readframes(handle.getnframes()), handle.getframerate()
        return content, self.sample_rate

    def synthesize(self, text: str):
        _require_endpoint(self.base_url, self.api_key, "TTS")
        if not str(text or "").strip():
            return
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "input": str(text),
            "voice": self.voice,
            "language": self.language,
            "instruct": self.instruct,
            "response_format": "pcm",
        }
        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={**_headers(self.api_key), "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        pcm, sample_rate = self._pcm_payload(
            response.content, response.headers.get("Content-Type", "audio/pcm")
        )
        self.sample_rate = int(sample_rate)
        self.config.sample_rate = self.sample_rate
        self.last_event = {
            "backend": self.provider,
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "bytes": len(pcm),
            # Safe runtime provenance. These fields make it possible to tell
            # whether a different browser/native voice was added downstream;
            # no credential or audio content is recorded.
            "model": self.model,
            "voice_requested": self.voice,
            "language": self.language,
        }
        yield SimpleNamespace(audio_int16_bytes=pcm)


def config_from_env(cfg: dict, *, api_key: str = "") -> dict:
    """Resolve configured endpoint/key environment names without logging values."""
    out = dict(cfg)
    base_env = out.get("base_url_env")
    if base_env:
        out["base_url"] = os.environ.get(base_env, "")
    key_env = out.get("api_key_env")
    out["api_key"] = api_key or (
        os.environ.get(key_env, "") if key_env else ""
    )
    return out

