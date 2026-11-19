"""ME3 ASR MCP Server.

Exposes SOTA speech-to-text (faster-whisper) as MCP tools so any MCP client
(the Pipecat agent, Codex, Antigravity) can transcribe audio on the RPi.

Tools:
  - transcribe_pcm(base64 pcm16, sample_rate) -> {text, duration_s, latency_ms}
  - transcribe_file(path) -> {text, duration_s, latency_ms}
  - health() -> {ok, model, compute_type}

Run:  python scripts/run_me3_module.py me3.mcp_asr
"""
from __future__ import annotations

import base64
import io
import time
import wave
from types import SimpleNamespace

from fastmcp import FastMCP

from .config import load_config

mcp = FastMCP("me3-asr")

_engine = None
_cfg = None


class DeepgramASR:
    """Small adapter that keeps Deepgram behind the same ASR interface."""

    last_backend = "deepgram"
    fallback_model = None

    def __init__(self):
        self._load_ms = 0.0

    def transcribe(self, audio, sample_rate=16000, **_kwargs):
        import numpy as np
        pcm = np.asarray(audio, dtype=np.float32)
        pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(int(sample_rate))
            out.writeframes(pcm16)
        started = time.perf_counter()
        from .deepgram_asr import transcribe_wav
        text, _latency_ms = transcribe_wav(buf.getvalue())
        latency_ms = (time.perf_counter() - started) * 1000
        return [SimpleNamespace(text=text)], SimpleNamespace(
            duration=len(pcm16) / 2 / int(sample_rate), latency_ms=latency_ms)


def _get_engine():
    """Lazy-load faster-whisper (heavy; load once)."""
    global _engine, _cfg
    if _engine is None:
        _cfg = load_config()["asr"]
        t0 = time.perf_counter()
        if _cfg.get("engine") == "openai":
            from .openai_audio import OpenAIASR
            _engine = OpenAIASR(_cfg["model"])
        elif _cfg.get("engine") == "qwen_remote":
            from .qwen_audio import QwenRemoteASR, config_from_env
            _engine = QwenRemoteASR(config_from_env(_cfg))
        elif _cfg.get("engine") == "deepgram":
            _engine = DeepgramASR()
        else:
            from faster_whisper import WhisperModel
            _engine = WhisperModel(
                _cfg["model"],
                device="cpu",
                compute_type=_cfg.get("compute_type", "int8"),
            )
        _engine._load_ms = (time.perf_counter() - t0) * 1000
    return _engine, _cfg


@mcp.tool()
def transcribe_pcm(audio_b64: str, sample_rate: int = 16000) -> dict:
    """Transcribe base64-encoded PCM16 mono audio to text.

    Args:
        audio_b64: base64 of raw little-endian int16 mono PCM.
        sample_rate: sample rate of the audio (default 16000).

    Returns:
        {text, duration_s, latency_ms}
    """
    import numpy as np

    engine, cfg = _get_engine()
    raw = base64.b64decode(audio_b64)
    if cfg.get("engine") in {"qwen_remote", "openai", "deepgram"}:
        segments, info = engine.transcribe(
            np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0,
            language=cfg.get("language"),
            sample_rate=sample_rate,
        )
        return {
            "text": " ".join(s.text.strip() for s in segments).strip(),
            "duration_s": round(float(info.duration), 3),
            "latency_ms": round(float(info.latency_ms), 1),
        }
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != 16000:
        # naive resample via scipy if available, else require 16k
        try:
            from scipy.signal import resample_poly
            g = 16
            pcm = resample_poly(pcm, 16000 // g, sample_rate // g).astype(np.float32)
        except Exception:
            pass

    t0 = time.perf_counter()
    segments, info = engine.transcribe(
        pcm,
        beam_size=cfg.get("beam_size", 1),
        # None enables Whisper language detection for Filipino/English/
        # Taglish. English-only models are not suitable for the PH profile.
        language=cfg.get("language") or None,
        vad_filter=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "text": text,
        "duration_s": round(len(pcm) / 16000.0, 3),
        "latency_ms": round(latency_ms, 1),
    }


@mcp.tool()
def transcribe_file(path: str) -> dict:
    """Transcribe an audio file (wav/flac/mp3) to text.

    Args:
        path: filesystem path to the audio file.

    Returns:
        {text, duration_s, latency_ms}
    """
    engine, cfg = _get_engine()
    if cfg.get("engine") in {"qwen_remote", "openai", "deepgram"}:
        segments, info = engine.transcribe(path, language=cfg.get("language"))
        return {
            "text": " ".join(s.text.strip() for s in segments).strip(),
            "duration_s": round(float(info.duration), 3),
            "latency_ms": round(float(info.latency_ms), 1),
        }
    t0 = time.perf_counter()
    segments, info = engine.transcribe(
        path,
        beam_size=cfg.get("beam_size", 1),
        language=cfg.get("language") or None,
        vad_filter=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "text": text,
        "duration_s": round(float(info.duration), 3),
        "latency_ms": round(latency_ms, 1),
    }


@mcp.tool()
def health() -> dict:
    """Report ASR server health and loaded model."""
    try:
        engine, cfg = _get_engine()
        return {
            "ok": True,
            "engine": cfg.get("engine", "faster-whisper"),
            "model": cfg["model"],
            "compute_type": cfg.get("compute_type", "int8"),
            "load_ms": round(getattr(engine, "_load_ms", 0.0), 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
