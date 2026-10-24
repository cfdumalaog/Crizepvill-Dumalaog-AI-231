"""ME3 ASR MCP Server.

Exposes SOTA speech-to-text (faster-whisper) as MCP tools so any MCP client
(the Pipecat agent, Codex, Antigravity) can transcribe audio on the RPi.

Tools:
  - transcribe_pcm(base64 pcm16, sample_rate) -> {text, duration_s, latency_ms}
  - transcribe_file(path) -> {text, duration_s, latency_ms}
  - health() -> {ok, model, compute_type}

Run:  python -m me3.mcp_asr
"""
from __future__ import annotations

import base64
import io
import time

from fastmcp import FastMCP

from .config import load_config

mcp = FastMCP("me3-asr")

_engine = None
_cfg = None


def _get_engine():
    """Lazy-load faster-whisper (heavy; load once)."""
    global _engine, _cfg
    if _engine is None:
        from faster_whisper import WhisperModel
        _cfg = load_config()["asr"]
        t0 = time.perf_counter()
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
        language=cfg.get("language", "en"),
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
    t0 = time.perf_counter()
    segments, info = engine.transcribe(
        path,
        beam_size=cfg.get("beam_size", 1),
        language=cfg.get("language", "en"),
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
            "engine": "faster-whisper",
            "model": cfg["model"],
            "compute_type": cfg.get("compute_type", "int8"),
            "load_ms": round(getattr(engine, "_load_ms", 0.0), 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
