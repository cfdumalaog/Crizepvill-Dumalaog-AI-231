"""ME3 TTS MCP Server.

Exposes SOTA text-to-speech (Piper 1.8.0) as MCP tools so any MCP client can
synthesize speech on the RPi and receive audio bytes.

Tools:
  - synthesize(text) -> {audio_b64 (wav), sample_rate, duration_s, latency_ms, ttfb_ms}
  - synthesize_stream(text) -> yields wav chunks (for low-TTFB streaming)
  - voices() -> list of installed piper voices
  - health() -> {ok, voice, engine}

Run:  python -m me3.mcp_tts

Voice models live in <project>/models/tts/<voice>.onnx and are downloaded on
first use via `python -m piper.download_voices` (piper 1.8.0 CLI). No API key.
"""
from __future__ import annotations

import base64
import io
import subprocess
import sys
import time
import wave
from pathlib import Path

from fastmcp import FastMCP

from .config import load_config

mcp = FastMCP("me3-tts")

_engine = None
_cfg = None
_voice_dir = Path(__file__).resolve().parent.parent / "models" / "tts"


def _ensure_voice(voice_name: str) -> Path:
    """Locate the Piper voice .onnx, downloading it if missing."""
    _voice_dir.mkdir(parents=True, exist_ok=True)
    cand = _voice_dir / f"{voice_name}.onnx"
    if cand.exists():
        return cand
    print(f"[me3-tts] voice {voice_name} not found; downloading to {_voice_dir} ...",
          file=sys.stderr, flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "piper.download_voices", voice_name,
         "--download-dir", str(_voice_dir)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not cand.exists():
        raise RuntimeError(
            f"piper voice download failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout)[-300:]}"
        )
    return cand


def _get_engine():
    """Lazy-load Piper (heavy; load once)."""
    global _engine, _cfg
    if _engine is None:
        from piper import PiperVoice
        _cfg = load_config()["tts"]
        t0 = time.perf_counter()
        voice_path = _ensure_voice(_cfg["voice"])
        _engine = PiperVoice.load(voice_path)
        _engine._load_ms = (time.perf_counter() - t0) * 1000
    return _engine, _cfg


def _engine_sample_rate(engine) -> int:
    cfg = getattr(engine, "config", None)
    sr = getattr(cfg, "sample_rate", None)
    if sr is None:
        data = getattr(cfg, "data", None)
        sr = getattr(data, "sample_rate", None) if data is not None else None
    return int(sr or 22050)


def _pcm_to_wav_b64(pcm: bytes, sample_rate: int) -> tuple[str, float]:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode(), len(pcm) / 2 / sample_rate


@mcp.tool()
def synthesize(text: str) -> dict:
    """Synthesize text to WAV audio (base64).

    Args:
        text: text to speak.

    Returns:
        {audio_b64, sample_rate, duration_s, latency_ms, ttfb_ms}
    """
    engine, _cfg = _get_engine()
    sr = _engine_sample_rate(engine)
    t0 = time.perf_counter()
    chunks = []
    first_byte_ms = None
    for chunk in engine.synthesize(text):
        pcm_bytes = chunk.audio_int16_bytes
        if first_byte_ms is None:
            first_byte_ms = (time.perf_counter() - t0) * 1000
        chunks.append(pcm_bytes)
    pcm = b"".join(chunks)
    latency_ms = (time.perf_counter() - t0) * 1000
    audio_b64, dur = _pcm_to_wav_b64(pcm, sr)
    return {
        "audio_b64": audio_b64,
        "sample_rate": sr,
        "duration_s": round(dur, 3),
        "latency_ms": round(latency_ms, 1),
        "ttfb_ms": round(first_byte_ms or latency_ms, 1),
    }


@mcp.tool()
def synthesize_stream(text: str) -> list[dict]:
    """Synthesize text and return per-chunk WAV pieces for streaming playback.

    Returns a list of {chunk_i, audio_b64, sample_rate, ttfb_ms_first_chunk}.
    The first element's ttfb_ms is the time-to-first-audio-byte metric.
    """
    engine, _cfg = _get_engine()
    sr = _engine_sample_rate(engine)
    t0 = time.perf_counter()
    out = []
    for i, chunk in enumerate(engine.synthesize(text)):
        ttfb = (time.perf_counter() - t0) * 1000 if i == 0 else None
        b64, _dur = _pcm_to_wav_b64(chunk.audio_int16_bytes, sr)
        item = {"chunk_i": i, "audio_b64": b64, "sample_rate": sr}
        if ttfb is not None:
            item["ttfb_ms"] = round(ttfb, 1)
        out.append(item)
    return out


@mcp.tool()
def voices() -> dict:
    """List available Piper voices (downloaded under models/tts/)."""
    try:
        names = sorted(p.stem for p in _voice_dir.glob("*.onnx"))
        return {"ok": True, "voices": names, "dir": str(_voice_dir)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@mcp.tool()
def health() -> dict:
    """Report TTS server health and configured voice."""
    try:
        engine, cfg = _get_engine()
        return {
            "ok": True,
            "engine": "piper",
            "voice": cfg["voice"],
            "sample_rate": _engine_sample_rate(engine),
            "load_ms": round(getattr(engine, "_load_ms", 0.0), 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
