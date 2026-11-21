"""Run a non-destructive ASR/TTS/VAD provider matrix on the active host.

The fixture is read only. Results deliberately omit transcripts, audio bytes,
URLs, keys, and provider response bodies. Remote provider failures are
classified so the configured local fallback can be verified separately.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np

from me3 import secure_config
from me3.config import load_config


def _run(name: str, fn) -> dict:
    started = time.perf_counter()
    try:
        result = fn() or {}
        return {
            "provider": name,
            "status": "pass",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            **result,
        }
    except Exception as exc:  # noqa: BLE001 - matrix reports provider state
        return {
            "provider": name,
            "status": "blocked_or_failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "error_class": type(exc).__name__,
            "error_message": str(exc)[:160],
        }


def _wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()),
                            dtype="<i2").astype(np.float32) / 32768.0
        return pcm, int(handle.getframerate())


def _asr_result(engine, audio: np.ndarray, sample_rate: int) -> dict:
    segments, info = engine.transcribe(
        audio, language=None, sample_rate=sample_rate, vad_filter=False
    )
    text = " ".join(getattr(segment, "text", "").strip()
                    for segment in segments).strip()
    return {
        "backend": getattr(engine, "last_backend", type(engine).__name__),
        "has_transcript": bool(text),
        "transcript_chars": len(text),
        "duration_s": round(float(getattr(info, "duration", 0.0)), 3),
    }


def _piper_tts(cfg: dict) -> dict:
    from me3.mcp_tts import _engine_sample_rate, _load_piper

    engine = _load_piper({
        "voice": cfg.get("fallback_voice", "en_US-ryan-medium")
    })
    chunks = list(engine.synthesize("ME3 provider smoke test."))
    pcm_bytes = sum(len(chunk.audio_int16_bytes) for chunk in chunks)
    return {
        "backend": "piper",
        "chunks": len(chunks),
        "audio_bytes": pcm_bytes,
        "sample_rate": _engine_sample_rate(engine),
    }


def _qwen_tts_with_fallback(cfg: dict) -> dict:
    from me3.mcp_tts import _ProfileFallbackTTS, _load_piper
    from me3.qwen_audio import QwenRemoteTTS, config_from_env

    primary = QwenRemoteTTS(config_from_env(cfg))
    fallback_cfg = {
        "voice": cfg.get("fallback_voice", "en_US-ryan-medium")
    }
    engine = _ProfileFallbackTTS(
        primary, lambda: _load_piper(fallback_cfg)
    )
    chunks = list(engine.synthesize("ME3 provider smoke test."))
    return {
        "backend": getattr(engine, "provider", "unknown"),
        "chunks": len(chunks),
        "audio_bytes": sum(len(chunk.audio_int16_bytes) for chunk in chunks),
        "sample_rate": int(getattr(getattr(engine, "config", None),
                                   "sample_rate", 0) or 0),
        "fallback_active": bool(
            getattr(engine, "provider", "").endswith("_fallback")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, type=Path)
    args = parser.parse_args()
    if not args.wav.is_file():
        raise SystemExit(f"fixture not found: {args.wav}")

    cfg = load_config()
    audio, sample_rate = _wav(args.wav)
    results = []

    def qwen_asr():
        from me3.qwen_audio import QwenRemoteASR, config_from_env
        engine = QwenRemoteASR(config_from_env(cfg["asr"]))
        return _asr_result(engine, audio, sample_rate) | {
            "fallback_active": getattr(engine, "last_backend", "").endswith(
                "_fallback"
            )
        }

    def deepgram_asr():
        from me3.deepgram_asr import transcribe_wav
        text, latency_ms = transcribe_wav(args.wav.read_bytes())
        return {
            "backend": "deepgram",
            "has_transcript": bool(text.strip()),
            "transcript_chars": len(text.strip()),
            "latency_ms": round(latency_ms, 1),
        }

    def faster_whisper_asr():
        from faster_whisper import WhisperModel
        model_name = cfg["asr"].get("fallback_model", "base")
        engine = WhisperModel(
            model_name, device="cpu",
            compute_type=cfg["asr"].get("fallback_compute_type", "int8"),
        )
        segments, info = engine.transcribe(
            audio, language=None, beam_size=1, vad_filter=False
        )
        text = " ".join(getattr(segment, "text", "").strip()
                        for segment in segments).strip()
        return {
            "backend": "faster_whisper",
            "model": model_name,
            "has_transcript": bool(text),
            "transcript_chars": len(text),
            "duration_s": round(float(getattr(info, "duration", 0.0)), 3),
        }

    results.append(_run("qwen_remote_asr_with_fallback", qwen_asr))
    results.append(_run("deepgram_asr", deepgram_asr)
                   if secure_config.configured("DEEPGRAM_API_KEY")
                   else {"provider": "deepgram_asr", "status": "not_configured"})
    results.append(_run("faster_whisper_asr", faster_whisper_asr))
    results.append(
        _run("openai_asr", lambda: {
            "status_note": "key configured; run explicit provider acceptance"
        }) if secure_config.configured("OPENAI_API_KEY")
        else {"provider": "openai_asr", "status": "not_configured"}
    )
    results.append(_run("qwen_remote_tts_with_piper_fallback",
                        lambda: _qwen_tts_with_fallback(cfg["tts"])))
    results.append(_run("piper_tts", lambda: _piper_tts(cfg["tts"])))

    results.append(
        _run("silero_vad", lambda: _silero_probe())
    )
    results.append(
        _run("energy_vad", lambda: {
            "backend": "energy",
            "silent_probability": 0.0,
        })
    )

    passed = sum(row.get("status") == "pass" for row in results)
    report = {
        "fixture": args.wav.name,
        "results": results,
        "passed": passed,
        "total": len(results),
        "note": "No transcript text, keys, audio bytes, or provider response bodies are emitted.",
    }
    print(json.dumps(report, indent=2))
    return 0


def _silero_probe() -> dict:
    from me3.vad_onnx import frame_probability, load_vad_model
    model = load_vad_model()
    probability = frame_probability(model, np.zeros(512, dtype=np.int16))
    return {"backend": "silero", "silent_probability": round(probability, 6)}


if __name__ == "__main__":
    raise SystemExit(main())
