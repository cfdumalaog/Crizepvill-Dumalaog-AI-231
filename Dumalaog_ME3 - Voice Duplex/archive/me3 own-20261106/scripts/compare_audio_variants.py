"""Compare the existing ME3 audio path with the optional Qwen Instruct path.

This is deliberately a provider-health and latency harness, not a claim that
the remote service is reachable. It records only model names, status, timing,
and error classes. API keys, endpoint values, audio bytes, and transcripts are
never written to the result.

Examples:
    python scripts/compare_audio_variants.py --text "Magandang araw"
    python scripts/compare_audio_variants.py --audio runs/sample.wav \
        --secret-file "..\\..\\Secrets New ME3 Model Qwen Instruct.txt"
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from me3_bootstrap import configure_me3_path  # noqa: E402
configure_me3_path(PROJECT)


def _secret_values(path: Path) -> tuple[str, str]:
    """Read the supplied handoff file in memory and return (ip, key)."""
    ip = ""
    key = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not ip:
            import re
            match = re.search(r"(?i)IP\s*Address.*?((?:\d{1,3}\.){3}\d{1,3})", line)
            if match:
                ip = match.group(1)
        if not key:
            import re
            match = re.search(r"(?i)API\s*Key.*?\bis\s+(.+?)\s*$", line)
            if match:
                key = match.group(1).strip().strip('`\"\'')
    return ip, key


def _configure_qwen(secret_file: Path | None) -> dict:
    """Set Qwen audio env vars without returning or printing secret values."""
    if secret_file and secret_file.is_file():
        ip, key = _secret_values(secret_file)
        if ip:
            os.environ.setdefault("QWEN_INSTRUCT_ASR_BASE_URL", f"http://{ip}:8000/v1")
            os.environ.setdefault("QWEN_INSTRUCT_TTS_BASE_URL", f"http://{ip}:8001/v1")
        if key:
            os.environ.setdefault("QWEN_INSTRUCT_API_KEY", key)
    return {
        "secret_file_present": bool(secret_file and secret_file.is_file()),
        "asr_endpoint_configured": bool(os.environ.get("QWEN_INSTRUCT_ASR_BASE_URL", "").strip()),
        "tts_endpoint_configured": bool(os.environ.get("QWEN_INSTRUCT_TTS_BASE_URL", "").strip()),
        "shared_key_configured": bool(os.environ.get("QWEN_INSTRUCT_API_KEY", "").strip()),
    }


def _safe_error(exc: Exception) -> dict:
    return {"status": "blocked_or_failed", "error_class": type(exc).__name__}


def _tts_runs(engine, text: str, runs: int) -> dict:
    samples = []
    errors = []
    for _ in range(max(1, runs)):
        started = time.perf_counter()
        try:
            first = None
            total_bytes = 0
            for chunk in engine.synthesize(text):
                if first is None:
                    first = (time.perf_counter() - started) * 1000
                total_bytes += len(getattr(chunk, "audio_int16_bytes", b""))
            total = (time.perf_counter() - started) * 1000
            if total_bytes <= 0:
                raise RuntimeError("provider returned no audio")
            samples.append({"ttfb_ms": round(first or total, 1),
                            "total_ms": round(total, 1),
                            "bytes": total_bytes})
        except Exception as exc:  # noqa: BLE001
            errors.append(_safe_error(exc))
    if not samples:
        return {"status": "blocked_or_failed", "samples": [],
                "errors": errors[:3]}
    return {
        "status": "pass" if not errors else "partial",
        "samples": samples,
        "median_ttfb_ms": round(statistics.median(x["ttfb_ms"] for x in samples), 1),
        "median_total_ms": round(statistics.median(x["total_ms"] for x in samples), 1),
        "errors": errors[:3],
    }


def _asr_run(engine, audio: Path) -> dict:
    started = time.perf_counter()
    try:
        # Passing an array keeps the current faster-whisper lane independent
        # of optional PyAV keyword-version drift. The Qwen wrapper accepts the
        # same float32 array and packages it as a WAV upload.
        with wave.open(str(audio), "rb") as handle:
            sample_rate = handle.getframerate()
            pcm = np.frombuffer(handle.readframes(handle.getnframes()),
                                dtype=np.int16).astype(np.float32) / 32768.0
        if getattr(engine, "base_url", None):
            segments, info = engine.transcribe(pcm, language=None,
                                                sample_rate=sample_rate)
        else:
            segments, info = engine.transcribe(
                pcm, beam_size=1, language=None, vad_filter=False)
        text = " ".join(getattr(s, "text", "").strip() for s in segments).strip()
        # Presence/length is enough for comparison; the transcript itself is
        # intentionally not persisted because fixtures can contain personal data.
        return {
            "status": "pass" if text else "empty_transcript",
            "duration_s": round(float(info.duration), 3),
            "latency_ms": round(float(getattr(info, "latency_ms", (time.perf_counter() - started) * 1000)), 1),
            "transcript_chars": len(text),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


def _load_cfg(variant: str) -> dict:
    os.environ["ME3_VARIANT"] = variant
    from me3.config import load_config
    load_config.cache_clear()
    return load_config()


def compare(args) -> dict:
    if args.secret_file:
        qwen_setup = _configure_qwen(Path(args.secret_file))
    else:
        qwen_setup = _configure_qwen(None)

    text = args.text
    result = {
        "schema": "me3.audio_variant_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text_test_chars": len(text),
        "runs": args.runs,
        "qwen_setup": qwen_setup,
        "variants": {},
        "interpretation": "Qwen Instruct changes remote ASR/TTS only; the ME3 text ToolBrain and bounded tools are shared.",
    }

    # Existing path: use the configured current ASR/TTS engines.
    os.environ["ME3_VARIANT"] = "current"
    cfg = _load_cfg("current")
    from me3 import mcp_asr, mcp_tts
    mcp_asr._engine = None
    mcp_asr._cfg = None
    mcp_tts._engine = None
    mcp_tts._cfg = None
    current = {
        "asr_model": cfg.get("asr", {}).get("model", ""),
        "tts_engine_configured": cfg.get("tts", {}).get("engine", ""),
        "tts": {},
        "asr": {"status": "skipped_no_audio"},
    }
    try:
        tts_engine, tts_cfg = mcp_tts._get_engine()
        current["tts_provider"] = getattr(tts_engine, "provider", tts_cfg.get("engine", ""))
        current["tts"] = _tts_runs(tts_engine, text, args.runs)
    except Exception as exc:  # noqa: BLE001
        current["tts"] = _safe_error(exc)
    if args.audio:
        try:
            asr_engine, _ = mcp_asr._get_engine()
            current["asr"] = _asr_run(asr_engine, Path(args.audio))
        except Exception as exc:  # noqa: BLE001
            current["asr"] = _safe_error(exc)
    result["variants"]["current"] = current

    # New path: instantiate the Qwen wrappers directly so the comparison does
    # not leave the process-level MCP caches in the wrong variant.
    qcfg = _load_cfg("qwen_instruct")
    from me3.qwen_audio import QwenRemoteASR, QwenRemoteTTS, config_from_env
    qwen_asr_cfg = config_from_env(qcfg["asr"])
    qwen_tts_cfg = config_from_env(qcfg["tts"])
    if args.timeout is not None:
        qwen_asr_cfg["timeout_s"] = args.timeout
        qwen_tts_cfg["timeout_s"] = args.timeout
    qwen_asr = QwenRemoteASR(qwen_asr_cfg)
    qwen_tts = QwenRemoteTTS(qwen_tts_cfg)
    qwen = {
        "asr_model": qcfg.get("asr", {}).get("model", ""),
        "tts_model": qcfg.get("tts", {}).get("model", ""),
        "tts": _tts_runs(qwen_tts, text, args.runs),
        "asr": (_asr_run(qwen_asr, Path(args.audio)) if args.audio
                else {"status": "skipped_no_audio"}),
    }
    result["variants"]["qwen_instruct"] = qwen
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=None,
                        help="optional mono WAV fixture for matched ASR")
    parser.add_argument("--text", default="Magandang araw. Kumusta ka ngayon?",
                        help="same text sent to both TTS providers")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Qwen audio request timeout in seconds")
    parser.add_argument("--secret-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.audio and not args.audio.is_file():
        raise SystemExit(f"audio fixture not found: {args.audio}")
    os.environ.setdefault("ME3_TTS_ENGINE", "piper")
    result = compare(args)
    output = args.output or (PROJECT / "runs" / "audio_variant_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "current_tts": result["variants"]["current"]["tts"].get("status"),
        "qwen_tts": result["variants"]["qwen_instruct"]["tts"].get("status"),
        "current_asr": result["variants"]["current"]["asr"].get("status"),
        "qwen_asr": result["variants"]["qwen_instruct"]["asr"].get("status"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
