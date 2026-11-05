"""ME3 VAD MCP Server.

Exposes Silero VAD through Pipecat's bundled CPU ONNX model (no network fetch) as
MCP tools for voice-activity detection and endpointing. Streaming-friendly:
clients feed frames, get speech events.

Tools:
  - detect_frame(frame_b64, sample_rate) -> {speech: bool, prob: float, latency_ms}
  - detect_stream(frames_b64, sample_rate) -> {events: [...], speech_ratio}
  - endpoint_state(speech_ms, silence_ms) -> {should_endpoint: bool}  (helper)
  - health() -> {ok, model, threshold}

Run:  python -m me3.mcp_vad
"""
from __future__ import annotations

import base64
import time

import numpy as np

from fastmcp import FastMCP

from .config import load_config

mcp = FastMCP("me3-vad")

_state = {"model": None, "cfg": None, "sr": 16000}


def _reset_model(model) -> None:
    """Reset internal LSTM state (stateless per-frame classification)."""
    if hasattr(model, "reset_states"):
        model.reset_states()
    elif hasattr(model, "state") and hasattr(model.state, "zero_"):
        model.state.zero_()


def _get_model():
    """Lazy-load the Pipecat-bundled Silero ONNX model."""
    if _state["model"] is None:
        from .vad_onnx import load_vad_model
        _cfg = load_config()["vad"]
        t0 = time.perf_counter()
        _state["model"] = load_vad_model()
        _state["cfg"] = _cfg
        _state["sr"] = 16000
        _state["load_ms"] = (time.perf_counter() - t0) * 1000
    return _state["model"], _state["cfg"]


def _frame_prob(frame: np.ndarray, sample_rate: int = 16000) -> float:
    from .vad_onnx import frame_probability

    model, _cfg = _get_model()
    _reset_model(model)
    return frame_probability(model, frame.astype(np.float32), sample_rate)


@mcp.tool()
def detect_frame(frame_b64: str, sample_rate: int = 16000) -> dict:
    """Detect speech in a single PCM16 mono frame.

    Args:
        frame_b64: base64 of raw int16 mono PCM frame.
        sample_rate: sample rate of the frame (default 16000).

    Returns:
        {speech: bool, prob: float, latency_ms}
    """
    model, cfg = _get_model()
    raw = base64.b64decode(frame_b64)
    frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.perf_counter()
    prob = _frame_prob(frame, sample_rate)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "speech": bool(prob >= cfg.get("threshold", 0.5)),
        "prob": round(prob, 4),
        "latency_ms": round(latency_ms, 2),
    }


@mcp.tool()
def detect_stream(frames_b64: list, sample_rate: int = 16000) -> dict:
    """Run VAD over a sequence of frames; return speech segments.

    Frames are evaluated with streaming state (LSTM carries across frames),
    so this approximates the live pipeline behavior.

    Args:
        frames_b64: list of base64 PCM16 mono frames (same sample_rate).
        sample_rate: sample rate of the frames (default 16000).

    Returns:
        {events: [{start_frame, end_frame, start_ms, end_ms, peak_prob}],
         speech_ratio}
    """
    model, cfg = _get_model()
    from .vad_onnx import frame_probability
    thr = cfg.get("threshold", 0.5)
    _reset_model(model)
    probs = []
    for fb in frames_b64:
        raw = base64.b64decode(fb)
        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        probs.append(frame_probability(model, frame, sample_rate))

    frame_ms = 1000.0 * (len(frames_b64[0]) // 2) / sample_rate if frames_b64 else 30.0
    events = []
    start = None
    peak = 0.0
    for i, p in enumerate(probs):
        if p >= thr and start is None:
            start = i
            peak = p
        elif p >= thr:
            peak = max(peak, p)
        elif start is not None:
            events.append({
                "start_frame": start, "end_frame": i - 1,
                "start_ms": round(start * frame_ms, 1),
                "end_ms": round(i * frame_ms, 1),
                "peak_prob": round(peak, 4),
            })
            start = None
            peak = 0.0
    if start is not None:
        events.append({
            "start_frame": start, "end_frame": len(probs) - 1,
            "start_ms": round(start * frame_ms, 1),
            "end_ms": round(len(probs) * frame_ms, 1),
            "peak_prob": round(peak, 4),
        })
    speech_ratio = sum(1 for p in probs if p >= thr) / max(len(probs), 1)
    return {"events": events, "speech_ratio": round(speech_ratio, 4)}


@mcp.tool()
def endpoint_state(speech_ms: float, silence_ms: float) -> dict:
    """Decide whether to end the current turn based on accumulated silence.

    Args:
        speech_ms: milliseconds of continuous speech seen.
        silence_ms: milliseconds of silence since last speech.

    Returns:
        {should_endpoint: bool, reason}
    """
    cfg = load_config()["vad"]
    min_silence = cfg.get("min_silence_ms", 450)
    min_speech = cfg.get("min_speech_ms", 120)
    if speech_ms < min_speech:
        return {"should_endpoint": False, "reason": "below_min_speech"}
    if silence_ms >= min_silence:
        return {"should_endpoint": True, "reason": "silence_threshold_met"}
    return {"should_endpoint": False, "reason": "still_speaking"}


@mcp.tool()
def health() -> dict:
    """Report VAD server health."""
    try:
        _model, cfg = _get_model()
        return {
            "ok": True,
            "engine": "silero-vad",
            "threshold": cfg.get("threshold", 0.5),
            "min_silence_ms": cfg.get("min_silence_ms", 450),
            "load_ms": round(_state.get("load_ms", 0.0), 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
