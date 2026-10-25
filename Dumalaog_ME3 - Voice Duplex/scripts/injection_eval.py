"""Injection-only ME3 voice-agent evaluation.

Feeds recorded mono PCM into the real LocalVoiceLoop at wall-clock cadence,
keeps VAD/ASR/Qwen/tool calls/TTS/barge-in intact, and replaces the physical
speaker with a virtual discard sink. This is an engineering evaluation lane;
it is deliberately not physical-speaker or paper-native FDB v1.5 evidence.

Example on the Pi (after copying non-secret WAV fixtures to the active release):
    python scripts/injection_eval.py \
      --input-wav runs/injection_eval/input.wav \
      --interrupt-wav runs/injection_eval/stop.wav \
      --brain remote
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from me3.agent import LocalVoiceLoop, StubBrain  # noqa: E402
from me3.audio_io import InjectedDuplexAudio, resample_to  # noqa: E402
from me3.brain import ToolBrain, build_system_prompt  # noqa: E402
from me3.config import load_config  # noqa: E402

log = logging.getLogger("me3.injection_eval")


def load_wav16(path: Path) -> np.ndarray:
    """Load mono int16 WAV and resample it to the ME3 16-kHz pipeline rate."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return resample_to(pcm, rate, 16000) if rate != 16000 else pcm


def topology() -> dict:
    """Return safe PortAudio topology; never returns environment values."""
    try:
        import sounddevice as sd
        devices = list(sd.query_devices())
        rows = []
        for i, d in enumerate(devices):
            rows.append({
                "index": i,
                "name": str(d.get("name", "")),
                "max_input_channels": int(d.get("max_input_channels", 0)),
                "max_output_channels": int(d.get("max_output_channels", 0)),
                "default_samplerate": d.get("default_samplerate"),
            })
        return {"portaudio_ok": True, "devices": rows}
    except Exception as exc:  # noqa: BLE001
        return {"portaudio_ok": False, "devices": [],
                "error_class": type(exc).__name__}


def _tool_event(events: list[dict], kind: str, name: str, payload) -> None:
    """Record only tool identity and success; never arguments/results."""
    if kind == "start":
        events.append({"kind": "start", "name": name,
                       "time_abs": time.perf_counter()})
    else:
        ok = payload.get("ok") if isinstance(payload, dict) else True
        events.append({"kind": "finish", "name": name, "ok": bool(ok),
                       "time_abs": time.perf_counter()})


def _feed_frame(loop: LocalVoiceLoop, frame: np.ndarray, events: list[dict]):
    now = time.perf_counter()
    for p in loop._vad_probs(frame):
        if p >= loop.vad_thr:
            events.append({"kind": "vad_speech", "time_abs": now,
                           "probability": round(float(p), 4)})
        loop._on_vad_frame(p, frame)


def evaluate(args) -> tuple[dict, Path]:
    cfg = load_config()
    cfg.setdefault("agent", {})["ack_wait_s"] = 0.0
    primary = load_wav16(Path(args.input_wav))
    interrupt = load_wav16(Path(args.interrupt_wav)) \
        if args.interrupt_wav else None

    tool_events: list[dict] = []
    if args.brain == "remote":
        brain_cfg = dict(cfg["llm"]["remote"])
        brain_cfg["system_prompt"] = build_system_prompt(
            cfg["agent"]["system_prompt"])
        brain = ToolBrain(
            brain_cfg, brain_cfg["system_prompt"],
            on_tool=lambda kind, name, payload:
                _tool_event(tool_events, kind, name, payload),
        )
    else:
        brain = StubBrain()

    loop = LocalVoiceLoop(cfg, brain=brain)
    virtual = InjectedDuplexAudio(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        frame_ms=int(cfg["audio"]["frame_ms"]),
    )
    loop.audio = virtual
    loop._load_engines()
    virtual.start_capture()

    boundaries: list[dict] = []
    original_end = loop._end_turn

    def wrapped_end(t):
        boundaries.append({"kind": "endpoint", "time_abs": time.perf_counter()})
        return original_end(t)

    loop._end_turn = wrapped_end
    tail_ms = int(cfg["vad"].get("min_silence_ms", 450)) + 300
    primary_start = virtual.inject(primary, label="primary", tail_ms=tail_ms)
    interrupt_scheduled = False
    interrupt_start = None
    started = time.perf_counter()
    deadline = started + float(args.max_seconds)

    try:
        while time.perf_counter() < deadline:
            if (interrupt is not None and not interrupt_scheduled
                    and len(virtual.output_events) >= 1
                    and loop.state == "speaking"):
                target = time.perf_counter() + float(args.interrupt_delay)
                interrupt_start = virtual.inject(
                    interrupt, label="interrupt", start_at=target,
                    tail_ms=tail_ms)
                interrupt_scheduled = True
                log.info("scheduled injected interrupt at +%.2f s",
                         target - started)

            frame = virtual.read_frame(timeout=0.05)
            if frame is not None:
                _feed_frame(loop, frame, boundaries)

            if loop.state == "speaking" and virtual.playback_done():
                loop.state = "idle"

            expected_outputs = 2 if interrupt is not None else 1
            if (len(virtual.output_events) >= expected_outputs
                    and loop.state == "idle"
                    and virtual.injections_done()
                    and virtual.playback_done()):
                break
    finally:
        virtual.close()

    elapsed = time.perf_counter() - started
    users = [h.get("content", "") for h in loop.history
             if h.get("role") == "user"]
    replies = [h.get("content", "") for h in loop.history
               if h.get("role") == "assistant"]
    tool_names = [e["name"] for e in tool_events if e["kind"] == "start"]
    barge = bool(loop.stats.n_barge_in)
    result = {
        "status": "pass" if (virtual.output_events and not loop.stats.n_errors)
        else "fail",
        "mode": "injection_only",
        "physical_speaker_exercised": False,
        "physical_microphone_exercised": False,
        "input_wav": str(Path(args.input_wav).resolve()),
        "interrupt_wav": (str(Path(args.interrupt_wav).resolve())
                           if args.interrupt_wav else None),
        "brain": args.brain,
        "host": platform.node(),
        "python": platform.python_version(),
        "topology_at_start": topology(),
        "primary_injection": {
            "start_abs": primary_start,
            "duration_s": round(len(primary) / virtual.sr, 3),
        },
        "interrupt_injection": ({
            "start_abs": interrupt_start,
            "duration_s": round(len(interrupt) / virtual.sr, 3),
        } if interrupt is not None and interrupt_start is not None else None),
        "elapsed_s": round(elapsed, 3),
        "turns": loop.stats.n_turns,
        "errors": loop.stats.n_errors,
        "barge_in_detected": barge,
        "tool_calls": sorted(set(tool_names)),
        "tool_events": [
            {k: v for k, v in e.items() if k != "time_abs"}
            for e in tool_events
        ],
        "transcripts": users,
        "replies": replies,
        "boundaries": [
            {k: v for k, v in e.items() if k != "time_abs"}
            for e in boundaries
        ],
        "virtual_output_events": [
            {k: v for k, v in e.items() if k not in {"started_abs", "stopped_abs"}}
            for e in virtual.output_events
        ],
        "limitations": [
            "Input is injected PCM, not a live microphone capture stream.",
            "TTS samples are discarded by a virtual sink; no physical speaker was exercised.",
            "This is not paper-native FULL-DUPLEX-BENCH v1.5 evidence.",
        ],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = PROJECT / "runs" / "injection_eval" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    summary = [
        "# ME3 injection-only voice-agent evaluation",
        "",
        f"- Status: **{result['status']}**",
        "- Mode: injection-only; physical speaker intentionally unused",
        f"- Brain: {args.brain}",
        f"- Turns: {result['turns']} | errors: {result['errors']}",
        f"- Tool calls: {', '.join(result['tool_calls']) or '-'}",
        f"- Barge-in detected: {result['barge_in_detected']}",
        "- Physical-speaker and paper-native FDB v1.5 claims: not made",
        "",
        "The input WAV was delivered to the same VAD/ASR/brain/TTS control path",
        "at the configured frame cadence. TTS was synthesized into a virtual",
        "discard sink, allowing injected interrupt audio to test the barge-in",
        "state transition without opening a speaker device.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n",
                                          encoding="utf-8")
    return result, out_dir


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--interrupt-wav", default=None)
    parser.add_argument("--interrupt-delay", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--brain", choices=("remote", "stub"), default="remote")
    args = parser.parse_args()
    result, out_dir = evaluate(args)
    print(json.dumps({
        "status": result["status"],
        "mode": result["mode"],
        "turns": result["turns"],
        "errors": result["errors"],
        "tool_calls": result["tool_calls"],
        "barge_in_detected": result["barge_in_detected"],
        "artifacts": str(out_dir),
    }, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
