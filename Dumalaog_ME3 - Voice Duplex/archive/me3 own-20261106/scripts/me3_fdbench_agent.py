"""Connect the real ME3 LocalVoiceLoop to the integrated fd-benchmark.

This is the ME3 adapter for the public ``ai231-me3-full-duplex-benchmarking``
tool.  It uses the benchmark's direct-feed WebSocket protocol: fdbench sends
16-kHz mono int16 microphone frames, while ME3 sends generated PCM back and
emits ``{"event":"clear"}`` on barge-in.  No microphone or speaker is opened
in this mode, so the evaluator is deterministic and safe to run on Windows or
on a Pi. Benchmark mode also disables ME3 tool calls, external harness/agent
dispatch, and filler/backchannel audio so latency measures the voice path
rather than optional task side effects.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from me3_bootstrap import configure_me3_path  # noqa: E402

configure_me3_path(PROJECT)

from me3.agent import LocalVoiceLoop, StubBrain  # noqa: E402
from me3.brain import build_system_prompt, build_tool_brain  # noqa: E402
from me3.config import load_config  # noqa: E402

log = logging.getLogger("me3.fdbench")


class FdbenchAudio:
    """Audio transport matching the LocalVoiceLoop API over one WebSocket."""

    def __init__(self, ws, sample_rate: int = 16000, frame_ms: int = 30):
        self.ws = ws
        self.sr = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_n = int(self.sr * self.frame_ms / 1000)
        self.in_idx = "fdbench"
        self.out_idx = "fdbench"
        self.io_in_rate = self.sr
        self.io_out_rate = self.sr
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._closed = threading.Event()
        self._playing = False
        self._play_until = 0.0
        self._lock = threading.Lock()
        self._receiver = threading.Thread(target=self._receive, daemon=True,
                                          name="me3-fdbench-recv")

    def start_capture(self):
        self._receiver.start()

    def startup_signal_check(self, seconds: float = 1.5) -> bool:
        return True

    def _receive(self):
        try:
            for message in self.ws:
                if not isinstance(message, (bytes, bytearray)):
                    continue
                audio = np.frombuffer(message, dtype="<i2").copy()
                for start in range(0, len(audio), self.frame_n):
                    frame = audio[start:start + self.frame_n]
                    if len(frame) < self.frame_n:
                        frame = np.pad(frame, (0, self.frame_n - len(frame)))
                    self._q.put(np.ascontiguousarray(frame, dtype=np.int16))
        except Exception:  # noqa: BLE001 - fdbench closes after the run
            log.info("fdbench WebSocket closed")
        finally:
            self._closed.set()

    def read_frame(self, timeout: float = 0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def play(self, pcm: np.ndarray, blocking: bool = False):
        x = np.asarray(pcm, dtype=np.int16).ravel()
        duration = float(len(x) / self.sr)
        with self._lock:
            self._playing = True
            self._play_until = time.perf_counter() + duration
        self.ws.send(x.astype("<i2", copy=False).tobytes())
        if blocking:
            while not self.playback_done():
                time.sleep(0.02)

    def playback_done(self) -> bool:
        with self._lock:
            if self._playing and time.perf_counter() >= self._play_until:
                self._playing = False
            return not self._playing

    def stop_playback(self, silent: bool = False):
        with self._lock:
            self._playing = False
            self._play_until = time.perf_counter()
        try:
            self.ws.send(json.dumps({"event": "clear"}))
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        self._closed.set()
        self.stop_playback(silent=True)
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def make_brain(cfg: dict, mode: str, *, benchmark_mode: bool = True):
    if mode == "stub":
        return StubBrain()
    selection = "qwen" if mode == "remote" else mode
    prompt = build_system_prompt(cfg["agent"]["system_prompt"])
    if benchmark_mode:
        prompt += (
            "\n\nBenchmark mode is active. Do not call tools, external agents, "
            "or harnesses. Return one concise spoken response only. Do not "
            "emit a filler, backchannel, or progress announcement."
        )
    return build_tool_brain(
        cfg, selection, prompt,
        on_tool=lambda kind, name, payload: log.info(
            "tool %s %s ok=%s", kind, name,
            payload.get("ok") if isinstance(payload, dict) else True),
        allow_tools=not benchmark_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8791")
    parser.add_argument("--brain", choices=("auto", "remote", "qwen", "luna", "stub"),
                        default="auto")
    parser.add_argument("--variant", choices=("current", "qwen_instruct"),
                        default="current")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-tools", action="store_true",
                        help="opt in to tools; class benchmark mode keeps them disabled")
    args = parser.parse_args()
    if args.check:
        import websockets
        print(f"ME3 fdbench adapter ready; websockets={websockets.__version__}; url={args.url}")
        return 0
    os.environ["ME3_VARIANT"] = args.variant
    cfg = load_config()
    try:
        brain = make_brain(cfg, args.brain,
                           benchmark_mode=not args.allow_tools)
    except RuntimeError:
        if args.brain != "auto":
            raise
        brain = StubBrain()
    from websockets.sync.client import connect
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    log.info("connecting to fdbench at %s", args.url)
    with connect(args.url, max_size=None, compression=None, ping_interval=20) as ws:
        audio = FdbenchAudio(ws, sample_rate=16000,
                             frame_ms=int(cfg["audio"]["frame_ms"]))
        loop = LocalVoiceLoop(cfg, brain=brain, audio=audio)
        thread = threading.Thread(target=loop.run, daemon=True,
                                  name="me3-fdbench-loop")
        thread.start()
        while thread.is_alive() and not audio._closed.wait(0.25):
            pass
        loop.stop()
        thread.join(timeout=10)
        audio.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
