"""Browser-speaker transport for the ME2-style Pi web experience.

The Raspberry Pi remains the capture host.  ``BrowserOutputAudio`` delegates
mic capture to the normal PortAudio transport but publishes generated 16-kHz
mono PCM to one or more browser clients over a small WebSocket hub.  The
browser is responsible for selecting its own physical speaker through the
standard Web Audio output device picker.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np

from .audio_io import FullDuplexAudio

log = logging.getLogger("me3.web_audio")


class BrowserAudioHub:
    """Thread-safe WebSocket fan-out for raw little-endian PCM frames."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8127):
        self.host = str(host)
        self.port = int(port)
        self._server = None
        self._thread: threading.Thread | None = None
        self._clients: set = set()
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._started = threading.Event()
        self._stopped = threading.Event()
        self.sent_frames = 0
        self.sent_samples = 0
        self.dropped_frames = 0

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            from websockets.sync.server import serve
        except ImportError as exc:  # pragma: no cover - deployment diagnostic
            raise RuntimeError("browser audio needs the websockets package") from exc
        self._server = serve(
            self._handler, self.host, self.port, max_size=None,
            compression=None, ping_interval=20,
        )
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="me3-browser-audio",
        )
        self._thread.start()
        self._started.set()
        log.info("browser audio WebSocket: ws://%s:%d/audio", self.host, self.port)

    def _handler(self, conn) -> None:
        with self._lock:
            self._clients.add(conn)
        try:
            for message in conn:
                if isinstance(message, str):
                    try:
                        event = json.loads(message).get("event")
                    except (TypeError, ValueError):
                        event = None
                    if event == "ping":
                        try:
                            conn.send(json.dumps({"event": "pong"}))
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001 - browser closes are normal
            pass
        finally:
            with self._lock:
                self._clients.discard(conn)

    def publish(self, pcm: np.ndarray) -> None:
        payload = np.asarray(pcm, dtype=np.int16).ravel().astype("<i2", copy=False).tobytes()
        if not payload:
            return
        with self._lock:
            clients = list(self._clients)
        if not clients:
            self.dropped_frames += 1
            return
        dead = []
        with self._send_lock:
            for conn in clients:
                try:
                    conn.send(payload)
                except Exception:  # noqa: BLE001
                    dead.append(conn)
        if dead:
            with self._lock:
                for conn in dead:
                    self._clients.discard(conn)
        self.sent_frames += 1
        self.sent_samples += int(np.asarray(pcm).size)

    def clear(self) -> None:
        with self._lock:
            clients = list(self._clients)
        with self._send_lock:
            for conn in clients:
                try:
                    conn.send(json.dumps({"event": "clear"}))
                except Exception:  # noqa: BLE001
                    pass

    def snapshot(self) -> dict:
        with self._lock:
            clients = len(self._clients)
        return {
            "enabled": True,
            "host": self.host,
            "port": self.port,
            "clients": clients,
            "url_path": "/audio",
            "sent_frames": self.sent_frames,
            "sent_samples": self.sent_samples,
            "dropped_frames": self.dropped_frames,
        }

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self._stopped.set()


class BrowserOutputAudio:
    """Capture on the Pi and play TTS in a browser through ``hub``."""

    def __init__(self, config: dict, hub: BrowserAudioHub):
        self.capture = FullDuplexAudio(config)
        self.hub = hub
        self.sr = self.capture.sr
        self.frame_ms = self.capture.frame_ms
        self.frame_n = self.capture.frame_n
        self.in_idx = None
        self.out_idx = "browser"
        self.io_in_rate = None
        self.io_out_rate = self.sr
        self._playing = False
        self._play_until = 0.0
        self._lock = threading.Lock()
        self.output_events: list[dict] = []

    def start_capture(self):
        self.capture.start_capture()
        self.in_idx = self.capture.in_idx
        self.io_in_rate = self.capture.io_in_rate

    def startup_signal_check(self, seconds: float = 1.5) -> bool:
        return self.capture.startup_signal_check(seconds)

    def read_frame(self, timeout: float = 0.5):
        return self.capture.read_frame(timeout)

    def play(self, pcm: np.ndarray, blocking: bool = False):
        x = np.asarray(pcm, dtype=np.int16).ravel()
        started = time.perf_counter()
        duration = float(len(x) / self.sr) if self.sr else 0.0
        with self._lock:
            self._playing = True
            self._play_until = started + duration
            self.output_events.append({
                "started_abs": started,
                "samples": int(x.size),
                "duration_s": round(duration, 3),
                "sink": "browser_websocket",
                "stopped_abs": None,
            })
        self.hub.publish(x)
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
            if not self._playing:
                return
            self._playing = False
            now = time.perf_counter()
            self._play_until = now
            if self.output_events:
                self.output_events[-1]["stopped_abs"] = now
        self.hub.clear()

    def close(self):
        self.stop_playback(silent=True)
        self.capture.close()
