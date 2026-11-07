"""Browser-speaker transport for the ME2-style Pi web experience.

The Raspberry Pi remains the capture host.  ``BrowserOutputAudio`` delegates
mic capture to the normal PortAudio transport but publishes generated 16-kHz
mono PCM to the current browser client over a small WebSocket hub.  The
browser is responsible for selecting its own physical speaker through the
standard Web Audio output device picker.

There is deliberately only one active audio output client. Additional
device-only clients (for example, the Android companion used for phone calls)
can register capabilities without receiving PCM, so they do not create extra
speaker echoes at the Pi microphone.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid

import numpy as np

from .audio_io import FullDuplexAudio

log = logging.getLogger("me3.web_audio")


class BrowserAudioHub:
    """Thread-safe single-audio-sink WebSocket transport plus action clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8127):
        self.host = str(host)
        self.port = int(port)
        self._server = None
        self._thread: threading.Thread | None = None
        self._clients: set = set()
        self._audio_client = None
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._started = threading.Event()
        self._stopped = threading.Event()
        self.sent_frames = 0
        self.sent_samples = 0
        self.dropped_frames = 0
        self._render_start_callback = None
        self._client_info: dict = {}
        self._action_lock = threading.Lock()
        self._pending_actions: dict[str, tuple[threading.Event, dict | None]] = {}
        self.render_ack_count = 0
        self.last_render_delay_ms: float | None = None

    def set_render_start_callback(self, callback) -> None:
        """Register a callback for browser-confirmed render timing.

        The callback receives the JSON ``render_started`` event sent by the
        browser immediately after it schedules the first PCM buffer.  This is
        needed for the distributed Pi-microphone/laptop-speaker route: sending
        time is not the same as acoustic render time.
        """
        self._render_start_callback = callback

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
            self._client_info[conn] = {
                "client_type": "web", "capabilities": [], "audio_output": True,
            }
            if self._audio_client is None:
                self._audio_client = conn
        log.info("ME3 browser/device client connected; clients=%d", len(self._clients))
        try:
            for message in conn:
                if isinstance(message, str):
                    try:
                        parsed = json.loads(message)
                        event_data = parsed if isinstance(parsed, dict) else {}
                        event = event_data.get("event")
                    except (TypeError, ValueError):
                        event_data = {}
                        event = None
                    if event == "ping":
                        try:
                            conn.send(json.dumps({"event": "pong"}))
                        except Exception:  # noqa: BLE001
                            pass
                    elif event == "render_started":
                        self.render_ack_count += 1
                        try:
                            self.last_render_delay_ms = round(
                                float(event_data.get("delay_ms", 0.0) or 0.0), 1)
                        except (TypeError, ValueError):
                            self.last_render_delay_ms = None
                        callback = self._render_start_callback
                        if callback is not None:
                            try:
                                callback(event_data)
                            except Exception:  # noqa: BLE001
                                log.debug("browser render acknowledgement failed",
                                          exc_info=True)
                    elif event == "client_info":
                        caps = event_data.get("capabilities") or []
                        info = {
                            "client_type": str(event_data.get("client_type") or "web")[:40],
                            "capabilities": [str(x)[:60] for x in caps[:20]],
                            "audio_output": bool(event_data.get("audio_output", True)),
                        }
                        with self._lock:
                            self._client_info[conn] = info
                            if info["audio_output"]:
                                self._audio_client = conn
                            elif self._audio_client is conn:
                                self._audio_client = next((
                                    candidate for candidate in self._clients
                                    if candidate is not conn and
                                    self._client_info.get(candidate, {}).get("audio_output")
                                ), None)
                    elif event == "device_action_result":
                        action_id = str(event_data.get("action_id") or "")
                        with self._action_lock:
                            pending = self._pending_actions.get(action_id)
                            if pending:
                                event_obj = pending[0]
                                self._pending_actions[action_id] = (event_obj, {
                                    "ok": bool(event_data.get("ok")),
                                    "status": str(event_data.get("status") or "")[:80],
                                    "error": str(event_data.get("error") or "")[:120] or None,
                                })
                                event_obj.set()
        except Exception:  # noqa: BLE001 - browser closes are normal
            pass
        finally:
            with self._lock:
                self._clients.discard(conn)
                self._client_info.pop(conn, None)
                if self._audio_client is conn:
                    self._audio_client = next((
                        candidate for candidate in self._clients
                        if self._client_info.get(candidate, {}).get("audio_output")
                    ), None)

    def has_client(self) -> bool:
        with self._lock:
            return self._audio_client is not None

    def publish(self, pcm: np.ndarray, playback_id: str | None = None) -> None:
        payload = np.asarray(pcm, dtype=np.int16).ravel().astype("<i2", copy=False).tobytes()
        if not payload:
            return
        with self._lock:
            client = self._audio_client
        if client is None:
            self.dropped_frames += 1
            return
        # Keep messages short enough that the websockets keepalive thread can
        # still service the connection while a long TTS response is queued.
        # The browser schedules these chunks back-to-back as one PCM stream.
        chunk_samples = 480  # 30 ms at the fixed 16-kHz browser output rate
        with self._send_lock:
            try:
                if playback_id:
                    client.send(json.dumps({
                        "event": "playback_start",
                        "playback_id": str(playback_id),
                        "sample_rate": 16000,
                        "samples": int(np.asarray(pcm).size),
                    }, separators=(",", ":")))
                for start in range(0, len(payload), chunk_samples * 2):
                    client.send(payload[start:start + chunk_samples * 2])
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._clients.discard(client)
                    self._client_info.pop(client, None)
                    if self._audio_client is client:
                        self._audio_client = None
        self.sent_frames += (len(payload) + chunk_samples * 2 - 1) // (chunk_samples * 2)
        self.sent_samples += int(np.asarray(pcm).size)

    def clear(self) -> None:
        with self._lock:
            client = self._audio_client
        with self._send_lock:
            if client is not None:
                try:
                    client.send(json.dumps({"event": "clear"}))
                except Exception:  # noqa: BLE001
                    pass

    def publish_event(self, event: dict) -> None:
        """Send a small JSON control event to the active browser client."""
        try:
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        with self._lock:
            client = self._audio_client
        if client is None:
            return
        with self._send_lock:
            try:
                client.send(payload)
            except Exception:  # noqa: BLE001 - browser disconnect is normal
                with self._lock:
                    self._clients.discard(client)
                    self._client_info.pop(client, None)
                    if self._audio_client is client:
                        self._audio_client = None

    def request_action(self, action: dict, timeout_s: float = 20.0) -> dict:
        """Request a bounded action from a connected capable browser client."""
        action_name = str(action.get("action") or "")
        required_capability = {"call_phone": "call_phone"}.get(action_name)
        with self._lock:
            target = next((
                conn for conn, info in self._client_info.items()
                if required_capability and
                required_capability in info.get("capabilities", [])
            ), None) if required_capability else self._audio_client
            client = dict(self._client_info.get(target, {})) if target is not None else {}
        if target is None:
            return {"ok": False, "error": "browser_device_not_connected"}
        action_id = uuid.uuid4().hex
        waiter = threading.Event()
        with self._action_lock:
            self._pending_actions[action_id] = (waiter, None)
        payload = json.dumps({"event": "device_action", "action_id": action_id,
                              "action": action_name,
                              "contact": str(action.get("contact") or "")[:160]},
                             ensure_ascii=False, separators=(",", ":"))
        try:
            with self._send_lock:
                target.send(payload)
        except Exception:
            with self._action_lock:
                self._pending_actions.pop(action_id, None)
            return {"ok": False, "error": "device_action_send_failed"}
        if not waiter.wait(max(0.5, min(float(timeout_s), 45.0))):
            with self._action_lock:
                self._pending_actions.pop(action_id, None)
            return {"ok": False, "error": "device_action_timeout"}
        with self._action_lock:
            _waiter, result = self._pending_actions.pop(action_id, (waiter, None))
        return result or {"ok": False, "error": "empty_device_action_result"}

    def snapshot(self) -> dict:
        with self._lock:
            clients = len(self._clients)
            client_info = dict(self._client_info.get(self._audio_client, {}))
            call_clients = sum(
                "call_phone" in info.get("capabilities", [])
                for info in self._client_info.values())
        return {
            "enabled": True,
            "host": self.host,
            "port": self.port,
            "clients": clients,
            "url_path": "/audio",
            "sent_frames": self.sent_frames,
            "sent_samples": self.sent_samples,
            "dropped_frames": self.dropped_frames,
            "client_policy": "single_audio_sink_with_device_action_clients",
            "active_client": client_info,
            "device_action_clients": call_clients,
            "chunk_samples": 480,
            "render_ack_count": self.render_ack_count,
            "last_render_delay_ms": self.last_render_delay_ms,
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
        self._echo_reference_active = False
        self._playback_id: str | None = None
        self._browser_render_syncs = 0
        self._last_browser_render_delay_ms: float | None = None
        setter = getattr(self.hub, "set_render_start_callback", None)
        if callable(setter):
            setter(self._on_browser_render_started)
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
        self.stop_playback(silent=True)
        started = time.perf_counter()
        playback_id = uuid.uuid4().hex[:16]
        duration = float(len(x) / self.sr) if self.sr else 0.0
        with self._lock:
            self._playing = True
            self._play_until = started + duration
            self._playback_id = playback_id
            self.output_events.append({
                "started_abs": started,
                "samples": int(x.size),
                "duration_s": round(duration, 3),
                "sink": "browser_websocket",
                "stopped_abs": None,
            })
        # Browser output is acoustic playback even though it does not use a
        # local PortAudio output stream. Register it with the capture transport
        # so its speaker reference is removed before VAD/ASR. If no browser is
        # connected, do not suppress real microphone speech for undelivered
        # audio.
        has_client = getattr(self.hub, "has_client", None)
        connected = bool(has_client()) if callable(has_client) else bool(
            self.hub.snapshot().get("clients", 0)
        )
        if connected:
            self.capture.register_external_playback(x)
            self._echo_reference_active = True
        # Register before publishing so the first browser chunk is covered by
        # the echo guard. The hub sends small frames and keeps one client only,
        # preventing stale tabs and keepalive timeouts from creating an
        # untracked acoustic source.
        try:
            self.hub.publish(x, playback_id=playback_id)
        except TypeError:
            # Keep small test doubles and older external hubs source-compatible.
            self.hub.publish(x)
        if blocking:
            while not self.playback_done():
                time.sleep(0.02)

    def _on_browser_render_started(self, event: dict) -> None:
        """Retimestamp the echo reference to the browser's scheduled render.

        ``delay_ms`` is measured in the browser from receipt of the first PCM
        chunk to the Web Audio ``AudioBufferSourceNode.start`` time.  The
        acknowledgement reaches the Pi after that scheduling decision, so the
        Pi uses the acknowledgement time plus the reported delay as the
        reference origin.  The normal acoustic delay search then covers the
        remaining network and room variation.
        """
        playback_id = str(event.get("playback_id", ""))
        delay_ms = float(event.get("delay_ms", 0.0) or 0.0)
        with self._lock:
            if playback_id and playback_id != self._playback_id:
                return
            if not self._echo_reference_active:
                return
        self.capture.mark_external_playback_started(
            time.perf_counter() + max(0.0, delay_ms) / 1000.0)
        self._browser_render_syncs += 1
        self._last_browser_render_delay_ms = round(delay_ms, 1)

    def playback_done(self) -> bool:
        clear_reference = False
        with self._lock:
            if self._playing and time.perf_counter() >= self._play_until:
                self._playing = False
                clear_reference = self._echo_reference_active
                self._echo_reference_active = False
            done = not self._playing
        if clear_reference:
            self.capture.clear_external_playback()
        return done

    def stop_playback(self, silent: bool = False):
        clear_reference = False
        with self._lock:
            clear_reference = self._echo_reference_active
            self._echo_reference_active = False
            self._playing = False
            now = time.perf_counter()
            self._play_until = now
            if self.output_events:
                self.output_events[-1]["stopped_abs"] = now
        # Always clear browser-side queued AudioBufferSourceNodes, even when
        # the local timing state already says playback ended. This handles a
        # tab that buffered chunks while the agent was doing inference.
        self.hub.clear()
        if clear_reference:
            self.capture.clear_external_playback()

    def close(self):
        self.stop_playback(silent=True)
        self.capture.close()
