"""Live status endpoint for the ME3 full-duplex voice loop.

Exposes a small localhost-only HTTP surface (default 127.0.0.1:8126) so
browser pages can render what the running ``LocalVoiceLoop`` is doing in
real time:

* ``GET /``            -> ``web/assistant.html`` (ME2-style live voice UI)
* ``GET /roadmap``     -> ``web/roadmap.html`` (remaining-work dashboard)
* ``GET /api/state``   -> JSON snapshot of the running loop
* ``GET /api/health``  -> small liveness probe

The snapshot is read live on every request, and the
release/acceptance/Pi block is re-read from ``deployment/`` whenever
those files change, so the pages always show the current canonical
release. The loop reports transcripts, replies, tool progress, VAD
level and model-load metrics through the thread-safe
:class:`LoopStatus` object below. Event payloads keep the same
sanitization policy as the rest of the agent: only tool names and success
flags are recorded -- never tool arguments or results.
"""
from __future__ import annotations

import json
import ipaddress
import logging
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

log = logging.getLogger("me3.agent_status")

_PAGE_FILES = {"/": "assistant.html", "/index.html": "assistant.html",
               "/roadmap": "roadmap.html", "/console": "assistant.html"}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
        timespec="seconds")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _latency_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"last": round(float(values[-1]), 1),
            "median": round(float(statistics.median(values)), 1),
            "p95": round(_percentile([float(v) for v in values], 95), 1)}


class LoopStatus:
    """Thread-safe live status shared by the voice loop and the HTTP server."""

    def __init__(self, release_info: dict | None = None,
                 brain_name: str = "", variant: str = ""):
        self._lock = threading.Lock()
        self.release_info = release_info or {}
        self.brain_name = brain_name
        self.variant = variant
        self.loop_running = False
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.state_reason = "loop not started"
        self.vad_probability = 0.0
        self.rms = 0.0
        self.last_transcript = ""
        self.last_transcript_at: float | None = None
        self.last_reply = ""
        self.last_reply_kind = ""
        self.last_reply_at: float | None = None
        self.last_error = ""
        self.last_error_stage = ""
        self.last_error_at: float | None = None
        self.last_barge_in_at: float | None = None
        self.models: dict = {}
        self.tool_events: deque = deque(maxlen=8)
        self._tool_active: str | None = None
        self.progress_events: deque = deque(maxlen=24)
        self.current_progress: dict = {}
        self.background: dict = {"long_tasks": [], "processes": []}

    # ---- reporter hooks (called from the loop; keep them cheap) ----------
    def report_started(self, reason: str = "audio capture started") -> None:
        with self._lock:
            self.loop_running = True
            self.started_at = time.time()
            self.stopped_at = None
            self.state_reason = reason

    def report_stopped(self) -> None:
        with self._lock:
            self.loop_running = False
            self.stopped_at = time.time()
            self.state_reason = "loop stopped"

    def report_state_reason(self, reason: str) -> None:
        with self._lock:
            self.state_reason = str(reason)

    def report_vad(self, probability: float, rms: float) -> None:
        with self._lock:
            self.vad_probability = float(probability)
            self.rms = float(rms)

    def report_transcript(self, text: str) -> None:
        with self._lock:
            self.last_transcript = str(text)
            self.last_transcript_at = time.time()

    def report_reply(self, text: str, kind: str = "reply") -> None:
        with self._lock:
            self.last_reply = str(text)
            self.last_reply_kind = str(kind)
            self.last_reply_at = time.time()

    def report_error(self, stage: str, message: str) -> None:
        """Publish a bounded operator-facing error, never raw payloads."""
        clean = " ".join(str(message).split())[:300]
        with self._lock:
            self.last_error = clean
            self.last_error_stage = str(stage)[:40]
            self.last_error_at = time.time()

    def clear_error(self) -> None:
        with self._lock:
            self.last_error = ""
            self.last_error_stage = ""
            self.last_error_at = None

    def report_tool(self, phase: str, name: str, ok: bool | None = None) -> None:
        with self._lock:
            self.tool_events.append({
                "phase": str(phase), "name": str(name),
                "ok": None if ok is None else bool(ok), "at": time.time(),
            })
            if phase == "tool_started":
                self._tool_active = str(name)
            elif phase == "tool_finished" and self._tool_active == str(name):
                self._tool_active = None

    def report_barge_in(self) -> None:
        with self._lock:
            self.last_barge_in_at = time.time()

    def report_models(self, **kwargs) -> None:
        with self._lock:
            self.models.update({k: v for k, v in kwargs.items() if v is not None})

    def report_progress(self, stage: str, detail: str) -> None:
        """Record a safe phase summary, never hidden model reasoning."""
        now = time.time()
        event = {
            "stage": str(stage)[:80].replace("\n", " "),
            "detail": str(detail)[:240].replace("\n", " "),
            "at": now,
        }
        with self._lock:
            self.current_progress = dict(event)
            self.progress_events.append(event)
            self.state_reason = event["detail"]

    def report_background(self, *, long_tasks: list[dict] | None = None,
                          processes: list[dict] | None = None) -> None:
        """Publish bounded long-task and subprocess telemetry for the UI."""
        with self._lock:
            self.background = {
                "long_tasks": list(long_tasks or [])[:12],
                "processes": list(processes or [])[:12],
            }

    def set_release_info(self, release_info: dict) -> None:
        with self._lock:
            self.release_info = release_info

    # ---- snapshot ---------------------------------------------------------
    def snapshot(self, state: str, stats) -> dict:
        with self._lock:
            now = time.time()
            active = self._tool_active
            recent = [{**dict(e), "at": _iso(e["at"])}
                      for e in self.tool_events]
            progress = [{**dict(e), "at": _iso(e["at"])}
                        for e in self.progress_events]
            current_progress = dict(self.current_progress)
            if current_progress.get("at") is not None:
                current_progress["at"] = _iso(current_progress["at"])
            payload = {
                "ok": True,
                "loop": self.loop_running,
                "state": str(state),
                "state_reason": self.state_reason,
                "variant": self.variant,
                "brain": self.brain_name,
                "uptime_s": round(now - self.started_at, 1)
                if (self.loop_running and self.started_at) else None,
                "started_at": _iso(self.started_at),
                "stopped_at": _iso(self.stopped_at),
                "updated_at": _iso(now),
                "vad": {"probability": round(self.vad_probability, 3),
                        "rms": round(self.rms, 3)},
                "last_transcript": {"text": self.last_transcript,
                                    "at": _iso(self.last_transcript_at)},
                "last_reply": {"text": self.last_reply,
                               "kind": self.last_reply_kind,
                               "at": _iso(self.last_reply_at)},
                "last_error": {"stage": self.last_error_stage,
                                "message": self.last_error,
                                "at": _iso(self.last_error_at)},
                "last_barge_in_at": _iso(self.last_barge_in_at),
                "stats": {
                    "n_turns": int(getattr(stats, "n_turns", 0) or 0),
                    "n_barge_in": int(getattr(stats, "n_barge_in", 0) or 0),
                    "n_control": int(getattr(stats, "n_control", 0) or 0),
                    "n_errors": int(getattr(stats, "n_errors", 0) or 0),
                    "latency": {
                        "asr_ms": _latency_summary(list(getattr(stats, "asr_ms", []) or [])),
                        "llm_ms": _latency_summary(list(getattr(stats, "llm_ms", []) or [])),
                        "tts_ms": _latency_summary(list(getattr(stats, "tts_ms", []) or [])),
                        "e2e_ms": _latency_summary(list(getattr(stats, "e2e_ms", []) or [])),
                    },
                },
                "tools": {"active": active, "recent": recent},
                "progress": {"current": current_progress, "recent": progress},
                "background": {
                    "long_tasks": [dict(x) for x in self.background.get("long_tasks", [])],
                    "processes": [dict(x) for x in self.background.get("processes", [])],
                },
                "models": dict(self.models),
                "release": dict(self.release_info),
            }
        return payload


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - records are optional
        return {}


def load_release_info(project: Path) -> dict:
    """Bounded read of the deployment records (no secrets live there)."""
    dep = Path(project) / "deployment"
    release = _read_json(dep / "latest_release.json")
    acceptance = _read_json(dep / "workstation_acceptance.json")
    pi = _read_json(dep / "last_pi_deployment.json")
    checks = acceptance.get("checks") or []
    passed = sum(1 for c in checks if c.get("status") == "pass")
    return {
        "release": {k: release.get(k) for k in
                    ("release_id", "status", "created_utc", "file_count",
                     "zip_sha256", "note") if release.get(k) is not None},
        "acceptance": {
            "acceptance_id": acceptance.get("acceptance_id"),
            "status": acceptance.get("status"),
            "created_utc": acceptance.get("created_utc"),
            "checks_passed": passed,
            "checks_total": len(checks),
            "failed_checks": list(acceptance.get("failed_checks") or []),
        },
        "pi": {k: pi.get(k) for k in ("status", "at", "release_id", "reason")
               if pi.get(k) is not None},
    }


class ReleaseInfoRefresher:
    """Keeps the status release block in step with deployment records.

    The acceptance gate rewrites ``deployment/`` on every run, so a
    startup-time snapshot would go stale within the same session.
    Re-reading is gated on file mtimes, so per-request polling only
    stats three tiny JSON files.
    """

    _FILES = ("latest_release.json", "workstation_acceptance.json",
              "last_pi_deployment.json")

    def __init__(self, project: Path):
        self.project = Path(project)
        self.dep = self.project / "deployment"
        self._sig: tuple[int, ...] | None = None

    def _signature(self) -> tuple[int, ...] | None:
        try:
            return tuple((self.dep / name).stat().st_mtime_ns
                         if (self.dep / name).is_file() else -1
                         for name in self._FILES)
        except OSError:
            return None

    def refresh(self, status: LoopStatus) -> None:
        sig = self._signature()
        if sig is None or sig == self._sig:
            return
        self._sig = sig
        status.set_release_info(load_release_info(self.project))


class _StatusHandler(BaseHTTPRequestHandler):
    server_version = "ME3Status/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: D102 - keep the console clean
        # OAuth callbacks contain single-use authorization codes and CSRF
        # state in the request target. Keep only the URL path in access logs.
        safe_args = list(args)
        if safe_args and isinstance(safe_args[0], str):
            parts = safe_args[0].split()
            if len(parts) >= 2:
                path = urlsplit(parts[1]).path
                safe_args[0] = " ".join([parts[0], path, *parts[2:]])
        log.debug("status: " + fmt, *safe_args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _is_loopback_request(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except (ValueError, IndexError, TypeError):
            return False

    def _oauth_result_page(self, provider: str, result: dict) -> None:
        if result.get("ok"):
            title, message, code = f"{provider} connected", f"{provider} is connected to ME3. Return to the assistant tab.", 200
        else:
            title, message, code = f"{provider} connection not completed", f"Return to ME3 and check the integration status before retrying.", 400
        page = ("<!doctype html><meta charset='utf-8'><meta name='referrer' content='no-referrer'>"
                "<title>" + title + "</title><main style='font:16px system-ui;max-width:620px;margin:12vh auto;padding:24px'>"
                "<h1>" + title + "</h1><p>" + message + "</p></main>")
        self._send(code, page.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        status = self.server.status  # type: ignore[attr-defined]
        if path == "/api/schedules":
            callback = getattr(self.server, "schedules_get", None)
            payload = callback() if callback is not None else {"ok": True, "items": []}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/spotify/status":
            callback = getattr(self.server, "spotify_status", None)
            payload = callback() if callback is not None else {
                "configured": False, "missing": ["Spotify integration is not enabled"]}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/google/status":
            callback = getattr(self.server, "google_status", None)
            payload = callback() if callback is not None else {
                "configured": False, "missing_env": ["Google Calendar is not enabled"]}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/telegram/status":
            callback = getattr(self.server, "telegram_status", None)
            payload = callback() if callback is not None else {
                "configured": False, "bot_configured": False, "target_mapped": False}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/google/connect":
            callback = getattr(self.server, "google_connect", None)
            result = callback() if callback is not None else {
                "ok": False, "error": "google_calendar_integration_not_enabled"}
            if not isinstance(result, dict) or not result.get("ok") or not result.get("url"):
                self._send(400, json.dumps(result or {"ok": False}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            self.send_response(302)
            self.send_header("Location", result["url"])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/api/spotify/connect":
            callback = getattr(self.server, "spotify_connect", None)
            result = callback() if callback is not None else {
                "ok": False, "error": "spotify_integration_not_enabled"}
            if not isinstance(result, dict) or not result.get("ok") or not result.get("url"):
                self._send(400, json.dumps(result or {"ok": False}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            self.send_response(302)
            self.send_header("Location", result["url"])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/api/spotify/callback":
            query = parse_qs(urlsplit(self.path).query)
            if query.get("error"):
                result = {"ok": False, "error": "spotify_authorization_denied"}
            else:
                callback = getattr(self.server, "spotify_callback", None)
                result = callback(
                    (query.get("code") or [""])[0],
                    (query.get("state") or [""])[0],
                ) if callback is not None else {
                    "ok": False, "error": "spotify_integration_not_enabled"}
            self._oauth_result_page("Spotify", result if isinstance(result, dict) else {})
            return
        if path == "/api/google/callback":
            query = parse_qs(urlsplit(self.path).query)
            if query.get("error"):
                result = {"ok": False, "error": "google_authorization_denied"}
            else:
                callback = getattr(self.server, "google_callback", None)
                result = callback(
                    (query.get("code") or [""])[0],
                    (query.get("state") or [""])[0],
                ) if callback is not None else {
                    "ok": False, "error": "google_calendar_integration_not_enabled"}
            self._oauth_result_page("Google Calendar", result if isinstance(result, dict) else {})
            return
        if path == "/api/demo/light":
            callback = getattr(self.server, "demo_light_get", None)
            if callback is None:
                self._send(404, b"demo light is not enabled", "text/plain; charset=utf-8")
                return
            try:
                payload = callback()
                self._send(200, json.dumps(payload).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - keep status UI reachable
                self._send(500, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/notifications/pending":
            callback = getattr(self.server, "notifications_pending", None)
            payload = callback() if callback is not None else {"ok": True, "notifications": []}
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/audio/devices":
            callback = getattr(self.server, "audio_info", None)
            payload = callback() if callback is not None else {
                "ok": True, "available": False,
                "message": "audio selection is only available in the Pi browser server",
            }
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/duplex-mode":
            callback = getattr(self.server, "duplex_mode_get", None)
            if callback is None:
                self._send(404, b"duplex mode is not enabled",
                           "text/plain; charset=utf-8")
                return
            try:
                payload = callback()
                self._send(200, json.dumps(payload).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - keep status UI reachable
                self._send(500, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/state":
            refresher = getattr(self.server, "release_refresher", None)
            if refresher is not None:
                refresher.refresh(status)
            snap = status.snapshot(
                self.server.get_state(),  # type: ignore[attr-defined]
                self.server.get_stats(),  # type: ignore[attr-defined]
            )
            self._send(200, json.dumps(snap).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/health":
            body = json.dumps({"ok": True,
                               "loop": bool(status.loop_running)}).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        page = _PAGE_FILES.get(path)
        if page:
            fpath = self.server.html_dir / page  # type: ignore[attr-defined]
            try:
                html = fpath.read_text(encoding="utf-8")
            except OSError:
                self._send(404, f"page {page} not found".encode(),
                           "text/plain; charset=utf-8")
                return
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/api/schedules", "/api/schedules/cancel"}:
            callback = getattr(self.server, "schedule_cancel" if path.endswith("/cancel")
                               else "schedule_create", None)
            if callback is None:
                self._send(404, b"clock actions are not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                argument = str(payload.get("id") or "") if path.endswith("/cancel") else payload
                result = callback(argument)
                code = 200 if result.get("ok") else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/integrations/google-config":
            if not self._is_loopback_request():
                self._send(403, b"use the local SSH tunnel to save credentials", "text/plain; charset=utf-8")
                return
            callback = getattr(self.server, "google_client_set", None)
            if callback is None:
                self._send(404, b"Google Calendar setup is not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = callback(str(payload.get("client_id") or ""),
                                  str(payload.get("client_secret") or ""))
                code = 200 if result.get("ok") else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/integrations/telegram-token":
            if not self._is_loopback_request():
                self._send(403, b"use the local SSH tunnel to save credentials", "text/plain; charset=utf-8")
                return
            callback = getattr(self.server, "telegram_token_set", None)
            if callback is None:
                self._send(404, b"Telegram setup is not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = callback(str(payload.get("bot_token") or ""))
                code = 200 if result.get("ok") else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/telegram/connect-target":
            if not self._is_loopback_request():
                self._send(403, b"use the local SSH tunnel for Telegram setup", "text/plain; charset=utf-8")
                return
            callback = getattr(self.server, "telegram_connect_target", None)
            if callback is None:
                self._send(404, b"Telegram setup is not enabled", "text/plain; charset=utf-8")
                return
            try:
                result = callback()
                code = 200 if result.get("ok") else 409
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/spotify/client-id":
            callback = getattr(self.server, "spotify_client_id_set", None)
            if callback is None:
                self._send(404, b"Spotify setup is not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = callback(str(payload.get("client_id") or ""))
                code = 200 if result.get("ok") else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/notifications/ack":
            callback = getattr(self.server, "notification_ack", None)
            if callback is None:
                self._send(404, b"notifications are not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = callback(str(payload.get("id") or ""))
                code = 200 if result.get("ok") else 409
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": type(exc).__name__}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        if path == "/api/demo/light":
            callback = getattr(self.server, "demo_light_control", None)
            if callback is None:
                self._send(404, b"demo light is not enabled", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("selection must be a JSON object")
                result = callback(payload)
                code = 200 if result.get("ok", True) else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - UI receives bounded error
                self._send(400, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/restart":
            callback = getattr(self.server, "restart_agent", None)
            if callback is None:
                self._send(403, b"agent restart is not enabled", "text/plain; charset=utf-8")
                return
            try:
                result = callback()
                self._send(200, json.dumps({"ok": True, **(result or {})}).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/restart-service":
            callback = getattr(self.server, "restart_service", None)
            if callback is None:
                self._send(403, b"service restart is not enabled", "text/plain; charset=utf-8")
                return
            try:
                result = callback()
                self._send(200, json.dumps({"ok": True, **(result or {})}).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/api/duplex-mode":
            callback = getattr(self.server, "duplex_mode_set", None)
            if callback is None:
                self._send(403, b"duplex mode is not enabled",
                           "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10000:
                    raise ValueError("invalid JSON body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("mode must be a JSON object")
                result = callback(payload)
                code = 200 if result.get("ok", True) else 400
                self._send(code, json.dumps(result).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - bounded UI error
                self._send(400, json.dumps({
                    "ok": False, "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path != "/api/audio/select":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        callback = getattr(self.server, "audio_select", None)
        if callback is None:
            self._send(403, b"audio selection is not enabled", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10000:
                raise ValueError("invalid JSON body")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = callback(payload)
            self._send(200, json.dumps(result).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001
            self._send(400, json.dumps({
                "ok": False, "error": type(exc).__name__,
                "message": str(exc)[:300],
            }).encode("utf-8"), "application/json; charset=utf-8")


class StatusServer:
    """Tiny 127.0.0.1 status/assistant UI server (daemon thread)."""

    def __init__(self, status: LoopStatus, *, html_dir: Path,
                 get_state, get_stats, port: int = 8126,
                 project: Path | None = None, host: str = "127.0.0.1",
                 audio_info=None, audio_select=None, restart_agent=None,
                 restart_service=None, demo_light_get=None,
                 demo_light_control=None, duplex_mode_get=None,
                 duplex_mode_set=None, notifications_pending=None,
                 notification_ack=None, spotify_status=None,
                 spotify_connect=None, spotify_callback=None,
                 spotify_client_id_set=None, google_status=None,
                 google_connect=None, google_callback=None,
                 google_client_set=None, telegram_status=None,
                 telegram_token_set=None, telegram_connect_target=None,
                 schedules_get=None, schedule_create=None,
                 schedule_cancel=None):
        self.status = status
        self.html_dir = Path(html_dir)
        self.get_state = get_state
        self.get_stats = get_stats
        self.port = int(port)
        self.host = str(host)
        self.project = project
        self.audio_info = audio_info
        self.audio_select = audio_select
        self.restart_agent = restart_agent
        self.restart_service = restart_service
        self.demo_light_get = demo_light_get
        self.demo_light_control = demo_light_control
        self.duplex_mode_get = duplex_mode_get
        self.duplex_mode_set = duplex_mode_set
        self.notifications_pending = notifications_pending
        self.notification_ack = notification_ack
        self.spotify_status = spotify_status
        self.spotify_connect = spotify_connect
        self.spotify_callback = spotify_callback
        self.spotify_client_id_set = spotify_client_id_set
        self.google_status = google_status
        self.google_connect = google_connect
        self.google_callback = google_callback
        self.google_client_set = google_client_set
        self.telegram_status = telegram_status
        self.telegram_token_set = telegram_token_set
        self.telegram_connect_target = telegram_connect_target
        self.schedules_get = schedules_get
        self.schedule_create = schedule_create
        self.schedule_cancel = schedule_cancel
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._httpd is not None:
            return True
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port),
                                              _StatusHandler)
        except OSError as exc:
            log.warning("status UI: cannot bind %s:%d (%s)",
                        self.host, self.port, exc)
            return False
        self._httpd.status = self.status
        self._httpd.html_dir = self.html_dir
        self._httpd.get_state = self.get_state
        self._httpd.get_stats = self.get_stats
        self._httpd.audio_info = self.audio_info
        self._httpd.audio_select = self.audio_select
        self._httpd.restart_agent = self.restart_agent
        self._httpd.restart_service = self.restart_service
        self._httpd.demo_light_get = self.demo_light_get
        self._httpd.demo_light_control = self.demo_light_control
        self._httpd.duplex_mode_get = self.duplex_mode_get
        self._httpd.duplex_mode_set = self.duplex_mode_set
        self._httpd.notifications_pending = self.notifications_pending
        self._httpd.notification_ack = self.notification_ack
        self._httpd.spotify_status = self.spotify_status
        self._httpd.spotify_connect = self.spotify_connect
        self._httpd.spotify_callback = self.spotify_callback
        self._httpd.spotify_client_id_set = self.spotify_client_id_set
        self._httpd.google_status = self.google_status
        self._httpd.google_connect = self.google_connect
        self._httpd.google_callback = self.google_callback
        self._httpd.google_client_set = self.google_client_set
        self._httpd.telegram_status = self.telegram_status
        self._httpd.telegram_token_set = self.telegram_token_set
        self._httpd.telegram_connect_target = self.telegram_connect_target
        self._httpd.schedules_get = self.schedules_get
        self._httpd.schedule_create = self.schedule_create
        self._httpd.schedule_cancel = self.schedule_cancel
        self._httpd.release_refresher = (
            ReleaseInfoRefresher(self.project) if self.project is not None
            else None)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="me3-status-ui",
            daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            finally:
                self._httpd = None
                self._thread = None
