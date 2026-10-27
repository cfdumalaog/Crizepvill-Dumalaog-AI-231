"""Local ME3 browser console for safe workstation testing.

This server is intentionally localhost-only by default.  It exposes the same
Qwen ToolBrain used by the voice loop through a small JSON API and serves the
tracked UI from ``web/ui.html``.  External communication tools are blocked in
this console even if credentials are present; the UI is for local evaluation,
not live phone, SMS, or email operations.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT / "web"
LOG = logging.getLogger("me3.local_ui")
EXTERNAL_TOOLS = {"send_sms", "make_call", "send_email"}

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from me3.agent import StubBrain  # noqa: E402
from me3.brain import ToolBrain, build_system_prompt  # noqa: E402
from me3.channels import channel_health  # noqa: E402
from me3.config import load_config  # noqa: E402
from me3.tools import dispatch, tool_definitions  # noqa: E402


def local_dispatch(name: str, arguments: dict):
    """Dispatch project tools while denying provider side effects."""
    if name in EXTERNAL_TOOLS:
        return {
            "ok": False,
            "error": "external_channels_disabled_in_local_ui",
            "provider": "twilio" if name in {"send_sms", "make_call"} else "resend",
            "network_called": False,
        }
    return dispatch(name, arguments)


def _safe_history(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    history = []
    for item in value[-12:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            history.append({"role": item["role"], "content": content[:4000]})
    return history


def _safe_metrics(brain) -> dict:
    raw = getattr(brain, "metrics", {}) or {}
    return {
        "llm_calls": int(raw.get("n_llm_calls", 0)),
        "prompt_tokens": int(raw.get("prompt_tokens", 0)),
        "completion_tokens": int(raw.get("completion_tokens", 0)),
        "first_tool_ms": (round(raw["first_tool_ms"])
                           if raw.get("first_tool_ms") is not None else None),
        "tool_calls": raw.get("tool_calls", []),
        "elapsed_ms": round((time.perf_counter() - raw["t0"]) * 1000)
        if raw.get("t0") else None,
    }


def _make_brain(mode: str, events: list[dict]):
    cfg = load_config()
    selected = mode
    key_env = cfg["llm"]["remote"].get("api_key_env", "VLLM_API_KEY")
    has_key = bool(os.environ.get(key_env, "").strip())
    if selected == "auto":
        selected = "qwen" if has_key else "stub"
    if selected == "stub":
        return StubBrain(), "stub"
    if selected != "qwen":
        raise ValueError("mode must be auto, qwen, or stub")
    brain_cfg = dict(cfg["llm"]["remote"])
    brain_cfg["system_prompt"] = build_system_prompt(cfg["agent"]["system_prompt"])
    brain = ToolBrain(
        brain_cfg,
        brain_cfg["system_prompt"],
        on_tool=lambda kind, name, payload: events.append({
            "kind": kind,
            "name": name,
            "ok": (bool(payload.get("ok")) if isinstance(payload, dict)
                   and kind == "finish" else None),
        }),
        dispatch_fn=local_dispatch,
    )
    return brain, "qwen"


class Handler(SimpleHTTPRequestHandler):
    server_version = "ME3LocalUI/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("request body must be between 1 byte and 1 MB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self.path = "/ui.html"
        elif path == "/api/health":
            cfg = load_config()
            key_env = cfg["llm"]["remote"].get("api_key_env", "VLLM_API_KEY")
            self._json(200, {
                "ok": True,
                "service": "me3-local-ui",
                "qwen": {
                    "configured": bool(os.environ.get(key_env, "").strip()),
                    "model": cfg["llm"]["remote"].get("model", ""),
                    "base_url": cfg["llm"]["remote"].get("base_url", ""),
                },
                "tools": len(tool_definitions()),
                "channels": channel_health(),
                "local_policy": {
                    "external_channels": "disabled",
                    "physical_audio": "not opened by this UI",
                },
            })
            return
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/chat":
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            payload = self._body()
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("message is required")
            if len(message) > 12000:
                raise ValueError("message is limited to 12000 characters")
            mode = str(payload.get("mode", "auto"))
            history = _safe_history(payload.get("history"))
            events: list[dict] = []
            brain, actual_mode = _make_brain(mode, events)
            started = time.perf_counter()
            reply = brain.chat(history, message.strip())
            metrics = _safe_metrics(brain)
            metrics["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            self._json(200, {
                "ok": True,
                "mode": actual_mode,
                "reply": reply,
                "events": events,
                "metrics": metrics,
                "external_channels": "disabled",
            })
        except Exception as exc:  # noqa: BLE001
            LOG.exception("local UI request failed")
            self._json(502, {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc)[:500],
                "hint": "Use Offline stub if the Qwen endpoint is unavailable.",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args()
    if not WEB_ROOT.is_dir() or not (WEB_ROOT / "ui.html").is_file():
        raise SystemExit(f"web UI is missing under {WEB_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ME3 local UI: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Binding: {args.host}:{args.port}; external channels disabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
