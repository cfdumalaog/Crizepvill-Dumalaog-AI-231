"""Local ME3 browser console for safe workstation testing.

This server is intentionally localhost-only by default.  It exposes the same
provider-neutral ToolBrain used by the voice loop through a small JSON API and serves the
tracked UI from ``web/ui.html``.  External communication tools are blocked in
this console even if credentials are present; the UI is for local evaluation,
not live phone, SMS, or email operations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT / "web"
LOG = logging.getLogger("me3.local_ui")
EXTERNAL_TOOLS = {"send_sms", "make_call", "send_email"}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="me3-ui-job")
MAX_RETAINED_JOBS = 48

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from me3_bootstrap import configure_me3_path  # noqa: E402
configure_me3_path(PROJECT)

from me3.agent import StubBrain  # noqa: E402
from me3.brain import (ToolBrain, build_system_prompt, build_tool_brain,
                       resolve_llm_config)  # noqa: E402
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


def _make_brain(mode: str, events: list[dict], on_progress=None):
    cfg = load_config()
    selected = mode
    if selected == "remote":
        selected = "qwen"
    if selected == "auto":
        provider, _provider_cfg = resolve_llm_config(cfg, "auto")
        selected = provider or "stub"
    if selected == "stub":
        return StubBrain(), "stub"
    if selected not in {"qwen", "luna"}:
        raise ValueError("mode must be auto, qwen, luna, or stub")
    system_prompt = build_system_prompt(cfg["agent"]["system_prompt"])
    def on_tool(kind, name, payload):
        event = {
            "kind": kind,
            "name": name,
            "ok": (bool(payload.get("ok")) if isinstance(payload, dict)
                   and kind == "finish" else None),
        }
        events.append(event)
        if on_progress:
            if kind == "start":
                on_progress("tool", f"Running tool: {name}")
            else:
                result = "ok" if event["ok"] else "error"
                on_progress("tool", f"Finished tool: {name} ({result})")

    brain = build_tool_brain(
        cfg, selected, system_prompt,
        on_tool=on_tool,
        dispatch_fn=local_dispatch,
        on_progress=on_progress,
    )
    return brain, selected


def _append_thinking(job_id: str, stage: str, task: str) -> None:
    """Record a bounded, safe task summary for the browser progress panel."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["stage"] = stage
        job["current_task"] = task
        elapsed_ms = round((time.perf_counter() - job["started_at"]) * 1000)
        entries = job["thinking"]
        if entries and entries[-1]["task"] == task:
            entries[-1]["elapsed_ms"] = elapsed_ms
            return
        entries.append({"stage": stage, "task": task, "elapsed_ms": elapsed_ms})
        del entries[:-40]


def _job_snapshot(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return deepcopy(job) if job else None


def _update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(changes)


def _prune_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= MAX_RETAINED_JOBS:
            return
        finished = sorted(
            (job for job in JOBS.values()
             if job["status"] in {"completed", "error"}),
            key=lambda job: job["started_at"],
        )
        for job in finished[:max(0, len(JOBS) - MAX_RETAINED_JOBS)]:
            JOBS.pop(job["job_id"], None)


def _run_chat_job(job_id: str, message: str, mode: str, history: list[dict]) -> None:
    events: list[dict] = []

    def progress(stage: str, task: str) -> None:
        _append_thinking(job_id, stage, task)

    _update_job(job_id, status="running")
    progress("planning", "Preparing the selected model and bounded tools")
    try:
        brain, actual_mode = _make_brain(mode, events, on_progress=progress)
        _update_job(job_id, mode=actual_mode)
        progress("thinking", f"Thinking with {actual_mode}; deciding what to do")
        reply = brain.chat(history, message)
        metrics = _safe_metrics(brain)
        metrics["elapsed_ms"] = round(
            (time.perf_counter() - JOBS[job_id]["started_at"]) * 1000)
        progress("complete", "Task complete; response is ready")
        _update_job(
            job_id,
            status="completed",
            stage="complete",
            current_task="Task complete; response is ready",
            reply=reply,
            events=events,
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.exception("local UI background request failed")
        progress("error", f"Task failed: {type(exc).__name__}")
        _update_job(
            job_id,
            status="error",
            stage="error",
            current_task=f"Task failed: {type(exc).__name__}",
            error=str(exc)[:500],
            events=events,
        )
    finally:
        _prune_jobs()


def _new_chat_job(message: str, mode: str, history: list[dict]) -> dict:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "ok": True,
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "current_task": "Queued task",
            "thinking": [],
            "events": [],
            "mode": mode,
            "reply": None,
            "metrics": None,
            "error": None,
            "started_at": time.perf_counter(),
        }
    JOB_EXECUTOR.submit(_run_chat_job, job_id, message, mode, history)
    return {"ok": True, "job_id": job_id, "status": "queued"}


def _parse_chat_payload(handler: "Handler") -> tuple[str, str, list[dict]]:
    payload = handler._body()
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message is required")
    if len(message) > 12000:
        raise ValueError("message is limited to 12000 characters")
    return message.strip(), str(payload.get("mode", "auto")), _safe_history(
        payload.get("history"))


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
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.path = "/ui.html"
        elif path == "/api/chat/status":
            job_id = (parse_qs(parsed.query).get("job_id") or [""])[0]
            if not job_id:
                self._json(400, {"ok": False, "error": "job_id_required"})
                return
            snapshot = _job_snapshot(job_id)
            if snapshot is None:
                self._json(404, {"ok": False, "error": "job_not_found"})
                return
            snapshot.pop("started_at", None)
            self._json(200, snapshot)
            return
        elif path == "/api/health":
            cfg = load_config()
            audio_cfg = cfg.get("asr", {})
            tts_cfg = cfg.get("tts", {})
            self._json(200, {
                "ok": True,
                "service": "me3-local-ui",
                "audio_variant": cfg.get("runtime_variant", "current"),
                "audio_models": {
                    "asr": audio_cfg.get("model", ""),
                    "asr_engine": audio_cfg.get("engine", ""),
                    "tts": tts_cfg.get("model", tts_cfg.get("voice", "")),
                    "tts_engine": tts_cfg.get("engine", ""),
                    "asr_endpoint_configured": bool(os.environ.get(
                        audio_cfg.get("base_url_env", ""), "").strip()),
                    "tts_endpoint_configured": bool(os.environ.get(
                        tts_cfg.get("base_url_env", ""), "").strip()),
                    "shared_key_configured": bool(os.environ.get(
                        audio_cfg.get("api_key_env", ""), "").strip()),
                },
                "qwen": {
                    "configured": bool(os.environ.get(
                        cfg["llm"]["remote"].get("api_key_env", "VLLM_API_KEY"), "").strip()),
                    "model": cfg["llm"]["remote"].get("model", ""),
                    "base_url": cfg["llm"]["remote"].get("base_url", ""),
                },
                "luna": {
                    "configured": bool(os.environ.get(
                        cfg["llm"].get("openai", {}).get("api_key_env", "OPENAI_API_KEY"), "").strip()),
                    "model": cfg["llm"].get("openai", {}).get("model", ""),
                    "reasoning_effort": "disabled",
                    "stream": bool(cfg["llm"].get("openai", {}).get("stream", False)),
                    "base_url": cfg["llm"].get("openai", {}).get("base_url", ""),
                },
                "default_mode": getattr(self.server, "default_mode", "auto"),
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
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/chat/start"}:
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            message, mode, history = _parse_chat_payload(self)
            if path == "/api/chat/start":
                self._json(202, _new_chat_job(message, mode, history))
                return

            events: list[dict] = []
            progress: list[dict] = []

            def on_progress(stage: str, task: str) -> None:
                progress.append({"stage": stage, "task": task})

            brain, actual_mode = _make_brain(
                mode, events, on_progress=on_progress)
            started = time.perf_counter()
            reply = brain.chat(history, message.strip())
            metrics = _safe_metrics(brain)
            metrics["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            self._json(200, {
                "ok": True,
                "mode": actual_mode,
                "reply": reply,
                "events": events,
                "thinking": progress,
                "current_task": "Task complete; response is ready",
                "metrics": metrics,
                "external_channels": "disabled",
            })
        except Exception as exc:  # noqa: BLE001
            LOG.exception("local UI request failed")
            self._json(502, {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc)[:500],
                "hint": "Use GPT-5.6 Luna, Qwen, or Offline stub according to the configured provider and key.",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--default-mode", choices=("auto", "qwen", "luna", "stub"),
                        default="auto")
    parser.add_argument("--variant", choices=("current", "qwen_instruct"),
                        default="current")
    args = parser.parse_args()
    os.environ["ME3_VARIANT"] = args.variant
    if not WEB_ROOT.is_dir() or not (WEB_ROOT / "ui.html").is_file():
        raise SystemExit(f"web UI is missing under {WEB_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.default_mode = args.default_mode
    print(f"ME3 local UI: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Binding: {args.host}:{args.port}; variant={args.variant}; external channels disabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
