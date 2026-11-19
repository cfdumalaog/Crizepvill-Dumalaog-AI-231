"""Private, runtime permission policy for ME3 tool execution.

The policy is deliberately separate from provider credentials and from the
model prompt.  It is enforced by :func:`me3.tools.dispatch`, so direct voice
routes and model-generated function calls have the same behavior.

The policy file contains only the selected mode and a timestamp.  It never
contains tool arguments, credentials, or conversation text.  Pending approval
arguments live in memory for one action only and expire automatically.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .secure_config import _path as _credentials_path
from .config import load_config

ASK_APPROVAL = "ask_approval"
APPROVE_FOR_ME = "approve_for_me"
FULL_ACCESS = "full_access"
MODES = (ASK_APPROVAL, APPROVE_FOR_ME, FULL_ACCESS)

_MODE_INFO = {
    ASK_APPROVAL: {
        "label": "Ask for approval",
        "description": "Ask before internet use, external files, shell commands, or outbound actions.",
    },
    APPROVE_FOR_ME: {
        "label": "Approve for me",
        "description": "Run ordinary requested work automatically; pause only for potentially unsafe or external side effects.",
    },
    FULL_ACCESS: {
        "label": "Full access",
        "description": "Do not pause for ME3 approvals. The process still runs as the Pi OS user and protected credential paths remain blocked.",
    },
}

_READ_ONLY = {
    "list_files", "read_file", "search_text", "task_status", "task_output",
    "system_info", "git_status", "web_servers", "music_state", "list_timers",
    "list_alarms", "list_reminders", "channel_health", "github_pages_health",
    "github_repositories", "github_repository_access", "get_time", "get_weather",
    "list_schedules",
}
_INTERNET = {
    "fetch_url", "web_search", "github_pages_health", "github_repositories",
    "github_repository_access", "deploy_github_pages", "play_music",
    "media_control", "music_state", "send_email", "send_telegram_message",
    "call_contact", "send_message", "schedule_meeting",
}
_OUTBOUND = {
    "deploy_github_pages", "send_email", "send_telegram_message", "call_contact",
    "send_message", "schedule_meeting", "play_music", "media_control",
}
_CONFIRM_ARGUMENT_TOOLS = {
    "send_message", "send_telegram_message", "send_email", "call_contact",
    "schedule_meeting",
}
_LOCK = threading.RLock()
_PENDING: dict | None = None
_PENDING_TTL_S = 10 * 60


def _path() -> Path:
    configured = os.getenv("ME3_PERMISSION_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _credentials_path().with_name("permission_policy.json")


def _default_mode() -> str:
    """Preserve the existing ME3 full-Pi-access demo behavior on first boot."""
    try:
        access = load_config().get("agent_access") or {}
        if bool(access.get("full_pi_user_access")):
            return FULL_ACCESS
    except Exception:  # noqa: BLE001 - policy must remain available during boot
        pass
    return ASK_APPROVAL


def _read_mode() -> tuple[str, bool]:
    try:
        payload = json.loads(_path().read_text(encoding="utf-8"))
        mode = str(payload.get("mode") or "").strip()
        if mode in MODES:
            return mode, True
    except (OSError, ValueError, TypeError):
        pass
    return _default_mode(), False


def status() -> dict:
    mode, persisted = _read_mode()
    with _LOCK:
        pending = _pending_locked() is not None
    info = _MODE_INFO[mode]
    return {
        "ok": True,
        "mode": mode,
        "label": info["label"],
        "description": info["description"],
        "persisted": persisted,
        "pending_approval": pending,
        "modes": [
            {"mode": key, "label": value["label"], "description": value["description"]}
            for key, value in _MODE_INFO.items()
        ],
    }


def mode() -> str:
    return _read_mode()[0]


def set_mode(value: str) -> dict:
    global _PENDING
    selected = str(value or "").strip().lower()
    if selected not in MODES:
        return {"ok": False, "error": "invalid_permission_mode", "allowed": list(MODES)}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "mode": selected,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.name == "nt":
        temporary.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.parent.chmod(0o700)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name != "nt":
        path.chmod(0o600)
    # A pending action was created under the old policy. Never allow a later
    # mode change to approve it accidentally under a different policy.
    with _LOCK:
        _PENDING = None
    return status()


def _is_agent_workspace(value: object) -> bool:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return True
    if raw.startswith("agent_workspace/") or raw == "agent_workspace":
        return True
    try:
        candidate = Path(raw).expanduser().resolve()
        root = Path(__file__).resolve().parents[1] / "agent_workspace"
        return candidate == root or root in candidate.parents
    except (OSError, RuntimeError, ValueError):
        return False


def _external_file_action(name: str, arguments: dict) -> bool:
    if name == "write_file":
        return not _is_agent_workspace(arguments.get("path"))
    if name in {"run_command", "start_background_task"}:
        return not _is_agent_workspace(arguments.get("cwd"))
    return False


def requires_approval(name: str, arguments: dict | None = None) -> bool:
    """Return whether the current mode must pause before this tool executes."""
    arguments = arguments if isinstance(arguments, dict) else {}
    selected = mode()
    if selected == FULL_ACCESS:
        return False
    if selected == APPROVE_FOR_ME:
        return name in _OUTBOUND or _external_file_action(name, arguments) or name in {
            "start_web_server", "stop_web_server",
        }
    # Strict mode: internet and outbound tools, all shell commands, service
    # controls, and edits outside the private agent workspace need approval.
    if name in _INTERNET or name in {"run_command", "start_background_task",
                                     "cancel_task", "start_web_server", "stop_web_server"}:
        return True
    return _external_file_action(name, arguments)


def confirm_arguments(name: str, arguments: dict, *, approved: bool = False) -> dict:
    """Add legacy tool confirmations only after policy approval is satisfied."""
    result = dict(arguments or {})
    if approved or mode() == FULL_ACCESS:
        if name in _CONFIRM_ARGUMENT_TOOLS:
            result["confirm"] = True
    return result


def _safe_target(name: str, arguments: dict) -> str:
    if name in {"fetch_url"}:
        try:
            from urllib.parse import urlsplit
            host = urlsplit(str(arguments.get("url") or "")).hostname or "the requested host"
            return host[:80]
        except Exception:  # noqa: BLE001
            return "the requested URL"
    if name in {"write_file", "read_file", "list_files", "run_command", "start_background_task"}:
        raw = str(arguments.get("path") or arguments.get("cwd") or "the requested path")
        return raw.replace("\r", " ").replace("\n", " ")[:100]
    if name in {"deploy_github_pages", "github_repositories", "github_repository_access"}:
        return str(arguments.get("repository") or arguments.get("repo") or "the requested repository")[:100]
    if name in _OUTBOUND:
        return "the requested external service"
    return "the requested action"


def action_summary(name: str, arguments: dict | None = None) -> str:
    return f"{name.replace('_', ' ')} ({_safe_target(name, arguments or {})})"


def _pending_locked() -> dict | None:
    global _PENDING
    if _PENDING and time.time() - float(_PENDING.get("created_at", 0)) > _PENDING_TTL_S:
        _PENDING = None
    return _PENDING


def pending() -> dict | None:
    with _LOCK:
        item = _pending_locked()
        if item is None:
            return None
        return {"id": item["id"], "tool": item["tool"], "summary": item["summary"]}


def create_pending(name: str, arguments: dict) -> dict:
    global _PENDING
    with _LOCK:
        existing = _pending_locked()
        if existing is not None:
            return {"id": existing["id"], "tool": existing["tool"],
                    "summary": existing["summary"], "existing": True}
        _PENDING = {
            "id": secrets.token_urlsafe(9),
            "tool": name,
            "arguments": dict(arguments or {}),
            "summary": action_summary(name, arguments),
            "created_at": time.time(),
        }
        return {"id": _PENDING["id"], "tool": name,
                "summary": _PENDING["summary"], "existing": False}


def take_pending() -> dict | None:
    global _PENDING
    with _LOCK:
        item = _pending_locked()
        _PENDING = None
        return dict(item) if item is not None else None


def clear_pending() -> dict | None:
    return take_pending()


def decision(text: str) -> str | None:
    value = " ".join(str(text or "").strip().lower().split())
    value = re.sub(r"[^a-z' ]+", "", value)
    value = " ".join(value.split())
    if value in {"yes", "y", "ok", "okay", "approve", "approved",
                 "approve it", "proceed", "go ahead", "yes go ahead",
                 "yes proceed", "continue", "do it"}:
        return "approve"
    if value in {"no", "n", "cancel", "cancel it", "deny", "denied",
                 "stop", "dont", "do not"}:
        return "deny"
    return None
