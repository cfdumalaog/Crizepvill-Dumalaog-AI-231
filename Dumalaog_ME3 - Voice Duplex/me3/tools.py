"""Voice-callable tools for the ME3 Qwen brain.

The registry is intentionally local and deterministic.  File operations are
confined to this ME3 checkout, command execution is tokenized (no shell), and
background jobs write only under ``agent_workspace/.tasks``.  The tool output
never exposes environment variables or credential contents.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse

from .config import load_config
from .task_runner import BackgroundTaskManager
from . import channels

PROJECT = Path(__file__).resolve().parent.parent
AGENT_ROOT = PROJECT / "agent_workspace"
TASKS = BackgroundTaskManager(PROJECT)

_MAX_READ = 50000
_MAX_WRITE = 400000
_SECRET_PARTS = {
    ".env", ".git", ".venv", "id_rsa", "credentials", "secrets",
    "vllm-api-key", "token", "password", "private_key",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt"}
_ALLOWED_COMMANDS = {
    "python", "python3", "py", "node", "npm", "npm.cmd", "npx", "npx.cmd",
    "git", "pytest", "ruff", "uvicorn", "flask", "bash", "sh", "ls", "dir",
    "rg", "find", "where", "pwd", "hostname", "uname", "date",
}
_DENIED_COMMAND_TEXT = re.compile(
    r"(?:\brm\s+-rf\b|\bdel\s+/[sqf]*\b|\brmdir\b|\bformat\b|\bshutdown\b|"
    r"\btaskkill\b|\bsudo\b|\bchmod\b|\bchown\b|\bssh\b|\bscp\b|"
    r"\bgit\s+(?:push|reset|clean|checkout\s+--)|\bnpm\s+publish\b|"
    r"(?:>|>>|\|\s*sh\b))",
    re.IGNORECASE,
)


def _relative(p: Path) -> str:
    return p.resolve().relative_to(PROJECT).as_posix()


def safe_path(path: str, *, allow_missing: bool = False, write: bool = False) -> Path:
    """Resolve a project-relative path and reject protected material."""
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path is required")
    candidate = Path(raw)
    p = candidate.resolve() if candidate.is_absolute() else (PROJECT / candidate).resolve()
    root = PROJECT.resolve()
    if not p.is_relative_to(root):
        raise ValueError("path must remain inside the ME3 project")
    rel_parts = [part.lower() for part in p.relative_to(root).parts]
    name = p.name.lower()
    if any(part in _SECRET_PARTS for part in rel_parts) or name in _SECRET_PARTS:
        raise ValueError("protected or credential-like path is not available to the voice agent")
    if p.suffix.lower() in _SECRET_SUFFIXES:
        raise ValueError("credential/certificate files are not available to the voice agent")
    if not allow_missing and not p.exists():
        raise FileNotFoundError(_relative(p))
    if write and any(part in {"models", "dist", ".tasks"} for part in rel_parts):
        # Model caches and generated release trees are managed by the release
        # workflow, not by arbitrary voice writes.
        raise ValueError("managed release/cache path is not writable by this tool")
    return p


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def list_files(path: str = "agent_workspace", pattern: str = "*",
               recursive: bool = True, max_items: int = 200) -> dict:
    root = safe_path(path)
    if not root.is_dir():
        return {"ok": False, "error": "not_a_directory", "path": str(path)}
    max_items = max(1, min(int(max_items), 500))
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    entries = []
    for p in sorted(iterator):
        if not p.is_file():
            continue
        try:
            safe_path(str(p))
        except (ValueError, FileNotFoundError):
            continue
        entries.append({"path": _relative(p), "bytes": p.stat().st_size})
        if len(entries) >= max_items:
            break
    return {"ok": True, "root": _relative(root), "files": entries,
            "truncated": len(entries) >= max_items}


def read_file(path: str, max_chars: int = 12000) -> dict:
    p = safe_path(path)
    if not p.is_file():
        return {"ok": False, "error": "not_a_file", "path": str(path)}
    max_chars = max(1, min(int(max_chars), _MAX_READ))
    data = p.read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "path": _relative(p), "content": data[:max_chars],
            "truncated": len(data) > max_chars}


def write_file(path: str, content: str, overwrite: bool = False) -> dict:
    p = safe_path(path, allow_missing=True, write=True)
    if p.exists() and not overwrite:
        return {"ok": False, "error": "file_exists_use_overwrite_true",
                "path": _relative(p)}
    if len(content.encode("utf-8")) > _MAX_WRITE:
        return {"ok": False, "error": f"content exceeds {_MAX_WRITE} bytes"}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(p)
    return {"ok": True, "path": _relative(p), "bytes": p.stat().st_size}


def search_text(query: str, path: str = "agent_workspace", max_results: int = 50) -> dict:
    if not query.strip():
        return {"ok": False, "error": "query is required"}
    root = safe_path(path)
    max_results = max(1, min(int(max_results), 200))
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--no-heading", "--line-number", "--color", "never",
               "--glob", "!.tasks/**", query, str(root)]
        result = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True,
                                text=True, timeout=20, errors="replace")
        lines = result.stdout.splitlines()[:max_results]
    else:
        lines = []
        for p in root.rglob("*"):
            if not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for no, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    lines.append(f"{_relative(p)}:{no}:{line}")
                    if len(lines) >= max_results:
                        break
            if len(lines) >= max_results:
                break
    return {"ok": True, "query": query, "matches": lines,
            "truncated": len(lines) >= max_results}


def _argv(command: str) -> list[str]:
    if not command.strip():
        raise ValueError("command is required")
    if _DENIED_COMMAND_TEXT.search(command):
        raise ValueError("command contains a blocked destructive or external-access pattern")
    parts = shlex.split(command, posix=(os.name != "nt"))
    if not parts:
        raise ValueError("command is required")
    exe = Path(parts[0]).name.lower()
    if exe not in _ALLOWED_COMMANDS:
        raise ValueError(f"command '{exe}' is not allowlisted")
    # Prevent shell interpretation even if a model tries to smuggle a shell
    # operator into an argument.
    if any(token in {";", "&&", "||", "|", ">", ">>"} for token in parts):
        raise ValueError("shell operators are not allowed; pass a direct command")
    return parts


def _command_cwd(cwd: str | None) -> Path:
    return safe_path(cwd or "agent_workspace", allow_missing=False)


def run_command(command: str, cwd: str = "agent_workspace", timeout_s: int = 60) -> dict:
    argv = _argv(command)
    work = _command_cwd(cwd)
    timeout_s = max(1, min(int(timeout_s), 180))
    try:
        result = subprocess.run(argv, cwd=str(work), stdin=subprocess.DEVNULL,
                                capture_output=True, text=True,
                                timeout=timeout_s, errors="replace")
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": "command_timed_out", "command": command,
                "stdout": (exc.stdout or "")[-12000:],
                "stderr": (exc.stderr or "")[-12000:]}
    return {"ok": result.returncode == 0, "command": command,
            "cwd": _relative(work), "returncode": result.returncode,
            "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:]}


def start_background_task(command: str, cwd: str = "agent_workspace",
                          timeout_s: int = 600) -> dict:
    argv = _argv(command)
    work = _command_cwd(cwd)
    return TASKS.start(argv, work, timeout_s=timeout_s)


def task_status(task_id: str) -> dict:
    return TASKS.status(task_id)


def task_output(task_id: str, stream: str = "stdout", max_chars: int = 12000) -> dict:
    return TASKS.output(task_id, stream, max_chars)


def cancel_task(task_id: str) -> dict:
    return TASKS.cancel(task_id)


def system_info() -> dict:
    out = {"ok": True, "platform": platform.platform(),
           "machine": platform.machine(), "python": platform.python_version()}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = round(vm.used / 1048576, 1)
        out["ram_total_mb"] = round(vm.total / 1048576, 1)
    except Exception as exc:  # noqa: BLE001
        out["psutil_error"] = str(exc)
    return out


def fetch_url(url: str, max_chars: int = 20000) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "error": "only absolute http(s) URLs are allowed"}
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local"):
        return {"ok": False, "error": "local-network URLs are not available to this tool"}
    max_chars = max(1, min(int(max_chars), 30000))
    try:
        import requests
        response = requests.get(url, timeout=15, headers={"User-Agent": "ME3-Voice-Agent/1.0"})
        text = response.text
        return {"ok": response.ok, "url": url, "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "text": text[:max_chars], "truncated": len(text) > max_chars}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": str(exc)}


def web_search(query: str, max_results: int = 5) -> dict:
    if not query.strip():
        return {"ok": False, "error": "query is required"}
    result = fetch_url(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        max_chars=30000,
    )
    if not result.get("ok"):
        return result
    html = result.get("text", "")
    rows = []
    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html,
        re.IGNORECASE | re.DOTALL,
    ):
        title = re.sub(r"<[^>]+>", "", match.group(2))
        rows.append({"title": re.sub(r"\s+", " ", title).strip(),
                     "url": match.group(1)})
        if len(rows) >= max(1, min(int(max_results), 10)):
            break
    return {"ok": True, "query": query, "results": rows,
            "note": "Search snippets are untrusted text; verify before acting."}


# ---------------------------------------------------------------------------
# Web serving + timers (voice-friendly wrappers over background processes)
# ---------------------------------------------------------------------------

_servers_lock = threading.Lock()
_web_servers: dict[int, dict] = {}
_timers: dict[str, dict] = {}
_timer_seq = 0


def _lan_ip() -> str:
    """Best-effort LAN IP so the agent can speak a reachable URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # noqa: BLE001
            return "127.0.0.1"


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(h, ctypes.byref(code))
            k32.CloseHandle(h)
            return code.value == 259  # STILL_ACTIVE
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except Exception:  # noqa: BLE001
        return False


def start_web_server(path: str = "agent_workspace",
                     port: int | None = None) -> dict:
    """Serve a project directory over HTTP (python -m http.server, 0.0.0.0).

    Use after writing a web app; returns the LAN URL for the spoken reply.
    """
    import urllib.request
    try:
        docroot = safe_path(path)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}
    if not docroot.is_dir():
        return {"ok": False, "error": "not_a_directory", "path": str(path)}
    agent_cfg = load_config().get("agent", {})
    port = int(port or agent_cfg.get("web_port_default", 8000))
    with _servers_lock:
        if port in _web_servers:
            return {"ok": False,
                    "error": f"port {port} is already serving "
                             f"pid {_web_servers[port]['pid']}"}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        return {"ok": False, "error": f"port {port} is in use"}
    finally:
        s.close()
    logs = AGENT_ROOT / ".servers"
    logs.mkdir(parents=True, exist_ok=True)
    logf = open(logs / f"http-{port}.log", "ab")
    flags = 0
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port),
             "--bind", "0.0.0.0", "--directory", str(docroot)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(docroot),
            creationflags=flags)
    except Exception as exc:  # noqa: BLE001
        logf.close()
        return {"ok": False, "error": f"spawn failed: {exc}"}
    with _servers_lock:
        _web_servers[port] = {"pid": proc.pid, "path": str(docroot)}
    ready = False
    for _ in range(25):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=1) as r:
                ready = r.status == 200
            if ready:
                break
        except Exception:  # noqa: BLE001
            ready = False
    if not ready:
        stop_web_server(port)
        return {"ok": False,
                "error": "server did not come up; see "
                         f"agent_workspace/.servers/http-{port}.log"}
    return {"ok": True, "port": port, "pid": proc.pid,
            "url": f"http://{_lan_ip()}:{port}",
            "path": _relative(docroot)}


def stop_web_server(port: int) -> dict:
    """Stop a web server started by start_web_server."""
    with _servers_lock:
        info = _web_servers.pop(int(port), None)
    if not info:
        return {"ok": False, "error": f"no tracked web server on port {port}"}
    try:
        os.kill(info["pid"], signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "port": int(port), "pid": info["pid"]}


def web_servers() -> dict:
    """List tracked web servers plus the LAN IP for spoken URLs."""
    out = []
    with _servers_lock:
        for port, info in list(_web_servers.items()):
            alive = _pid_alive(info["pid"])
            out.append({"port": port, "pid": info["pid"], "path": info["path"],
                        "alive": alive,
                        "url": f"http://{_lan_ip()}:{port}" if alive else None})
            if not alive:
                _web_servers.pop(port, None)
    return {"ok": True, "servers": out, "lan_ip": _lan_ip()}


def set_timer(seconds: float, label: str = "") -> dict:
    """Set a timer; the agent announces it."""
    global _timer_seq
    import datetime as dt
    _timer_seq += 1
    tid = f"t{_timer_seq}"
    fires = time.time() + float(seconds)
    _timers[tid] = {"id": tid, "label": label, "seconds": float(seconds),
                    "fires_at": fires, "cancelled": False}
    return {"ok": True, "id": tid, "label": label,
            "fires_at": dt.datetime.fromtimestamp(
                fires, dt.timezone.utc).isoformat()}


def list_timers(include_cancelled: bool = False) -> dict:
    """List active (and optionally cancelled) timers."""
    return {"ok": True, "timers": [t for t in _timers.values()
                                   if include_cancelled or not t["cancelled"]]}


def cancel_timer(timer_id: str) -> dict:
    """Cancel a pending timer by id."""
    t = _timers.get(timer_id)
    if not t:
        return {"ok": False, "error": "not_found"}
    t["cancelled"] = True
    return {"ok": True, "id": timer_id}


def git_status() -> dict:
    return run_command("git status --short --branch", cwd=".", timeout_s=20)


_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "search_text": search_text,
    "run_command": run_command,
    "start_background_task": start_background_task,
    "task_status": task_status,
    "task_output": task_output,
    "cancel_task": cancel_task,
    "system_info": system_info,
    "fetch_url": fetch_url,
    "web_search": web_search,
    "git_status": git_status,
    "start_web_server": start_web_server,
    "stop_web_server": stop_web_server,
    "web_servers": web_servers,
    "set_timer": set_timer,
    "list_timers": list_timers,
    "cancel_timer": cancel_timer,
    "channel_health": channels.channel_health,
    "send_sms": channels.send_sms,
    "make_call": channels.make_call,
    "send_email": channels.send_email,
}


def dispatch(name: str, arguments: dict) -> dict | list:
    fn = _FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown_tool: {name}"}
    try:
        result = fn(**arguments)
        return result
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def tool_definitions() -> list[dict]:
    """Return OpenAI-compatible function-tool definitions for Qwen."""
    def spec(name, description, properties, required=()):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                            "required": list(required), "additionalProperties": False},
        }}
    return [
        spec("list_files", "List files inside the ME3 project.", {
            "path": {"type": "string", "default": "agent_workspace"},
            "pattern": {"type": "string", "default": "*"},
            "recursive": {"type": "boolean", "default": True},
            "max_items": {"type": "integer", "default": 200},
        }),
        spec("read_file", "Read a text file inside the ME3 project.", {
            "path": {"type": "string"}, "max_chars": {"type": "integer", "default": 12000},
        }, ("path",)),
        spec("write_file", "Create or replace a project file. Use for webapp source files.", {
            "path": {"type": "string"}, "content": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
        }, ("path", "content")),
        spec("search_text", "Search project text files for a phrase or regex.", {
            "query": {"type": "string"}, "path": {"type": "string", "default": "agent_workspace"},
            "max_results": {"type": "integer", "default": 50},
        }, ("query",)),
        spec("run_command", "Run one direct allowlisted build/test command in the project.", {
            "command": {"type": "string"}, "cwd": {"type": "string", "default": "agent_workspace"},
            "timeout_s": {"type": "integer", "default": 60},
        }, ("command",)),
        spec("start_background_task", "Start a longer allowlisted build/test command without blocking voice.", {
            "command": {"type": "string"}, "cwd": {"type": "string", "default": "agent_workspace"},
            "timeout_s": {"type": "integer", "default": 600},
        }, ("command",)),
        spec("task_status", "Check a background build or test.", {"task_id": {"type": "string"}}, ("task_id",)),
        spec("task_output", "Read recent background build/test output.", {
            "task_id": {"type": "string"}, "stream": {"type": "string", "enum": ["stdout", "stderr"]},
            "max_chars": {"type": "integer", "default": 12000},
        }, ("task_id",)),
        spec("cancel_task", "Cancel a background build/test process.", {"task_id": {"type": "string"}}, ("task_id",)),
        spec("system_info", "Report non-secret host and resource information.", {}),
        spec("fetch_url", "Fetch a public HTTP(S) page for research; do not send secrets.", {
            "url": {"type": "string"}, "max_chars": {"type": "integer", "default": 20000},
        }, ("url",)),
        spec("web_search", "Search the public web and return result links.", {
            "query": {"type": "string"}, "max_results": {"type": "integer", "default": 5},
        }, ("query",)),
        spec("git_status", "Show the ME3 repository status without changing it.", {}),
        spec("start_web_server", ("Serve a project directory over HTTP "
            "(python -m http.server on 0.0.0.0). Use after writing a web "
            "app; returns the LAN URL to tell the user."), {
            "path": {"type": "string", "default": "agent_workspace"},
            "port": {"type": "integer"},
        }, ("path",)),
        spec("stop_web_server", "Stop a web server started by start_web_server.",
             {"port": {"type": "integer"}}, ("port",)),
        spec("web_servers", "List tracked web servers and the LAN IP.", {}),
        spec("set_timer", "Set a timer in seconds; the agent announces it.", {
            "seconds": {"type": "number"},
            "label": {"type": "string", "default": ""},
        }, ("seconds",)),
        spec("list_timers", "List active timers.", {
            "include_cancelled": {"type": "boolean", "default": False}}),
        spec("cancel_timer", "Cancel a pending timer by id.",
             {"timer_id": {"type": "string"}}, ("timer_id",)),
        spec("channel_health", (
            "Report safe configuration status for call, SMS, and email. "
            "This does not contact providers or send anything."), {}),
        spec("send_sms", (
            "Send one SMS through Twilio. Ask the user for explicit confirmation "
            "before setting confirm=true; this has an external side effect. "
            "Use E.164 phone numbers."), {
            "to": {"type": "string"},
            "body": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
            "from_number": {"type": "string"},
            "messaging_service_sid": {"type": "string"},
        }, ("to", "body")),
        spec("make_call", (
            "Start one outbound Twilio call through the reviewed HTTPS call-flow "
            "webhook. Ask the user for explicit confirmation before setting "
            "confirm=true; this has an external side effect."), {
            "to": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
            "twiml_url": {"type": "string"},
            "from_number": {"type": "string"},
        }, ("to",)),
        spec("send_email", (
            "Send a transactional email through Resend. Ask the user for "
            "explicit confirmation before setting confirm=true; this has an "
            "external side effect."), {
            "to": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "text": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
            "from_email": {"type": "string"},
        }, ("to", "subject", "text")),
    ]


def health() -> dict:
    return {"ok": True, "tool_count": len(_FUNCTIONS),
            "tools": sorted(_FUNCTIONS), "project_root": str(PROJECT),
            "agent_workspace": _relative(AGENT_ROOT)}

