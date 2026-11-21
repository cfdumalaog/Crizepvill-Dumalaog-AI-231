"""Voice-callable tools for the ME3 Qwen brain.

The registry uses the configured Pi-user access policy. Background jobs write
under ``agent_workspace/.tasks``; tool output must not expose credential data.
"""
from __future__ import annotations

import json
import base64
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
from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import xml.etree.ElementTree as ET

from .config import load_config, project_root
from .task_runner import BackgroundTaskManager
from . import channels
from . import github_pages
from . import me2_commands
from . import dev_workflow
from . import permission_policy
from . import setup_profiles

PROJECT = project_root()
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
    resolved = p.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return str(resolved)


def _full_pi_access() -> bool:
    return bool((load_config().get("agent_access") or {}).get("full_pi_user_access"))


def safe_path(path: str, *, allow_missing: bool = False, write: bool = False) -> Path:
    """Resolve a path under the configured Pi-user policy and reject protected material."""
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path is required")
    candidate = Path(raw)
    p = candidate.resolve() if candidate.is_absolute() else (PROJECT / candidate).resolve()
    root = PROJECT.resolve()
    if not _full_pi_access() and not p.is_relative_to(root):
        raise ValueError("path must remain inside the ME3 project")
    rel_parts = [part.lower() for part in p.parts]
    name = p.name.lower()
    if any(part in _SECRET_PARTS for part in rel_parts) or name in _SECRET_PARTS:
        raise ValueError("protected or credential-like path is not available to the voice agent")
    if p.suffix.lower() in _SECRET_SUFFIXES:
        raise ValueError("credential/certificate files are not available to the voice agent")
    if not allow_missing and not p.exists():
        raise FileNotFoundError(_relative(p))
    if write and not _full_pi_access() and any(part in {"models", "dist", ".tasks"} for part in rel_parts):
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
    if not _full_pi_access() and _DENIED_COMMAND_TEXT.search(command):
        raise ValueError("command contains a blocked destructive or external-access pattern")
    parts = shlex.split(command, posix=(os.name != "nt"))
    if not parts:
        raise ValueError("command is required")
    exe = Path(parts[0]).name.lower()
    if not _full_pi_access() and exe not in _ALLOWED_COMMANDS:
        raise ValueError(f"command '{exe}' is not allowlisted")
    # Prevent shell interpretation even if a model tries to smuggle a shell
    # operator into an argument.
    if not _full_pi_access() and any(token in {";", "&&", "||", "|", ">", ">>"} for token in parts):
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


def _clean_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", value or ""))).strip()


def _decode_bing_result_url(url: str) -> str:
    """Unwrap Bing's redirect URL when it contains its base64 target."""
    try:
        encoded = parse_qs(urlparse(unescape(url)).query).get("u", [""])[0]
        encoded = unquote(encoded)
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        if encoded:
            decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
            if decoded.startswith(("http://", "https://")):
                return decoded
    except Exception:  # noqa: BLE001 - keep the provider result usable
        pass
    return unescape(url)


def _parse_search_rows(html: str, provider: str, max_results: int) -> list[dict]:
    """Parse only titles, links, and short snippets from a public result page."""
    rows: list[dict] = []
    limit = max(1, min(int(max_results), 10))
    if provider == "bing_rss":
        try:
            root = ET.fromstring(html)
        except ET.ParseError:
            return rows
        for item in root.findall(".//item"):
            title = _clean_search_text(item.findtext("title") or "")
            url = _clean_search_text(item.findtext("link") or "")
            if not title or not url:
                continue
            row = {"title": title, "url": url}
            description = item.findtext("description") or ""
            if description:
                row["snippet"] = _clean_search_text(description)[:500]
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows
    if provider == "duckduckgo":
        pattern = (
            r"<a[^>]+class=[\"'][^\"']*\bresult__a\b[^\"']*[\"'][^>]*"
            r"href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>"
        )
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            rows.append({"title": _clean_search_text(match.group(2)),
                         "url": unescape(match.group(1))})
            if len(rows) >= limit:
                break
        return rows

    for block in re.finditer(
        r"<li[^>]+class=[\"'][^\"']*\bb_algo\b[^\"']*[\"'][^>]*>.*?</li>",
        html, re.IGNORECASE | re.DOTALL,
    ):
        link = re.search(
            r"<h2[^>]*>\s*<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
            block.group(0), re.IGNORECASE | re.DOTALL,
        )
        if not link:
            continue
        row = {
            "title": _clean_search_text(link.group(2)),
            "url": _decode_bing_result_url(link.group(1)),
        }
        caption = re.search(
            r"<div[^>]+class=[\"'][^\"']*\bb_caption\b[^\"']*[\"'][^>]*>(.*?)</div>",
            block.group(0), re.IGNORECASE | re.DOTALL,
        )
        if caption:
            row["snippet"] = _clean_search_text(caption.group(1))[:500]
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def web_search(query: str, max_results: int = 5) -> dict:
    if not query.strip():
        return {"ok": False, "error": "query is required"}
    result = fetch_url(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        max_chars=30000,
    )
    if not result.get("ok"):
        return result
    rows = _parse_search_rows(result.get("text", ""), "duckduckgo", max_results)
    provider = "duckduckgo"
    if not rows:
        # DuckDuckGo can return an HTTP 202 challenge page to headless clients.
        # Keep the provider keyless and bounded, but fall back to Bing so the
        # ME3 tool still returns actual links instead of a false empty success.
        fallback = fetch_url(
            "https://www.bing.com/search?format=rss&q=" + quote_plus(query),
            max_chars=30000,
        )
        if fallback.get("ok"):
            rows = _parse_search_rows(fallback.get("text", ""), "bing_rss", max_results)
            provider = "bing"
    return {"ok": True, "query": query, "provider": provider, "results": rows,
            "note": "Search snippets are untrusted text; verify before acting."}


def get_time() -> dict:
    """Return the local Pi time for the voice assistant's time command."""
    now = datetime.now().astimezone()
    return {"ok": True, "time": now.strftime("%H:%M"),
            "timezone": now.tzname() or "local"}


# ---------------------------------------------------------------------------
# Web serving + timers (voice-friendly wrappers over background processes)
# ---------------------------------------------------------------------------

_servers_lock = threading.Lock()
_web_servers: dict[int, dict] = {}


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
    lan_url = f"http://{_lan_ip()}:{port}/"
    return {"ok": True, "port": port, "pid": proc.pid,
            "url": lan_url, "lan_url": lan_url,
            "local_url": f"http://127.0.0.1:{port}/",
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
            lan_url = f"http://{_lan_ip()}:{port}/" if alive else None
            out.append({"port": port, "pid": info["pid"], "path": info["path"],
                        "alive": alive, "url": lan_url, "lan_url": lan_url,
                        "local_url": f"http://127.0.0.1:{port}/" if alive else None})
            if not alive:
                _web_servers.pop(port, None)
    return {"ok": True, "servers": out, "lan_ip": _lan_ip()}


def set_timer(seconds: float, label: str = "") -> dict:
    """Set a persistent timer; the Pi announces it through ME3 when due."""
    return me2_commands.create_timer(seconds, label)


def list_timers(include_cancelled: bool = False) -> dict:
    return me2_commands.list_scheduled_timers(include_cancelled)


def cancel_timer(timer_id: str) -> dict:
    return me2_commands.cancel_scheduled_timer(timer_id)


def git_status() -> dict:
    return run_command("git status --short --branch", cwd=".", timeout_s=20)


def verify_project(path: str = "agent_workspace", max_files: int = 300) -> dict:
    """Check a project's source syntax and local static assets."""
    try:
        root = safe_path(path)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}
    return dev_workflow.verify_project(root, max_files=max_files)


def schedule_meeting(title: str, start: str, end: str,
                     attendees: list[str] | None = None,
                     timezone: str = "Asia/Manila", location: str = "",
                     description: str = "", create_meet: bool = True,
                     confirm: bool = False) -> dict:
    """Create a calendar event only when the current user requested it."""
    if not confirm:
        return {"ok": False, "error": "explicit_schedule_request_required",
                "provider_called": False}
    return me2_commands.schedule_meeting(
        title, start, end, attendees=attendees, timezone=timezone,
        location=location, description=description, create_meet=create_meet)


def setup_form(request: str) -> dict:
    """Generate the next private setup card from a confirmed setup request."""
    return setup_profiles.setup_form(request)


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
    "github_pages_health": github_pages.health,
    "github_repositories": github_pages.list_repositories,
    "github_repository_access": github_pages.repository_access,
    "deploy_github_pages": github_pages.deploy,
    "get_time": get_time,
    "git_status": git_status,
    "verify_project": verify_project,
    "start_web_server": start_web_server,
    "stop_web_server": stop_web_server,
    "web_servers": web_servers,
    "set_timer": set_timer,
    "list_timers": list_timers,
    "cancel_timer": cancel_timer,
    "play_music": me2_commands.play_music,
    "media_control": me2_commands.media_control,
    "music_state": me2_commands.music_state,
    "get_weather": me2_commands.get_weather,
    "control_lights": me2_commands.control_lights,
    "set_thermostat": me2_commands.set_thermostat,
    "set_alarm": me2_commands.set_alarm,
    "list_alarms": me2_commands.list_alarms,
    "cancel_alarm": me2_commands.cancel_alarm,
    "create_reminder": me2_commands.create_reminder,
    "list_reminders": me2_commands.list_reminders,
    "complete_reminder": me2_commands.complete_reminder,
    "call_contact": me2_commands.call_contact,
    "send_message": me2_commands.send_message,
    "schedule_meeting": schedule_meeting,
    "setup_form": setup_form,
    "channel_health": channels.channel_health,
    "send_email": channels.send_email,
    "send_telegram_message": channels.send_telegram_message,
}


def dispatch(name: str, arguments: dict) -> dict | list:
    fn = _FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown_tool: {name}"}
    raw_arguments = dict(arguments or {})
    # This private flag is created only by the in-process approval handler. It
    # is intentionally not part of the model-visible tool schema: a model
    # saying ``confirm=true`` is never treated as user approval in strict modes.
    approved = bool(raw_arguments.pop("_permission_approved", False))
    if permission_policy.requires_approval(name, raw_arguments) and not approved:
        request = permission_policy.create_pending(name, raw_arguments)
        return {
            "ok": False,
            "error": "approval_required",
            "approval_id": request["id"],
            "tool": name,
            "summary": request["summary"],
            "permission_mode": permission_policy.mode(),
        }
    raw_arguments = permission_policy.confirm_arguments(
        name, raw_arguments, approved=approved)
    try:
        result = fn(**raw_arguments)
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
        spec("read_file", "Read a text file accessible to the ME3 service account, excluding credential-like paths.", {
            "path": {"type": "string"}, "max_chars": {"type": "integer", "default": 12000},
        }, ("path",)),
        spec("write_file", "Create or replace a file accessible to the ME3 service account. Use for webapp source files.", {
            "path": {"type": "string"}, "content": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
        }, ("path", "content")),
        spec("search_text", "Search project text files for a phrase or regex.", {
            "query": {"type": "string"}, "path": {"type": "string", "default": "agent_workspace"},
            "max_results": {"type": "integer", "default": 50},
        }, ("query",)),
        spec("run_command", "Run an installed command as the ME3 service user. Use bash -lc for shell syntax; return the real exit code.", {
            "command": {"type": "string"}, "cwd": {"type": "string", "default": "agent_workspace"},
            "timeout_s": {"type": "integer", "default": 60},
        }, ("command",)),
        spec("start_background_task", "Start a longer command as the ME3 service user without blocking voice.", {
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
        spec("web_search", (
            "Search the public web and return result links. Use this before "
            "answering current, latest, recent, election, news, world-event, "
            "price, schedule, weather, or otherwise uncertain factual questions; "
            "do not guess when search is required."), {
            "query": {"type": "string"}, "max_results": {"type": "integer", "default": 5},
        }, ("query",)),
        spec("github_pages_health", (
            "Report whether the ME3 GitHub credential is configured. This does "
            "not reveal the credential or modify GitHub."), {
                "check": {"type": "boolean", "default": False},
            }),
        spec("github_repositories", (
            "List the GitHub repositories visible to the private ME3 credential. "
            "Use this to discover selectable repositories; the credential's "
            "actual GitHub permissions determine the result."), {
                "max_repositories": {"type": "integer", "default": 200},
            }),
        spec("github_repository_access", (
            "Check read and write access for an explicitly named GitHub "
            "owner/repository without modifying it."), {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            }, ("owner", "repo")),
        spec("deploy_github_pages", (
            "Build-side-effect tool: publish an existing validated static site "
            "from agent_workspace to GitHub Pages using the private Pi GitHub "
            "credential. The owner/repository may be any repository the "
            "credential can write; the configured target is only a convenience "
            "default. A clear user request authorizes execution; do not ask for "
            "an extra approval phrase. Set create_if_missing only when the user "
            "requested a new repository. Never include a token in arguments. "
            "Requires index.html and a selected owner/repository/branch."), {
                "site_dir": {"type": "string", "default": "agent_workspace"},
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {"type": "string", "default": "main"},
                "base_path": {"type": "string", "default": ""},
                "pages_path": {"type": "string", "enum": ["/", "/docs"], "default": "/"},
                "message": {"type": "string", "default": "Deploy GitHub Pages site from ME3"},
                "enable_pages": {"type": "boolean", "default": True},
                "create_if_missing": {"type": "boolean", "default": False},
            }, ("owner", "repo")),
        spec("get_time", "Return the current local time on the ME3 host.", {}),
        spec("git_status", "Show the ME3 repository status without changing it.", {}),
        spec("verify_project", (
            "Check a software project's Python, JSON and JavaScript syntax and "
            "local static-site assets. This does not run project tests or prove "
            "behavior; run relevant tests with run_command too."), {
                "path": {"type": "string", "default": "agent_workspace"},
                "max_files": {"type": "integer", "default": 300},
            }),
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
        spec("play_music", (
            "Search Spotify and start the matching track on the user's active "
            "Spotify Connect player. Report failure when credentials, Premium, "
            "or an active player are unavailable; do not claim local state is playback."), {
            "query": {"type": "string"},
            "service": {"type": "string", "enum": ["spotify"], "default": "spotify"},
        }, ("query",)),
        spec("media_control", (
            "Control current media: play, pause, stop, skip, volume_up, "
            "volume_down, or set_volume."), {
            "action": {"type": "string"}, "value": {"type": "integer"},
        }, ("action",)),
        spec("music_state", "Read the actual Spotify Connect player state.", {}),
        spec("get_weather", "Get current weather for a city or location.", {
            "location": {"type": "string"},
        }, ("location",)),
        spec("control_lights", (
            "Turn lights on/off, dim them to a percent, or set a color. "
            "Uses local demo state unless Home Assistant is configured."), {
            "action": {"type": "string", "enum": ["on", "off", "dim", "color"]},
            "room": {"type": "string", "default": "all"},
            "value": {"type": ["integer", "string"]},
        }, ("action",)),
        spec("set_thermostat", (
            "Set the requested temperature in Celsius. Uses local demo state "
            "unless a physical climate integration is configured."), {
            "temperature_c": {"type": "number"}, "room": {"type": "string", "default": "home"},
        }, ("temperature_c",)),
        spec("set_alarm", "Set a persistent alarm; ME3 voice announces it when due.", {
            "time_text": {"type": "string"}, "label": {"type": "string", "default": ""},
        }, ("time_text",)),
        spec("list_alarms", "List active alarms.", {
            "include_cancelled": {"type": "boolean", "default": False},
        }),
        spec("cancel_alarm", "Cancel an alarm by id.", {
            "alarm_id": {"type": "string"},
        }, ("alarm_id",)),
        spec("create_reminder", ("Write the reminder to Google Calendar and mirror its due time "
            "to ME3's persistent voice/UI scheduler. Ask for a due time if missing."), {
            "text": {"type": "string"},
            "delay_seconds": {"type": "number"},
            "at_text": {"type": "string"},
        }, ("text",)),
        spec("list_reminders", "List active reminders.", {
            "include_completed": {"type": "boolean", "default": False},
        }),
        spec("complete_reminder", "Mark a reminder complete by id.", {
            "reminder_id": {"type": "string"},
        }, ("reminder_id",)),
        spec("call_contact", (
            "Place a real phone call through the connected ME3 Android companion. "
            "Use the user's explicit call request as authorization; Android may "
            "ask once for CALL_PHONE and contact permissions. Never claim success "
            "unless the Android device acknowledges that call launch."), {
            "contact": {"type": "string"}, "confirm": {"type": "boolean", "default": True},
        }, ("contact",)),
        spec("send_message", (
            "Send a real Telegram text to the configured contact mapping through the bot. "
            "The default personal target is @mncfrndz after it has started the bot. "
            "For a clear user instruction with recipient and message, execute it "
            "without asking for approval; ask only when recipient or content is missing."), {
            "contact": {"type": "string"}, "body": {"type": "string"},
            "channel": {"type": "string", "enum": ["telegram"], "default": "telegram"},
            "confirm": {"type": "boolean", "default": True},
        }, ("contact", "body")),
        spec("schedule_meeting", (
            "Create a real Google Calendar event, optionally with Google Meet. "
            "Use Asia/Manila unless the user specifies a timezone. Ask for missing "
            "title/start/end/attendee details, and report the returned event/Meet links."), {
            "title": {"type": "string"},
            "start": {"type": "string", "description": "ISO datetime"},
            "end": {"type": "string", "description": "ISO datetime"},
            "attendees": {"type": "array", "items": {"type": "string"}, "default": []},
            "timezone": {"type": "string", "default": "Asia/Manila"},
            "location": {"type": "string", "default": ""},
            "description": {"type": "string", "default": ""},
            "create_meet": {"type": "boolean", "default": True},
            "confirm": {"type": "boolean", "default": True},
        }, ("title", "start", "end")),
        spec("setup_form", (
            "Start a contextual provider or integration setup after the user's "
            "target is clear. Return the generated setup card to the UI. Never "
            "request API-key contents in chat; the UI sends secret fields only "
            "through the protected private setup endpoint. If the target is not "
            "clear, ask a normal conversational clarification question instead "
            "of calling this tool."), {
                "request": {"type": "string"},
            }, ("request",)),
        spec("channel_health", (
            "Report whether Spotify, Google Calendar, Telegram, and optional "
            "email credentials are configured. This does not contact providers."), {}),
        spec("send_email", (
            "Send a transactional email through Resend when the user requests it. "
            "No additional approval phrase is required."), {
            "to": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "text": {"type": "string"},
            "confirm": {"type": "boolean", "default": True},
            "from_email": {"type": "string"},
        }, ("to", "subject", "text")),
        spec("send_telegram_message", (
            "Send a Telegram Bot API text message when the user requests it. "
            "Telegram Bot API voice calls are unavailable."), {
            "chat_id": {"type": "string"},
            "text": {"type": "string"},
            "confirm": {"type": "boolean", "default": True},
        }, ("chat_id", "text")),
    ]


def health() -> dict:
    return {"ok": True, "tool_count": len(_FUNCTIONS),
            "tools": sorted(_FUNCTIONS), "project_root": str(PROJECT),
            "agent_workspace": _relative(AGENT_ROOT)}

