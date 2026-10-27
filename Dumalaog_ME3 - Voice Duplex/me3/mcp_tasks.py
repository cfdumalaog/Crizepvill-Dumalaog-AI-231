"""ME3 Tasks MCP Server.

Exposes real-world task tools the voice agent can invoke. These are the
"real-world tasks" capability: timers, system info, file ops, and a bridge
to the existing VCM/ME2 command model.

Tools:
  - set_timer(seconds, label) -> {id, fires_at_iso}
  - list_timers() -> [...]
  - cancel_timer(id) -> {ok}
  - system_info() -> {cpu_pct, ram_mb, uptime_s, platform}
  - read_file(path) -> {content}          (sandboxed to project dir)
  - write_file(path, content) -> {ok, path}
  - run_command(cmd) -> {stdout, stderr, returncode}  (allowlisted)
  - vcm_classify(audio_b64) -> {label, confidence}   # bridge to ME2 ONNX
  - health() -> {ok, tools: [...]}

Run:  python -m me3.mcp_tasks
"""
from __future__ import annotations

import base64
import datetime as dt
import os
import subprocess
import time
from pathlib import Path

from fastmcp import FastMCP

from .config import load_config
from . import tools as project_tools

mcp = FastMCP("me3-tasks")

_project_root = Path(__file__).resolve().parent.parent
_timers: dict[str, dict] = {}
_timer_seq = 0

# Safety: only these commands may run (real-world task bridge).
_ALLOWED_CMDS = {"date", "uname", "free", "df", "uptime", "hostname", "whoami"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@mcp.tool()
def set_timer(seconds: float, label: str = "") -> dict:
    """Set a timer. The agent announces it; fires_at is when it elapses.

    Args:
        seconds: timer duration in seconds.
        label: human label for the timer.

    Returns:
        {id, label, seconds, fires_at_iso}
    """
    global _timer_seq
    _timer_seq += 1
    tid = f"t{_timer_seq}"
    fires = time.time() + seconds
    _timers[tid] = {
        "id": tid, "label": label, "seconds": seconds,
        "fires_at": fires, "fires_at_iso": dt.datetime.fromtimestamp(
            fires, dt.timezone.utc).isoformat(),
        "cancelled": False,
    }
    return {"id": tid, "label": label, "seconds": seconds,
            "fires_at_iso": _timers[tid]["fires_at_iso"]}


@mcp.tool()
def list_timers(include_cancelled: bool = False) -> list:
    """List active (and optionally cancelled) timers."""
    return [t for t in _timers.values()
            if include_cancelled or not t["cancelled"]]


@mcp.tool()
def cancel_timer(timer_id: str) -> dict:
    """Cancel a pending timer by id."""
    t = _timers.get(timer_id)
    if not t:
        return {"ok": False, "error": "not_found"}
    t["cancelled"] = True
    return {"ok": True, "id": timer_id}


@mcp.tool()
def system_info() -> dict:
    """Return CPU%, RAM MB, uptime, platform for the Pi."""
    import platform
    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uptime_s": round(time.time() - _boot_epoch(), 1),
    }
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=0.2)
        vm = psutil.virtual_memory()
        out["ram_mb"] = round(vm.used / (1024 * 1024), 1)
        out["ram_total_mb"] = round(vm.total / (1024 * 1024), 1)
    except Exception as e:  # noqa: BLE001
        out["psutil_error"] = str(e)
    return out


_boot = None


def _boot_epoch() -> float:
    global _boot
    if _boot is None:
        try:
            with open("/proc/uptime") as f:
                up = float(f.read().split()[0])
            _boot = time.time() - up
        except Exception:  # noqa: BLE001
            _boot = time.time()
    return _boot


def _safe_path(path: str) -> Path:
    """Resolve path, must stay within project root."""
    root = _project_root.resolve()
    p = (Path(path) if os.path.isabs(path) else root / path).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"path escapes project root: {path}")
    return p


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a text file within the project directory."""
    try:
        p = _safe_path(path)
        return {"ok": True, "path": str(p), "content": p.read_text(encoding="utf-8")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@mcp.tool()
def write_file(path: str, content: str) -> dict:
    """Write a text file within the project directory."""
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(p)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@mcp.tool()
def run_command(cmd: str) -> dict:
    """Run an allowlisted shell command (date/uname/free/df/uptime/...).

    Only commands in the allowlist may run; anything else is refused.
    """
    parts = cmd.split()
    if not parts or parts[0] not in _ALLOWED_CMDS:
        return {"ok": False,
                "error": f"command not allowed. allowlist: {sorted(_ALLOWED_CMDS)}"}
    try:
        r = subprocess.run(parts, capture_output=True, text=True, timeout=10)
        return {"ok": True, "stdout": r.stdout, "stderr": r.stderr,
                "returncode": r.returncode}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@mcp.tool()
def vcm_classify(audio_b64: str, sample_rate: int = 16000) -> dict:
    """Classify a voice command using the ME2 ONNX model (bridge).

    Loads the ME2 static-QDQ ONNX if present and runs one inference.
    Returns {label, confidence} or {ok:False} if the model isn't deployed.
    """
    try:
        import onnxruntime as ort
        import numpy as np
        cfg = load_config()
        # ME2 onnx expected at ../ME2_Tiny_Voice_Command_Model/export/*.onnx
        cand = _project_root.parent / "ME2_Tiny_Voice_Command_Model" / "export"
        onnx_files = list(cand.glob("*.onnx")) if cand.exists() else []
        if not onnx_files:
            return {"ok": False,
                    "error": "ME2 ONNX not found at ME2_Tiny_Voice_Command_Model/export"}
        sess = ort.InferenceSession(str(onnx_files[0]),
                                    providers=["CPUExecutionProvider"])
        raw = base64.b64decode(audio_b64)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # NOTE: exact preprocessing (mel, reshape) must match ME2 export;
        # this is a bridge stub until ME2's preprocess is imported.
        return {"ok": True, "model": onnx_files[0].name,
                "note": "bridge stub - wire ME2 mel preprocess before use",
                "input_samples": int(len(pcm))}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# The same bounded registry is exposed through MCP as well as the direct Qwen
# tool loop.  This keeps the Pi, local voice loop, and any future MCP client on
# one implementation instead of silently creating separate permission rules.
@mcp.tool()
def list_project_files(path: str = "agent_workspace", pattern: str = "*",
                       recursive: bool = True, max_items: int = 200) -> dict:
    """List project files available to the ME3 agent."""
    return project_tools.list_files(path, pattern, recursive, max_items)


@mcp.tool()
def read_project_file(path: str, max_chars: int = 12000) -> dict:
    """Read one project text file without exposing protected paths."""
    return project_tools.read_file(path, max_chars)


@mcp.tool()
def write_project_file(path: str, content: str, overwrite: bool = False) -> dict:
    """Create or replace one project text file."""
    return project_tools.write_file(path, content, overwrite)


@mcp.tool()
def search_project_text(query: str, path: str = "agent_workspace",
                        max_results: int = 50) -> dict:
    """Search project text files."""
    return project_tools.search_text(query, path, max_results)


@mcp.tool()
def run_project_command(command: str, cwd: str = "agent_workspace",
                        timeout_s: int = 60) -> dict:
    """Run one bounded, direct allowlisted build/test command."""
    return project_tools.run_command(command, cwd, timeout_s)


@mcp.tool()
def start_project_task(command: str, cwd: str = "agent_workspace",
                       timeout_s: int = 600) -> dict:
    """Start a longer allowlisted command without blocking voice input."""
    return project_tools.start_background_task(command, cwd, timeout_s)


@mcp.tool()
def project_task_status(task_id: str) -> dict:
    """Read background task status."""
    return project_tools.task_status(task_id)


@mcp.tool()
def project_task_output(task_id: str, stream: str = "stdout",
                        max_chars: int = 12000) -> dict:
    """Read recent background task output."""
    return project_tools.task_output(task_id, stream, max_chars)


@mcp.tool()
def cancel_project_task(task_id: str) -> dict:
    """Cancel one background task."""
    return project_tools.cancel_task(task_id)


@mcp.tool()
def fetch_public_url(url: str, max_chars: int = 20000) -> dict:
    """Fetch a public HTTP(S) page without exposing local-network paths."""
    return project_tools.fetch_url(url, max_chars)


@mcp.tool()
def search_public_web(query: str, max_results: int = 5) -> dict:
    """Search the public web for links and titles."""
    return project_tools.web_search(query, max_results)


@mcp.tool()
def project_git_status() -> dict:
    """Show ME3 repository status without changing it."""
    return project_tools.git_status()


@mcp.tool()
def channel_health() -> dict:
    """Report safe call, SMS, and email configuration without network calls."""
    return project_tools.channels.channel_health()


@mcp.tool()
def send_sms(to: str, body: str, confirm: bool = False,
             from_number: str = "", messaging_service_sid: str = "") -> dict:
    """Send one Twilio SMS only after explicit confirmation."""
    return project_tools.channels.send_sms(
        to, body, confirm=confirm,
        from_number=from_number or None,
        messaging_service_sid=messaging_service_sid or None,
    )


@mcp.tool()
def make_call(to: str, confirm: bool = False, twiml_url: str = "",
              from_number: str = "") -> dict:
    """Start one Twilio call through a reviewed HTTPS call-flow webhook."""
    return project_tools.channels.make_call(
        to, confirm=confirm, twiml_url=twiml_url or None,
        from_number=from_number or None,
    )


@mcp.tool()
def send_email(to: list[str], subject: str, text: str, confirm: bool = False,
               from_email: str = "") -> dict:
    """Send one Resend transactional email only after explicit confirmation."""
    return project_tools.channels.send_email(
        to, subject, text, confirm=confirm, from_email=from_email or None,
    )


@mcp.tool()
def health() -> dict:
    """Report tasks server health and available tools."""
    return {"ok": True,
            "tools": ["set_timer", "list_timers", "cancel_timer", "system_info",
                      "read_file", "write_file", "run_command", "vcm_classify",
                      "list_project_files", "read_project_file", "write_project_file",
                      "search_project_text", "run_project_command", "start_project_task",
                      "project_task_status", "project_task_output", "cancel_project_task",
                      "fetch_public_url", "search_public_web", "project_git_status",
                      "channel_health", "send_sms", "make_call", "send_email"],
            "allowlist_cmds": sorted(_ALLOWED_CMDS)}


if __name__ == "__main__":
    mcp.run()
