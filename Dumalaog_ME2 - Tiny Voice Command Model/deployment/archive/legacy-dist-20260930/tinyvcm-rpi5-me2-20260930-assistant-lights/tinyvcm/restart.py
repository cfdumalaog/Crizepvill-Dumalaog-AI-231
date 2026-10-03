"""Queue a local service restart without shelling out through a command string."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = PROJECT_ROOT / "scripts" / "restart_local_app.py"


def build_supervisor_command(app: str, parent_pid: int, port: int, arguments=()) -> list[str]:
    """Build a fixed-script command; launch arguments are encoded as data."""
    if app not in {"vcm", "recorder"}:
        raise ValueError("app must be 'vcm' or 'recorder'")
    if parent_pid <= 0 or not 1 <= port <= 65535:
        raise ValueError("parent PID and port must be valid")
    args = list(arguments)
    if app == "recorder" and args:
        raise ValueError("recorder restart does not accept launch arguments")
    encoded = base64.urlsafe_b64encode(
        json.dumps(args, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return [
        sys.executable,
        str(SUPERVISOR),
        "--app", app,
        "--parent-pid", str(parent_pid),
        "--port", str(port),
        "--arguments", encoded,
    ]


def queue_restart(app: str, port: int, arguments=()) -> None:
    """Start a detached supervisor that waits for this process to exit."""
    command = build_supervisor_command(app, os.getpid(), port, arguments)
    kwargs = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)
