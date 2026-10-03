"""Detached supervisor for restarting the local VCM or recording UI."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parent_exited(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        wait_for_single_object.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return True
        try:
            return wait_for_single_object(handle, 0) == 0  # WAIT_OBJECT_0
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_parent(pid: int, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _parent_exited(pid):
            time.sleep(0.5)  # allow the audio device and listener socket to be released
            return
        time.sleep(0.1)
    raise TimeoutError(f"Process {pid} did not exit; refusing to start a duplicate listener")


def _decode_arguments(encoded: str) -> list[str]:
    value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("restart arguments must be a JSON string list")
    return value


def restart(app: str, parent_pid: int, port: int, encoded_arguments: str) -> int:
    if app not in {"vcm", "recorder"}:
        raise ValueError("unsupported app")
    if parent_pid <= 0 or not 1 <= port <= 65535:
        raise ValueError("invalid process ID or port")
    arguments = _decode_arguments(encoded_arguments)
    if app == "recorder" and arguments:
        raise ValueError("recorder restart cannot accept launch arguments")

    _wait_for_parent(parent_pid)
    if os.name == "nt":
        workspace = PROJECT_ROOT.parents[1]
        python = workspace / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            raise FileNotFoundError(f"Shared workspace Python is missing: {python}")
        python_executable = str(python)
        creationflags = subprocess.DETACHED_PROCESS
    else:
        python_executable = sys.executable
        creationflags = 0

    entrypoint = PROJECT_ROOT / ("antigrav_demo.py" if app == "vcm" else "record_dataset.py")
    command = [python_executable, str(entrypoint), *arguments]
    kwargs = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    child = subprocess.Popen(command, **kwargs)
    return child.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", choices=("vcm", "recorder"), required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arguments", required=True)
    args = parser.parse_args()
    restart(args.app, args.parent_pid, args.port, args.arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
