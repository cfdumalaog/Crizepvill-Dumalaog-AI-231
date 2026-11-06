"""Bounded background-process support for the ME3 voice tool registry.

The voice agent must be able to start a web-app build or test without blocking
the microphone loop.  This module deliberately keeps process state inside the
ME3 project and records only command/output metadata; it never receives or
persists API credentials.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path


class BackgroundTaskManager:
    """Start, inspect, tail, and cancel bounded project-local processes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.task_root = self.project_root / "agent_workspace" / ".tasks"
        self.task_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict] = {}

    def _metadata_path(self, task_id: str) -> Path:
        return self.task_root / f"{task_id}.json"

    def start(self, argv: list[str], cwd: Path, timeout_s: int = 600) -> dict:
        timeout_s = max(1, min(int(timeout_s), 1800))
        task_id = f"task-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        out_path = self.task_root / f"{task_id}.stdout.log"
        err_path = self.task_root / f"{task_id}.stderr.log"
        metadata = {
            "task_id": task_id,
            "argv": argv,
            "cwd": str(cwd),
            "started_at": time.time(),
            "timeout_s": timeout_s,
            "stdout_path": str(out_path),
            "stderr_path": str(err_path),
            "status": "starting",
        }
        out = out_path.open("w", encoding="utf-8", errors="replace")
        err = err_path.open("w", encoding="utf-8", errors="replace")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                text=True,
                creationflags=creationflags,
            )
        except Exception:
            out.close()
            err.close()
            raise
        metadata.update({"pid": proc.pid, "status": "running"})
        with self._lock:
            self._tasks[task_id] = {"proc": proc, "metadata": metadata,
                                    "out": out, "err": err}
        self._metadata_path(task_id).write_text(
            json.dumps({k: v for k, v in metadata.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        threading.Thread(target=self._watch, args=(task_id,), daemon=True).start()
        return self.status(task_id)

    def _watch(self, task_id: str) -> None:
        with self._lock:
            item = self._tasks.get(task_id)
        if not item:
            return
        proc = item["proc"]
        timeout_s = item["metadata"]["timeout_s"]
        try:
            rc = proc.wait(timeout=timeout_s)
            status = "completed" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            self._terminate(item)
            rc = None
            status = "timed_out"
        finally:
            item["out"].close()
            item["err"].close()
        with self._lock:
            item["metadata"].update({
                "returncode": rc,
                "status": status,
                "finished_at": time.time(),
            })
            self._metadata_path(task_id).write_text(
                json.dumps(item["metadata"], indent=2) + "\n", encoding="utf-8"
            )

    @staticmethod
    def _terminate(item: dict) -> None:
        proc = item["proc"]
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.2)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    def status(self, task_id: str) -> dict:
        with self._lock:
            item = self._tasks.get(task_id)
        if item:
            meta = dict(item["metadata"])
            live = item["proc"].poll()
            if live is not None and meta.get("status") in {"starting", "running"}:
                meta["status"] = "completed" if live == 0 else "failed"
                meta["returncode"] = live
            return self._public(meta)
        path = self._metadata_path(task_id)
        if not path.is_file():
            return {"ok": False, "error": "task_not_found", "task_id": task_id}
        try:
            return self._public(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            return {"ok": False, "error": f"invalid_task_metadata: {exc}"}

    @staticmethod
    def _public(meta: dict) -> dict:
        out = {k: v for k, v in meta.items()
               if k not in {"stdout_path", "stderr_path"}}
        out["ok"] = True
        return out

    def output(self, task_id: str, stream: str = "stdout", max_chars: int = 12000) -> dict:
        if stream not in {"stdout", "stderr"}:
            return {"ok": False, "error": "stream must be stdout or stderr"}
        status = self.status(task_id)
        if not status.get("ok"):
            return status
        with self._lock:
            item = self._tasks.get(task_id)
            if item:
                path = Path(item["metadata"][f"{stream}_path"])
            else:
                raw = json.loads(self._metadata_path(task_id).read_text(encoding="utf-8"))
                path = Path(raw[f"{stream}_path"])
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        max_chars = max(1, min(int(max_chars), 30000))
        return {"ok": True, "task_id": task_id, "stream": stream,
                "text": text[-max_chars:], "truncated": len(text) > max_chars}

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            item = self._tasks.get(task_id)
        if not item:
            return {"ok": False, "error": "task_not_found", "task_id": task_id}
        self._terminate(item)
        item["metadata"].update({"status": "cancelled", "finished_at": time.time()})
        self._metadata_path(task_id).write_text(
            json.dumps(item["metadata"], indent=2) + "\n", encoding="utf-8"
        )
        return self.status(task_id)

