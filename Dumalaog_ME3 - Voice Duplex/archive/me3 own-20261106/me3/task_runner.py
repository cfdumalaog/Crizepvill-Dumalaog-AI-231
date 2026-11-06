"""Bounded background-process support for the ME3 voice tool registry.

The voice agent must be able to start a web-app build or test without blocking
the microphone loop.  This module deliberately keeps process state inside the
ME3 project and records only command/output metadata; it never receives or
persists API credentials.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path


_SECRET_TEXT = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|secret|authorization|bearer)\b\s*[:=]\s*)([^\s,;]+)"
)


def _safe_runtime_text(value: str, max_chars: int = 700) -> str:
    """Keep terminal telemetry useful without copying credentials to the UI."""
    text = str(value or "")
    text = _SECRET_TEXT.sub(r"\1<redacted>", text)
    text = re.sub(r"(?i)\bBearer\s+[^\s]+", "Bearer <redacted>", text)
    text = text.replace("\x00", "")
    return text[-max_chars:]


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

    def snapshot(self, max_tasks: int = 12, tail_lines: int = 5) -> list[dict]:
        """Return safe live/background process telemetry for the Runtime UI.

        The voice UI receives the PID, bounded command, status, elapsed time,
        and short redacted output tails. It never receives environment values,
        private log paths, or unrestricted process-environment data.
        """
        ids: set[str] = set()
        with self._lock:
            ids.update(self._tasks)
        for path in sorted(self.task_root.glob("task-*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            ids.add(path.stem)
            if len(ids) >= max(1, int(max_tasks)):
                break
        rows: list[dict] = []
        for task_id in list(ids)[:max(1, int(max_tasks))]:
            status = self.status(task_id)
            if not status.get("ok"):
                continue
            with self._lock:
                item = self._tasks.get(task_id)
                raw_meta = dict(item["metadata"]) if item else None
            if raw_meta is None:
                try:
                    raw_meta = json.loads(
                        self._metadata_path(task_id).read_text(encoding="utf-8"))
                except Exception:
                    raw_meta = {}
            argv = raw_meta.get("argv") or status.get("argv") or []
            try:
                command = shlex.join([str(x) for x in argv])
            except Exception:
                command = " ".join(str(x) for x in argv)
            cwd = Path(str(raw_meta.get("cwd", "")))
            try:
                cwd_display = cwd.resolve().relative_to(self.project_root).as_posix()
            except (OSError, ValueError):
                cwd_display = "project"
            started = float(raw_meta.get("started_at", time.time()) or time.time())
            finished = raw_meta.get("finished_at")
            end = float(finished or time.time())
            stdout_path = Path(str(raw_meta.get("stdout_path", "")))
            stderr_path = Path(str(raw_meta.get("stderr_path", "")))

            def tail(path: Path) -> str:
                if not path.is_file():
                    return ""
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    return "\n".join(_safe_runtime_text(x) for x in lines[-tail_lines:])
                except OSError:
                    return ""

            mtimes = [p.stat().st_mtime for p in (stdout_path, stderr_path)
                      if p.is_file()]
            rows.append({
                "task_id": task_id,
                "pid": status.get("pid"),
                "status": status.get("status", "unknown"),
                "returncode": status.get("returncode"),
                "command": _safe_runtime_text(command, 300),
                "cwd": cwd_display,
                "elapsed_s": round(max(0.0, end - started), 1),
                "timeout_s": status.get("timeout_s"),
                "stdout_tail": tail(stdout_path),
                "stderr_tail": tail(stderr_path),
                "updated_at": round(max([started, *mtimes,
                                         float(finished or 0.0)]), 3),
            })
        rows.sort(key=lambda row: row.get("updated_at", 0), reverse=True)
        return rows[:max(1, int(max_tasks))]

