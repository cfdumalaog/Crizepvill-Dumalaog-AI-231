"""Persistent timer, reminder, and alarm queue for the ME3 Pi service.

Scheduled actions survive an ME3 restart.  When a due item is reached, the
service queues a spoken announcement and keeps a notification pending until a
connected Assistant UI acknowledges that it displayed the alert.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


def default_db_path() -> Path:
    configured = os.getenv("ME3_ACTION_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "ME3"
    else:
        root = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "me3"
    return root / "scheduled_actions.sqlite3"


class ScheduledActionStore:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                self.path.parent.chmod(0o700)
            except OSError:
                pass
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_actions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    due_at REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    due_marked_at REAL,
                    acknowledged_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_due_state "
                         "ON scheduled_actions(state, due_at)")
        if os.name != "nt":
            try:
                self.path.chmod(0o600)
            except OSError:
                pass

    def create(self, kind: str, payload: dict, due_at: float) -> dict:
        if kind not in {"timer", "reminder", "alarm"}:
            raise ValueError("unsupported scheduled action")
        action_id = f"{kind[0]}-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scheduled_actions(id,kind,payload,due_at,state,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (action_id, kind, json.dumps(payload, ensure_ascii=False),
                 float(due_at), "pending", now),
            )
        return self.get(action_id)

    @staticmethod
    def _public(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "kind": row["kind"],
            "payload": json.loads(row["payload"]),
            "due_at": float(row["due_at"]), "state": row["state"],
            "created_at": float(row["created_at"]),
            "due_marked_at": row["due_marked_at"],
        }

    def get(self, action_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scheduled_actions WHERE id=?",
                               (str(action_id),)).fetchone()
        return self._public(row) if row else None

    def list(self, kind: str | None = None, include_finished: bool = False) -> list[dict]:
        query = "SELECT * FROM scheduled_actions"
        clauses: list[str] = []
        args: list[object] = []
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        if not include_finished:
            clauses.append("state NOT IN ('completed','cancelled','delivered')")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY due_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [self._public(row) for row in rows]

    def mark_due(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else float(now)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE scheduled_actions SET state='due', due_marked_at=? "
                "WHERE state='pending' AND due_at<=?", (now, now))
            rows = conn.execute(
                "SELECT * FROM scheduled_actions WHERE state='due' "
                "ORDER BY due_at"
            ).fetchall()
        return [self._public(row) for row in rows]

    def pending_notifications(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_actions WHERE state='due' "
                "ORDER BY due_at LIMIT ?", (max(1, min(int(limit), 100)),)
            ).fetchall()
        return [self._public(row) for row in rows]

    def acknowledge(self, action_id: str) -> dict:
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE scheduled_actions SET state='delivered', acknowledged_at=? "
                "WHERE id=? AND state='due'", (time.time(), str(action_id)))
        return {"ok": result.rowcount == 1, "id": str(action_id),
                "error": None if result.rowcount == 1 else "action_not_pending"}

    def finish(self, action_id: str, state: str) -> dict:
        if state not in {"completed", "cancelled"}:
            return {"ok": False, "error": "invalid_state"}
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE scheduled_actions SET state=? WHERE id=? "
                "AND state IN ('pending','due','delivered')",
                (state, str(action_id)))
        return {"ok": result.rowcount == 1, "id": str(action_id),
                "error": None if result.rowcount == 1 else "action_not_found_or_finished"}


class ScheduledActionRunner:
    def __init__(self, store: ScheduledActionStore,
                 on_due: Callable[[dict], None] | None = None,
                 poll_interval_s: float = 1.0):
        self.store = store
        self.on_due = on_due
        self.poll_interval_s = max(0.25, min(float(poll_interval_s), 10.0))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._announced: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="me3-scheduled-actions")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for item in self.store.mark_due():
                    action_id = item["id"]
                    if action_id in self._announced:
                        continue
                    self._announced.add(action_id)
                    if self.on_due is not None:
                        try:
                            self.on_due(item)
                        except Exception:
                            # Keep the durable due notification available to
                            # the UI even if speech output is temporarily down.
                            pass
            except Exception:
                pass
            self._stop.wait(self.poll_interval_s)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread and thread.is_alive():
            thread.join(timeout=2.0)


_DEFAULT_STORE: ScheduledActionStore | None = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> ScheduledActionStore:
    global _DEFAULT_STORE
    with _DEFAULT_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = ScheduledActionStore()
        return _DEFAULT_STORE


def due_announcement(item: dict) -> str:
    payload = item.get("payload") or {}
    kind = item.get("kind")
    if kind == "timer":
        label = str(payload.get("label") or "").strip()
        return f"Your timer{': ' + label if label else ''} is finished."
    if kind == "alarm":
        label = str(payload.get("label") or "your alarm").strip()
        return f"Alarm. {label}."
    reminder = str(payload.get("text") or "your reminder").strip()
    return f"Reminder: {reminder}."
