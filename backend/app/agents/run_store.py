"""Durable index of Goal Mode runs.

LangGraph's checkpointer persists graph *state*, but it has no notion of "which
runs exist and what happened to them". This SQLite index provides that view so
the UI can list runs, show status, and offer resume after a restart.

Kept intentionally small: the checkpointer remains the source of truth for
state, while this table is the queryable catalogue pointing at it via thread_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("vision.runs")


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # process died mid-run; resumable
    CANCELLED = "cancelled"


@dataclass
class RunRecord:
    id: str
    goal: str
    user_id: str
    status: RunStatus
    created_at: float
    updated_at: float
    final_answer: str = ""
    error: str = ""
    tasks: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "final_answer": self.final_answer,
            "error": self.error,
            "tasks": self.tasks or [],
            "resumable": self.status
            in (RunStatus.INTERRUPTED, RunStatus.RUNNING),
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    goal          TEXT NOT NULL,
    user_id       TEXT NOT NULL DEFAULT 'default',
    status        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    final_answer  TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    tasks_json    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        logger.info("run store ready at %s", self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("RunStore is not connected")
        return self._conn

    async def create(self, goal: str, user_id: str = "default") -> RunRecord:
        now = time.time()
        record = RunRecord(
            id=uuid.uuid4().hex[:16],
            goal=goal,
            user_id=user_id,
            status=RunStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        conn = self._require()
        async with self._lock:
            await conn.execute(
                "INSERT INTO runs (id, goal, user_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (record.id, goal, user_id, record.status.value, now, now),
            )
            await conn.commit()
        return record

    async def update(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        final_answer: str | None = None,
        error: str | None = None,
        tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        args: list[Any] = [time.time()]
        if status is not None:
            sets.append("status = ?")
            args.append(status.value)
        if final_answer is not None:
            sets.append("final_answer = ?")
            args.append(final_answer)
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        if tasks is not None:
            sets.append("tasks_json = ?")
            args.append(json.dumps(tasks)[:400_000])
        args.append(run_id)

        conn = self._require()
        async with self._lock:
            await conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", args)
            await conn.commit()

    async def get(self, run_id: str) -> RunRecord | None:
        conn = self._require()
        async with conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return _to_record(row) if row else None

    async def list(self, limit: int = 50) -> list[RunRecord]:
        conn = self._require()
        async with conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_to_record(r) for r in rows]

    async def mark_orphans_interrupted(self) -> int:
        """On boot, any run still marked RUNNING was killed by a restart.

        Flag them INTERRUPTED so the UI can offer a resume instead of showing a
        run that is silently dead forever.
        """
        conn = self._require()
        async with self._lock:
            cur = await conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE status = ?",
                (RunStatus.INTERRUPTED.value, time.time(), RunStatus.RUNNING.value),
            )
            await conn.commit()
            count = cur.rowcount or 0
        if count:
            logger.warning("marked %d orphaned run(s) as interrupted (resumable)", count)
        return count


def _to_record(row: aiosqlite.Row) -> RunRecord:
    try:
        tasks = json.loads(row["tasks_json"] or "[]")
    except json.JSONDecodeError:
        tasks = []
    return RunRecord(
        id=row["id"],
        goal=row["goal"],
        user_id=row["user_id"],
        status=RunStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        final_answer=row["final_answer"],
        error=row["error"],
        tasks=tasks,
    )
