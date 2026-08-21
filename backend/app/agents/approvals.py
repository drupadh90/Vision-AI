"""Human-in-the-loop approval for `needs_approval` verdicts.

The Guardian has three outcomes. `allow` and `deny` are decided by the machine;
`needs_approval` is the interesting one — legal-but-consequential actions
(installing packages, sending messages, irreversible external writes) where the
right answer depends on intent only the user has.

Design rules
------------
* **Fail closed.** A request that is never answered is DENIED when the deadline
  passes. Silence must never be read as consent.
* **Bounded wait.** The agent blocks on an `asyncio.Event`, not a poll loop, and
  always with a timeout, so a wave can never hang forever.
* **Concurrency safe.** Several sub-agents may request approval simultaneously;
  each gets its own request id and waiter.
* **Auditable.** Every request and its resolution is recorded, including who
  decided it and how long it took.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..config import Settings, get_settings
from ..guardian.guardian import ProposedAction, Verdict

logger = logging.getLogger("vision.approvals")


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    id: str
    run_id: str
    task_id: str
    tool: str
    payload: str
    reason: str
    safe_alternative: str | None
    violated_principles: list[str]
    requested_at: float
    expires_at: float
    state: ApprovalState = ApprovalState.PENDING
    decided_at: float | None = None
    decided_by: str = ""
    note: str = ""
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "tool": self.tool,
            "payload": self.payload[:2000],
            "reason": self.reason,
            "safe_alternative": self.safe_alternative,
            "violated_principles": self.violated_principles,
            "requested_at": self.requested_at,
            "expires_at": self.expires_at,
            "seconds_remaining": max(0.0, round(self.expires_at - time.time(), 1)),
            "state": self.state.value,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "note": self.note,
        }


@dataclass
class ApprovalOutcome:
    approved: bool
    state: ApprovalState
    note: str = ""
    decided_by: str = ""

    @property
    def message(self) -> str:
        if self.approved:
            return f"Approved by {self.decided_by or 'user'}."
        if self.state is ApprovalState.EXPIRED:
            return "No response before the deadline — denied (fail-closed)."
        if self.state is ApprovalState.CANCELLED:
            return "Run cancelled before a decision was made."
        return f"Denied by {self.decided_by or 'user'}." + (f" {self.note}" if self.note else "")


class ApprovalGate:
    """Registry of pending approvals plus the wait/decide handshake."""

    def __init__(
        self,
        settings: Settings | None = None,
        notifier: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[dict[str, Any]] = []
        self._notifier = notifier
        self._lock = asyncio.Lock()

    def set_notifier(self, notifier: Callable[[dict[str, Any]], Any] | None) -> None:
        """Attach the live event sink (a WebSocket) for the current run."""
        self._notifier = notifier

    async def _notify(self, event: dict[str, Any]) -> None:
        if not self._notifier:
            return
        try:
            result = self._notifier(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # a dead socket must never break the agent
            logger.debug("approval notifier failed", exc_info=True)

    async def request(
        self,
        *,
        action: ProposedAction,
        verdict: Verdict,
        tool: str,
        run_id: str = "",
        task_id: str = "",
    ) -> ApprovalOutcome:
        """Ask the human. Blocks until decided, or denies at the deadline."""
        mode = self.settings.approval_mode

        if mode == "auto_approve":
            logger.warning(
                "approval auto-granted (VISION_APPROVAL_MODE=auto_approve): %s", tool
            )
            self._record(None, ApprovalState.APPROVED, "auto_approve mode", "config")
            return ApprovalOutcome(True, ApprovalState.APPROVED, "auto-approved by config", "config")

        if mode == "auto_deny":
            self._record(None, ApprovalState.DENIED, "auto_deny mode", "config")
            return ApprovalOutcome(False, ApprovalState.DENIED, "auto-denied by config", "config")

        now = time.time()
        req = ApprovalRequest(
            id=uuid.uuid4().hex[:12],
            run_id=run_id,
            task_id=task_id,
            tool=tool,
            payload=action.payload,
            reason=verdict.reason,
            safe_alternative=verdict.safe_alternative,
            violated_principles=verdict.violated_principles,
            requested_at=now,
            expires_at=now + self.settings.approval_timeout_seconds,
        )
        async with self._lock:
            self._pending[req.id] = req

        logger.info("approval requested %s for tool=%s", req.id, tool)
        await self._notify({"type": "approval_required", "ts": now, "request": req.to_dict()})

        try:
            await asyncio.wait_for(
                req._event.wait(), timeout=self.settings.approval_timeout_seconds
            )
        except asyncio.TimeoutError:
            req.state = ApprovalState.EXPIRED
            req.decided_at = time.time()
            logger.warning("approval %s expired — denying (fail-closed)", req.id)

        async with self._lock:
            self._pending.pop(req.id, None)
        self._history.append(req.to_dict())

        await self._notify(
            {
                "type": "approval_resolved",
                "ts": time.time(),
                "request_id": req.id,
                "state": req.state.value,
                "decided_by": req.decided_by,
            }
        )
        return ApprovalOutcome(
            approved=req.state is ApprovalState.APPROVED,
            state=req.state,
            note=req.note,
            decided_by=req.decided_by,
        )

    async def decide(
        self, request_id: str, *, approved: bool, decided_by: str = "user", note: str = ""
    ) -> dict[str, Any] | None:
        """Resolve a pending request. Returns None if it is unknown/already done."""
        async with self._lock:
            req = self._pending.get(request_id)
            if req is None or req.state is not ApprovalState.PENDING:
                return None
            req.state = ApprovalState.APPROVED if approved else ApprovalState.DENIED
            req.decided_at = time.time()
            req.decided_by = decided_by
            req.note = note
            req._event.set()
        logger.info("approval %s -> %s by %s", request_id, req.state.value, decided_by)
        return req.to_dict()

    async def cancel_run(self, run_id: str) -> int:
        """Release every waiter for a run (used when a run is cancelled)."""
        cancelled = 0
        async with self._lock:
            for req in list(self._pending.values()):
                if run_id and req.run_id != run_id:
                    continue
                req.state = ApprovalState.CANCELLED
                req.decided_at = time.time()
                req.decided_by = "system"
                req._event.set()
                cancelled += 1
        return cancelled

    def pending(self, run_id: str | None = None) -> list[dict[str, Any]]:
        items = [
            r.to_dict()
            for r in self._pending.values()
            if run_id is None or r.run_id == run_id
        ]
        return sorted(items, key=lambda r: r["requested_at"])

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._history[-limit:]

    def _record(
        self, req: ApprovalRequest | None, state: ApprovalState, note: str, by: str
    ) -> None:
        self._history.append(
            {
                "id": req.id if req else uuid.uuid4().hex[:12],
                "state": state.value,
                "note": note,
                "decided_by": by,
                "decided_at": time.time(),
            }
        )
