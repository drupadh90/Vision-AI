"""Human-in-the-loop approval gate.

The critical property is fail-closed: an unanswered request must DENY, never
drift into an allow. These tests try hard to break that.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.approvals import ApprovalGate, ApprovalState
from app.agents.tools import ToolBelt
from app.config import Settings
from app.core.llm import LLMClient
from app.guardian.guardian import (
    ActionType,
    Decision,
    Guardian,
    ProposedAction,
    Verdict,
)


def _settings(tmp_path, **overrides) -> Settings:
    s = Settings(
        VISION_LLM_PROVIDER="mock",
        VISION_WORKSPACE_DIR=str(tmp_path / "ws"),
        VISION_CHROMA_PATH=str(tmp_path / "chroma"),
        VISION_AUDIO_CACHE=str(tmp_path / "audio"),
        VISION_EXECUTOR="subprocess",
        **overrides,
    )
    s.ensure_dirs()
    return s


def _action() -> ProposedAction:
    return ProposedAction(ActionType.SHELL, "pip install requests", description="add a dep")


def _verdict() -> Verdict:
    return Verdict(
        decision=Decision.NEEDS_APPROVAL,
        reason="Installing packages changes the environment.",
        layer="pattern_filter",
        safe_alternative="Confirm the package list first.",
    )


@pytest.mark.asyncio
async def test_approval_granted_unblocks_action(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell", run_id="r1")
    )
    await asyncio.sleep(0.05)

    pending = gate.pending()
    assert len(pending) == 1
    assert pending[0]["tool"] == "shell"

    assert await gate.decide(pending[0]["id"], approved=True, decided_by="alice")
    outcome = await waiter
    assert outcome.approved is True
    assert outcome.state is ApprovalState.APPROVED
    assert outcome.decided_by == "alice"
    assert gate.pending() == []


@pytest.mark.asyncio
async def test_approval_refused_blocks_action(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell")
    )
    await asyncio.sleep(0.05)
    req_id = gate.pending()[0]["id"]
    await gate.decide(req_id, approved=False, decided_by="bob", note="use uv instead")

    outcome = await waiter
    assert outcome.approved is False
    assert outcome.state is ApprovalState.DENIED
    assert "uv instead" in outcome.note


@pytest.mark.asyncio
async def test_unanswered_approval_fails_closed(tmp_path) -> None:
    """Silence must never be treated as consent."""
    gate = ApprovalGate(_settings(tmp_path, VISION_APPROVAL_TIMEOUT_SECONDS=0.3))
    outcome = await gate.request(action=_action(), verdict=_verdict(), tool="shell")
    assert outcome.approved is False
    assert outcome.state is ApprovalState.EXPIRED
    assert gate.pending() == []


@pytest.mark.asyncio
async def test_deciding_twice_is_rejected(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell")
    )
    await asyncio.sleep(0.05)
    req_id = gate.pending()[0]["id"]
    assert await gate.decide(req_id, approved=True) is not None
    await waiter
    # A replayed or duplicated decision must not resurrect the request.
    assert await gate.decide(req_id, approved=False) is None


@pytest.mark.asyncio
async def test_unknown_request_id_returns_none(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    assert await gate.decide("does-not-exist", approved=True) is None


@pytest.mark.asyncio
async def test_concurrent_approvals_are_independent(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    waiters = [
        asyncio.create_task(
            gate.request(action=_action(), verdict=_verdict(), tool=f"tool{i}", run_id="r1")
        )
        for i in range(3)
    ]
    await asyncio.sleep(0.05)
    pending = gate.pending("r1")
    assert len(pending) == 3
    assert len({p["id"] for p in pending}) == 3

    by_tool = {p["tool"]: p["id"] for p in pending}
    await gate.decide(by_tool["tool0"], approved=True)
    await gate.decide(by_tool["tool1"], approved=False)
    await gate.decide(by_tool["tool2"], approved=True)

    results = await asyncio.gather(*waiters)
    assert [r.approved for r in results] == [True, False, True]


@pytest.mark.asyncio
async def test_cancel_run_releases_waiters(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell", run_id="doomed")
    )
    await asyncio.sleep(0.05)
    assert await gate.cancel_run("doomed") == 1
    outcome = await waiter
    assert outcome.approved is False
    assert outcome.state is ApprovalState.CANCELLED


@pytest.mark.asyncio
async def test_auto_approve_mode(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path, VISION_APPROVAL_MODE="auto_approve"))
    outcome = await gate.request(action=_action(), verdict=_verdict(), tool="shell")
    assert outcome.approved is True


@pytest.mark.asyncio
async def test_auto_deny_mode(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path, VISION_APPROVAL_MODE="auto_deny"))
    outcome = await gate.request(action=_action(), verdict=_verdict(), tool="shell")
    assert outcome.approved is False


@pytest.mark.asyncio
async def test_notifier_receives_request_and_resolution(tmp_path) -> None:
    events: list[dict] = []
    gate = ApprovalGate(_settings(tmp_path), notifier=lambda e: events.append(e))
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell")
    )
    await asyncio.sleep(0.05)
    await gate.decide(gate.pending()[0]["id"], approved=True)
    await waiter
    assert [e["type"] for e in events] == ["approval_required", "approval_resolved"]
    assert events[0]["request"]["safe_alternative"]


@pytest.mark.asyncio
async def test_broken_notifier_does_not_break_approval(tmp_path) -> None:
    def explode(_event):
        raise RuntimeError("socket died")

    gate = ApprovalGate(_settings(tmp_path), notifier=explode)
    waiter = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell")
    )
    await asyncio.sleep(0.05)
    await gate.decide(gate.pending()[0]["id"], approved=True)
    assert (await waiter).approved is True


# ------------------------------------------------------------- toolbelt wiring

@pytest.mark.asyncio
async def test_toolbelt_waits_for_approval_then_executes(tmp_path) -> None:
    s = _settings(tmp_path)
    gate = ApprovalGate(s)
    belt = ToolBelt(Guardian(LLMClient(s), s), settings=s, approvals=gate)

    # `pip install` is classified needs_approval by the pattern filter.
    call = asyncio.create_task(belt.call("shell", command="pip install requests"))
    await asyncio.sleep(0.1)
    pending = gate.pending()
    assert len(pending) == 1, "tool did not raise an approval request"

    await gate.decide(pending[0]["id"], approved=True, decided_by="alice")
    result = await call
    assert result.blocked is False
    assert result.approval and result.approval["state"] == "approved"


@pytest.mark.asyncio
async def test_toolbelt_blocks_when_approval_refused(tmp_path) -> None:
    s = _settings(tmp_path)
    gate = ApprovalGate(s)
    belt = ToolBelt(Guardian(LLMClient(s), s), settings=s, approvals=gate)

    call = asyncio.create_task(belt.call("shell", command="pip install requests"))
    await asyncio.sleep(0.1)
    await gate.decide(gate.pending()[0]["id"], approved=False, decided_by="bob")
    result = await call
    assert result.blocked is True
    assert result.ok is False
    assert "approval refused" in result.output.lower()


@pytest.mark.asyncio
async def test_toolbelt_without_gate_blocks_consequential_action(tmp_path) -> None:
    """No approval channel must mean refusal, not silent execution."""
    s = _settings(tmp_path)
    belt = ToolBelt(Guardian(LLMClient(s), s), settings=s, approvals=None)
    result = await belt.call("shell", command="pip install requests")
    assert result.blocked is True
    assert "no approval channel" in result.output.lower()


@pytest.mark.asyncio
async def test_hard_denied_action_never_reaches_approval(tmp_path) -> None:
    """Constitutional violations are not negotiable by a human click."""
    s = _settings(tmp_path)
    gate = ApprovalGate(s)
    belt = ToolBelt(Guardian(LLMClient(s), s), settings=s, approvals=gate)
    result = await belt.call("shell", command="rm -rf /")
    assert result.blocked is True
    assert gate.pending() == [], "a hard-denied action must not ask for approval"


@pytest.mark.asyncio
async def test_cancel_only_affects_the_named_run(tmp_path) -> None:
    """Disconnecting one client must not free another run's approvals."""
    gate = ApprovalGate(_settings(tmp_path))
    a = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell", run_id="run-a")
    )
    b = asyncio.create_task(
        gate.request(action=_action(), verdict=_verdict(), tool="shell", run_id="run-b")
    )
    await asyncio.sleep(0.05)

    assert await gate.cancel_run("run-a") == 1
    outcome_a = await a
    assert outcome_a.state is ApprovalState.CANCELLED
    assert len(gate.pending("run-b")) == 1, "unrelated run was disturbed"

    await gate.decide(gate.pending("run-b")[0]["id"], approved=True)
    assert (await b).approved is True


@pytest.mark.asyncio
async def test_cancel_run_with_no_pending_is_safe(tmp_path) -> None:
    gate = ApprovalGate(_settings(tmp_path))
    assert await gate.cancel_run("nothing-here") == 0
