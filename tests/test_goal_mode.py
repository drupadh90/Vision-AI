"""Goal Mode: DAG validation, scheduling, concurrency and manager review."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agents.goal_graph import (
    GoalModeEngine,
    Task,
    TaskStatus,
    validate_plan,
)
from app.config import Settings
from app.core.llm import LLMClient
from app.guardian.guardian import Guardian
from app.memory.digital_twin import DigitalTwinMemory
from app.skills.skill_store import SkillLibrary


def _t(tid: str, deps: list[str] | None = None) -> Task:
    return Task(id=tid, title=tid, description="d", depends_on=deps or [])


def test_valid_dag_passes() -> None:
    assert validate_plan([_t("t1"), _t("t2", ["t1"]), _t("t3", ["t1", "t2"])], 12) == []


def test_cycle_is_detected() -> None:
    problems = validate_plan([_t("t1", ["t3"]), _t("t2", ["t1"]), _t("t3", ["t2"])], 12)
    assert "dependency cycle detected" in problems


def test_self_dependency_is_detected() -> None:
    assert any("depends on itself" in p for p in validate_plan([_t("t1", ["t1"])], 12))


def test_dangling_dependency_is_detected() -> None:
    assert any("unknown task" in p for p in validate_plan([_t("t1", ["nope"])], 12))


def test_duplicate_ids_detected() -> None:
    assert "duplicate task ids" in validate_plan([_t("t1"), _t("t1")], 12)


def test_task_budget_enforced() -> None:
    assert any("too many tasks" in p for p in validate_plan([_t(f"t{i}") for i in range(15)], 12))


def test_ready_set_respects_dependencies() -> None:
    tasks = [_t("t1"), _t("t2", ["t1"]), _t("t3", ["t1"])]
    assert [t.id for t in GoalModeEngine._ready(tasks)] == ["t1"]
    tasks[0].status = TaskStatus.DONE
    assert {t.id for t in GoalModeEngine._ready(tasks)} == {"t2", "t3"}


def test_failed_upstream_blocks_downstream() -> None:
    tasks = [_t("t1"), _t("t2", ["t1"])]
    tasks[0].status = TaskStatus.FAILED
    GoalModeEngine._ready(tasks)
    assert tasks[1].status is TaskStatus.BLOCKED


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(
        VISION_LLM_PROVIDER="mock",
        VISION_CHROMA_PATH=str(tmp_path / "chroma"),
        VISION_WORKSPACE_DIR=str(tmp_path / "ws"),
        VISION_AUDIO_CACHE=str(tmp_path / "audio"),
        VISION_EXECUTOR="disabled",
    )
    s.ensure_dirs()
    return s


def _engine(s: Settings, llm: LLMClient | None = None) -> GoalModeEngine:
    client = llm or LLMClient(s)
    return GoalModeEngine(
        llm=client,
        memory=DigitalTwinMemory(s),
        skills=SkillLibrary(s),
        guardian=Guardian(client, s),
        settings=s,
    )


@pytest.mark.asyncio
async def test_goal_run_completes_end_to_end(settings: Settings) -> None:
    state = await _engine(settings).run("Write a haiku about databases")
    assert state["done"] is True
    assert state["final_answer"]
    assert state["tasks"]
    assert all(t["status"] == "done" for t in state["tasks"])
    kinds = [e["type"] for e in state["events"]]
    assert "plan_ready" in kinds
    assert "goal_completed" in kinds
    assert "reflection_written" in kinds


@pytest.mark.asyncio
async def test_reflection_persists_memories(settings: Settings) -> None:
    memory = DigitalTwinMemory(settings)
    before = (await memory.stats())["total_memories"]
    await _engine(settings).run("Summarise the CAP theorem")
    assert (await memory.stats())["total_memories"] > before


@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently(settings: Settings) -> None:
    """A fan-out plan must execute in parallel, not serially."""
    plan = {
        "strategy": "fan out",
        "tasks": [
            {"id": "t1", "title": "seed", "description": "d", "agent": "research", "depends_on": []},
            {"id": "t2", "title": "a", "description": "d", "agent": "coder", "depends_on": ["t1"]},
            {"id": "t3", "title": "b", "description": "d", "agent": "writer", "depends_on": ["t1"]},
            {"id": "t4", "title": "c", "description": "d", "agent": "analyst", "depends_on": ["t1"]},
        ],
    }
    delay = 0.4

    class SlowLLM(LLMClient):
        async def complete_json(self, messages, **kw):  # type: ignore[override]
            if kw.get("role") == "planner":
                text = " ".join(m.content for m in messages)
                if "reviewing a completed sub-task" in text:
                    return {"accept": True, "quality": 0.9, "feedback": "", "reassign_to": None}
                return plan
            return await super().complete_json(messages, **kw)

        async def complete(self, messages, **kw):  # type: ignore[override]
            if kw.get("role") == "worker":
                await asyncio.sleep(delay)
            return await super().complete(messages, **kw)

    started = time.monotonic()
    state = await _engine(settings, SlowLLM(settings)).run("fan out test")
    elapsed = time.monotonic() - started

    waves = [e for e in state["events"] if e["type"] == "wave_started"]
    assert any(len(w["tasks"]) == 3 for w in waves), "no parallel wave scheduled"
    # Serial would cost 4*delay; parallel should be ~2*delay.
    assert elapsed < delay * 3.5, f"tasks appear to have run serially ({elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_manager_rejects_and_retries_weak_work(settings: Settings) -> None:
    calls = {"reviews": 0}

    class PickyLLM(LLMClient):
        async def complete_json(self, messages, **kw):  # type: ignore[override]
            if kw.get("role") == "planner":
                text = " ".join(m.content for m in messages)
                if "reviewing a completed sub-task" in text:
                    calls["reviews"] += 1
                    if calls["reviews"] == 1:  # reject once, then accept
                        return {
                            "accept": False,
                            "quality": 0.2,
                            "feedback": "Too shallow, add detail.",
                            "reassign_to": "writer",
                        }
                    return {"accept": True, "quality": 0.9, "feedback": "", "reassign_to": None}
                return {
                    "strategy": "single",
                    "tasks": [
                        {"id": "t1", "title": "only task", "description": "d",
                         "agent": "research", "depends_on": []}
                    ],
                }
            return await super().complete_json(messages, **kw)

    state = await _engine(settings, PickyLLM(settings)).run("test rejection")
    task = state["tasks"][0]
    assert task["attempts"] >= 2, "rejected task was not retried"
    assert task["agent"] == "writer", "task was not re-assigned to another specialist"
    assert any(e["type"] == "task_rejected" for e in state["events"])
    assert any(e["type"] == "task_reassigned" for e in state["events"])


@pytest.mark.asyncio
async def test_malformed_plan_is_repaired_not_fatal(settings: Settings) -> None:
    class BadPlanLLM(LLMClient):
        async def complete_json(self, messages, **kw):  # type: ignore[override]
            if kw.get("role") == "planner":
                text = " ".join(m.content for m in messages)
                if "reviewing a completed sub-task" in text:
                    return {"accept": True, "quality": 0.8, "feedback": "", "reassign_to": None}
                return {  # a cycle: t1 <-> t2
                    "strategy": "broken",
                    "tasks": [
                        {"id": "t1", "title": "a", "description": "d",
                         "agent": "research", "depends_on": ["t2"]},
                        {"id": "t2", "title": "b", "description": "d",
                         "agent": "writer", "depends_on": ["t1"]},
                    ],
                }
            return await super().complete_json(messages, **kw)

    state = await _engine(settings, BadPlanLLM(settings)).run("broken plan")
    assert any(e["type"] == "plan_repaired" for e in state["events"])
    assert state["final_answer"], "engine should still deliver an answer"


@pytest.mark.asyncio
async def test_reflection_failure_does_not_lose_deliverable(settings: Settings, monkeypatch) -> None:
    """A memory-layer outage must never discard a completed goal."""
    engine = _engine(settings)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(engine.reflection, "reflect", boom)
    state = await engine.run("survive a memory outage")
    assert state["final_answer"], "deliverable lost when reflection failed"
    assert state["done"] is True
