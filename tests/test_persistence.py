"""Durable Goal Mode runs: checkpointing, crash recovery and the run index."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.goal_graph import GoalModeEngine
from app.agents.run_store import RunStatus, RunStore
from app.config import Settings
from app.core.llm import LLMClient
from app.guardian.guardian import Guardian
from app.memory.digital_twin import DigitalTwinMemory
from app.skills.skill_store import SkillLibrary


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(
        VISION_LLM_PROVIDER="mock",
        VISION_CHROMA_PATH=str(tmp_path / "chroma"),
        VISION_WORKSPACE_DIR=str(tmp_path / "ws"),
        VISION_AUDIO_CACHE=str(tmp_path / "audio"),
        VISION_CHECKPOINT_PATH=str(tmp_path / "cp.sqlite"),
        VISION_EXECUTOR="disabled",
    )
    s.ensure_dirs()
    return s


async def _make_saver(settings: Settings):
    conn = await aiosqlite.connect(str(settings.checkpoint_path))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return conn, saver


def _engine(settings: Settings, saver, llm: LLMClient | None = None) -> GoalModeEngine:
    client = llm or LLMClient(settings)
    return GoalModeEngine(
        llm=client,
        memory=DigitalTwinMemory(settings),
        skills=SkillLibrary(settings),
        guardian=Guardian(client, settings),
        settings=settings,
        checkpointer=saver,
    )


# ------------------------------------------------------------------ run store

@pytest.mark.asyncio
async def test_run_store_lifecycle(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    await store.connect()
    try:
        record = await store.create("build a thing", "alice")
        assert record.status is RunStatus.RUNNING

        await store.update(
            record.id, status=RunStatus.COMPLETED, final_answer="done",
            tasks=[{"id": "t1", "status": "done"}],
        )
        fetched = await store.get(record.id)
        assert fetched is not None
        assert fetched.status is RunStatus.COMPLETED
        assert fetched.final_answer == "done"
        assert fetched.tasks and fetched.tasks[0]["id"] == "t1"

        assert len(await store.list()) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_run_store_survives_reconnect(tmp_path) -> None:
    path = tmp_path / "runs.sqlite"
    store = RunStore(path)
    await store.connect()
    record = await store.create("persist me")
    await store.close()

    reopened = RunStore(path)
    await reopened.connect()
    try:
        again = await reopened.get(record.id)
        assert again is not None and again.goal == "persist me"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_orphaned_runs_become_resumable_on_boot(tmp_path) -> None:
    """A run left RUNNING by a crash must be flagged, not silently stuck."""
    path = tmp_path / "runs.sqlite"
    store = RunStore(path)
    await store.connect()
    record = await store.create("killed mid-flight")
    await store.close()  # simulate the process dying

    rebooted = RunStore(path)
    await rebooted.connect()
    try:
        assert await rebooted.mark_orphans_interrupted() == 1
        after = await rebooted.get(record.id)
        assert after is not None
        assert after.status is RunStatus.INTERRUPTED
        assert after.to_dict()["resumable"] is True
    finally:
        await rebooted.close()


@pytest.mark.asyncio
async def test_completed_runs_are_not_marked_interrupted(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    await store.connect()
    try:
        record = await store.create("finished")
        await store.update(record.id, status=RunStatus.COMPLETED)
        assert await store.mark_orphans_interrupted() == 0
        after = await store.get(record.id)
        assert after is not None and after.status is RunStatus.COMPLETED
    finally:
        await store.close()


# --------------------------------------------------------------- checkpointing

@pytest.mark.asyncio
async def test_checkpoint_state_is_persisted(settings: Settings) -> None:
    conn, saver = await _make_saver(settings)
    try:
        engine = _engine(settings, saver)
        state = await engine.run("checkpoint me", run_id="run-abc")
        assert state["final_answer"]

        snapshot = await engine.get_state("run-abc")
        assert snapshot is not None
        assert snapshot["values"]["goal"] == "checkpoint me"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_interrupted_run_resumes_after_restart(settings: Settings) -> None:
    """The real test: kill a run mid-flight, rebuild everything, resume it."""
    fail_switch = {"armed": True}

    class CrashingLLM(LLMClient):
        async def complete(self, messages, **kw):  # type: ignore[override]
            # Blow up during synthesis, after planning and execution succeeded.
            text = " ".join(m.content for m in messages)
            if fail_switch["armed"] and "producing the final deliverable" in text.lower():
                raise RuntimeError("simulated crash during synthesis")
            return await super().complete(messages, **kw)

    conn, saver = await _make_saver(settings)
    try:
        engine = _engine(settings, saver, CrashingLLM(settings))
        # Synthesis catches LLM errors internally, so force a hard node failure.
        original = engine._node_synthesize

        async def exploding_synthesize(state):
            if fail_switch["armed"]:
                raise RuntimeError("simulated crash during synthesis")
            return await original(state)

        engine._node_synthesize = exploding_synthesize  # type: ignore[method-assign]
        engine.graph = engine._build_graph()

        with pytest.raises(RuntimeError):
            await engine.run("survive a crash", run_id="run-crash")

        # State up to the crash must be on disk.
        snapshot = await engine.get_state("run-crash")
        assert snapshot is not None
        assert snapshot["values"]["tasks"], "no work was checkpointed"
        assert "synthesize" in snapshot["next"], f"expected to be paused at synthesize, got {snapshot['next']}"
    finally:
        await conn.close()

    # --- process "restart": brand-new connection, saver and engine ---------
    fail_switch["armed"] = False
    conn2, saver2 = await _make_saver(settings)
    try:
        fresh_engine = _engine(settings, saver2)
        resumed = await fresh_engine.resume("run-crash")
        assert resumed["final_answer"], "resume produced no deliverable"
        assert resumed["done"] is True
        # It resumed rather than restarting: planning was not redone.
        assert resumed["goal"] == "survive a crash"
    finally:
        await conn2.close()


@pytest.mark.asyncio
async def test_resume_without_checkpointer_raises(settings: Settings) -> None:
    engine = _engine(settings, None)
    with pytest.raises(RuntimeError, match="no checkpointer"):
        await engine.resume("whatever")


@pytest.mark.asyncio
async def test_resume_unknown_run_raises(settings: Settings) -> None:
    conn, saver = await _make_saver(settings)
    try:
        engine = _engine(settings, saver)
        with pytest.raises(ValueError, match="No checkpoint"):
            await engine.resume("never-existed")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_runs_are_isolated_by_thread_id(settings: Settings) -> None:
    conn, saver = await _make_saver(settings)
    try:
        engine = _engine(settings, saver)
        await engine.run("first goal", run_id="run-1")
        await engine.run("second goal", run_id="run-2")

        s1 = await engine.get_state("run-1")
        s2 = await engine.get_state("run-2")
        assert s1 and s2
        assert s1["values"]["goal"] == "first goal"
        assert s2["values"]["goal"] == "second goal"
    finally:
        await conn.close()
