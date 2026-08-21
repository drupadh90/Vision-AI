"""Vision AI — FastAPI entry point.

Wires the five modules into one service:

  /api/goal      Goal Mode (LangGraph DAG + sub-agent spawning)  [+ WebSocket]
  /api/skills    YouTube Skill Acquisition + activation
  /api/memory    Digital Twin memory + reflection loop
  /api/guardian  Constitution + pre-execution filter
  /api/intuition Proactive suggestions
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .agents.approvals import ApprovalGate
from .agents.goal_graph import GoalModeEngine
from .agents.intuition import IntuitionEngine
from .agents.run_store import RunStatus, RunStore
from .agents.tools import ToolBelt
from .config import get_settings
from .core.executor import build_executor, safe_workspace_path
from .core.llm import LLMClient
from .guardian.constitution import CONSTITUTION, constitution_text
from .guardian.guardian import ActionType, Guardian, ProposedAction
from .memory.digital_twin import DigitalTwinMemory, MemoryItem, MemoryKind, ReflectionLoop
from .schemas import (
    ApprovalDecisionRequest,
    GoalRequest,
    GoalResponse,
    GuardianCheckRequest,
    IntuitionRequest,
    LearnRequest,
    MemoryQueryRequest,
    MemoryWriteRequest,
    ReflectRequest,
    SkillSearchRequest,
    TextSkillRequest,
)
from .skills.skill_store import SkillLibrary
from .skills.youtube_ingest import IngestError

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
)
logger = logging.getLogger("vision.main")


@dataclass
class Runtime:
    llm: LLMClient
    memory: DigitalTwinMemory
    skills: SkillLibrary
    guardian: Guardian
    toolbelt: ToolBelt
    intuition: IntuitionEngine
    reflection: ReflectionLoop
    approvals: ApprovalGate
    runs: RunStore
    checkpointer: Any = None

    def engine(self, event_sink: Any = None) -> GoalModeEngine:
        return GoalModeEngine(
            llm=self.llm, memory=self.memory, skills=self.skills,
            guardian=self.guardian, toolbelt=self.toolbelt,
            settings=settings, event_sink=event_sink,
            checkpointer=self.checkpointer, approvals=self.approvals,
        )


rt: Runtime | None = None


def runtime() -> Runtime:
    if rt is None:  # pragma: no cover
        raise HTTPException(503, "Runtime not initialised")
    return rt


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rt
    settings.ensure_dirs()
    llm = LLMClient(settings)
    memory = DigitalTwinMemory(settings)
    skills = SkillLibrary(settings)
    guardian = Guardian(llm, settings)
    approvals = ApprovalGate(settings)
    toolbelt = ToolBelt(guardian, build_executor(settings), settings, approvals=approvals)

    # --- durable state -----------------------------------------------------
    checkpointer = None
    checkpoint_conn = None
    if settings.checkpoint_enabled:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpoint_conn = await aiosqlite.connect(str(settings.checkpoint_path))
        checkpointer = AsyncSqliteSaver(checkpoint_conn)
        await checkpointer.setup()
        logger.info("checkpointer ready at %s", settings.checkpoint_path)

    runs = RunStore(settings.checkpoint_path.with_name("runs.sqlite"))
    await runs.connect()
    orphans = await runs.mark_orphans_interrupted()

    rt = Runtime(
        llm=llm, memory=memory, skills=skills, guardian=guardian, toolbelt=toolbelt,
        intuition=IntuitionEngine(llm, memory, skills, settings),
        reflection=ReflectionLoop(memory, llm),
        approvals=approvals, runs=runs, checkpointer=checkpointer,
    )
    logger.info(
        "Vision AI online | llm=%s embeddings=%s executor=%s guardian=%s "
        "checkpoint=%s approvals=%s",
        llm.provider_name, settings.effective_embedding_provider(),
        settings.executor, settings.guardian_mode if settings.guardian_enabled else "off",
        "on" if checkpointer else "off", settings.approval_mode,
    )
    if llm.is_mock:
        logger.warning("Running with the MOCK LLM — set OPENAI_API_KEY for real reasoning.")
    if orphans:
        logger.warning("%d run(s) interrupted by a previous shutdown are resumable.", orphans)

    yield

    await llm.aclose()
    await runs.close()
    if checkpoint_conn is not None:
        await checkpoint_conn.close()


app = FastAPI(
    title="Vision AI",
    version="0.1.0",
    description="Autonomous agent with a Digital Twin, YouTube skill acquisition, "
                "Goal Mode sub-agents and constitutional guardrails.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    r = runtime()
    return {
        "status": "ok",
        "llm_provider": r.llm.provider_name,
        "mock_mode": r.llm.is_mock,
        "embedding_provider": settings.effective_embedding_provider(),
        "executor": settings.executor,
        "guardian": {
            "enabled": settings.guardian_enabled,
            "mode": settings.guardian_mode,
            "llm_review": settings.guardian_llm_review,
        },
        "memory": await r.memory.stats(),
        "skills": await r.skills.stats(),
    }


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "max_subtasks": settings.max_subtasks,
        "max_concurrency": settings.max_concurrency,
        "max_task_retries": settings.max_task_retries,
        "proactive_enabled": settings.proactive_enabled,
        "transcriber": settings.transcriber,
        "agents": ["research", "coder", "writer", "analyst", "critic", "generalist"],
        "tools": runtime().toolbelt.available(),
    }


# ---------------------------------------------------------------------------
# Goal Mode
# ---------------------------------------------------------------------------

@app.post("/api/goal", response_model=GoalResponse)
async def run_goal(req: GoalRequest) -> GoalResponse:
    r = runtime()
    record = await r.runs.create(req.goal, req.user_id)
    try:
        state = await r.engine().run(req.goal, req.user_id, run_id=record.id)
    except Exception as exc:
        await r.runs.update(record.id, status=RunStatus.FAILED, error=str(exc))
        raise HTTPException(500, f"Goal run failed: {exc}") from exc

    await r.runs.update(
        record.id,
        status=RunStatus.FAILED if state.get("error") else RunStatus.COMPLETED,
        final_answer=state.get("final_answer", ""),
        error=state.get("error", ""),
        tasks=state.get("tasks", []),
    )
    return GoalResponse(
        run_id=record.id,
        goal=req.goal,
        strategy=state.get("strategy", ""),
        final_answer=state.get("final_answer", ""),
        tasks=state.get("tasks", []),
        events=state.get("events", []),
        activated_skills=state.get("activated_skills", []),
        reflection=state.get("reflection", {}),
        error=state.get("error", ""),
    )


@app.get("/api/runs")
async def list_runs(limit: int = 30) -> dict[str, Any]:
    records = await runtime().runs.list(limit)
    return {"runs": [r.to_dict() for r in records]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    r = runtime()
    record = await r.runs.get(run_id)
    if record is None:
        raise HTTPException(404, "Run not found")
    payload = record.to_dict()
    snapshot = await r.engine().get_state(run_id)
    payload["checkpoint"] = (
        {"next": snapshot["next"], "has_state": True} if snapshot else {"has_state": False}
    )
    return payload


@app.post("/api/runs/{run_id}/resume", response_model=GoalResponse)
async def resume_run(run_id: str) -> GoalResponse:
    """Continue a run that a restart or crash left unfinished."""
    r = runtime()
    record = await r.runs.get(run_id)
    if record is None:
        raise HTTPException(404, "Run not found")
    if record.status in (RunStatus.COMPLETED, RunStatus.CANCELLED):
        raise HTTPException(409, f"Run is {record.status.value}; nothing to resume.")
    if r.checkpointer is None:
        raise HTTPException(400, "Checkpointing is disabled; runs cannot be resumed.")

    await r.runs.update(run_id, status=RunStatus.RUNNING)
    try:
        state = await r.engine().resume(run_id)
    except ValueError as exc:
        await r.runs.update(run_id, status=RunStatus.FAILED, error=str(exc))
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        await r.runs.update(run_id, status=RunStatus.FAILED, error=str(exc))
        raise HTTPException(500, f"Resume failed: {exc}") from exc

    await r.runs.update(
        run_id,
        status=RunStatus.FAILED if state.get("error") else RunStatus.COMPLETED,
        final_answer=state.get("final_answer", ""),
        error=state.get("error", ""),
        tasks=state.get("tasks", []),
    )
    return GoalResponse(
        run_id=run_id,
        goal=record.goal,
        strategy=state.get("strategy", ""),
        final_answer=state.get("final_answer", ""),
        tasks=state.get("tasks", []),
        events=state.get("events", []),
        activated_skills=state.get("activated_skills", []),
        reflection=state.get("reflection", {}),
        error=state.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Human-in-the-loop approvals
# ---------------------------------------------------------------------------

@app.get("/api/approvals")
async def list_approvals(run_id: str | None = None) -> dict[str, Any]:
    gate = runtime().approvals
    return {"pending": gate.pending(run_id), "history": gate.history(20)}


@app.post("/api/approvals/{request_id}")
async def decide_approval(request_id: str, req: ApprovalDecisionRequest) -> dict[str, Any]:
    resolved = await runtime().approvals.decide(
        request_id, approved=req.approved, decided_by=req.decided_by, note=req.note
    )
    if resolved is None:
        raise HTTPException(
            404, "No such pending approval (it may have expired or been decided already)."
        )
    return resolved


@app.websocket("/ws/goal")
async def goal_socket(ws: WebSocket) -> None:
    """Stream the agent graph live: every plan, task, verdict and reflection."""
    await ws.accept()
    try:
        payload = json.loads(await ws.receive_text())
        goal = (payload.get("goal") or "").strip()
        user_id = payload.get("user_id", "default")
        if not goal:
            await ws.send_json({"type": "error", "message": "A goal is required."})
            return

        r = runtime()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def sink(event: dict[str, Any]) -> None:
            await queue.put(event)

        # Approval prompts must reach this socket too, so the user can answer
        # them live instead of the agent stalling until the deadline.
        r.approvals.set_notifier(sink)

        resume_id = (payload.get("resume_run_id") or "").strip()
        if resume_id:
            record = await r.runs.get(resume_id)
            if record is None:
                await ws.send_json({"type": "error", "message": "Run not found."})
                return
            run_id, goal = resume_id, record.goal
            await r.runs.update(run_id, status=RunStatus.RUNNING)
            coro = r.engine(event_sink=sink).resume(run_id)
        else:
            record = await r.runs.create(goal, user_id)
            run_id = record.id
            coro = r.engine(event_sink=sink).run(goal, user_id, run_id=run_id)

        await ws.send_json({"type": "run_started", "run_id": run_id, "goal": goal})

        # Accept decisions from the client while the graph runs.
        async def receive_decisions() -> None:
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "approval_decision":
                        await r.approvals.decide(
                            str(msg.get("request_id", "")),
                            approved=bool(msg.get("approved")),
                            decided_by=str(msg.get("decided_by", "user")),
                            note=str(msg.get("note", "")),
                        )
                    elif msg.get("type") == "cancel":
                        await r.approvals.cancel_run(run_id)
            except (WebSocketDisconnect, RuntimeError):
                # The human is gone. Release any waiter immediately instead of
                # blocking the agent until the approval deadline expires.
                released = await r.approvals.cancel_run(run_id)
                if released:
                    logger.warning(
                        "client disconnected; released %d pending approval(s) for run %s",
                        released, run_id,
                    )

        task = asyncio.create_task(coro)
        listener = asyncio.create_task(receive_decisions())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.4)
                    await ws.send_json(event)
                except asyncio.TimeoutError:
                    if task.done():
                        break
            while not queue.empty():
                await ws.send_json(queue.get_nowait())

            state = await task
        finally:
            listener.cancel()
            # Belt and braces: never leave a waiter stranded for this run.
            await r.approvals.cancel_run(run_id)
            r.approvals.set_notifier(None)

        await r.runs.update(
            run_id,
            status=RunStatus.FAILED if state.get("error") else RunStatus.COMPLETED,
            final_answer=state.get("final_answer", ""),
            error=state.get("error", ""),
            tasks=state.get("tasks", []),
        )
        await ws.send_json({
            "type": "final",
            "run_id": run_id,
            "final_answer": state.get("final_answer", ""),
            "tasks": state.get("tasks", []),
            "reflection": state.get("reflection", {}),
            "activated_skills": state.get("activated_skills", []),
        })
    except WebSocketDisconnect:
        logger.info("goal socket disconnected")
    except Exception as exc:
        logger.exception("goal socket error")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

@app.post("/api/skills/learn")
async def learn_skill(req: LearnRequest) -> dict[str, Any]:
    try:
        report = await runtime().skills.learn_from_youtube(
            req.url, skill_name=req.skill_name, force=req.force
        )
    except IngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc
    return report.to_dict()


@app.post("/api/skills/text")
async def learn_text(req: TextSkillRequest) -> dict[str, Any]:
    report = await runtime().skills.add_text_skill(req.title, req.text, req.source)
    return report.to_dict()


@app.get("/api/skills")
async def list_skills() -> dict[str, Any]:
    r = runtime()
    return {"skills": await r.skills.list_skills(), "stats": await r.skills.stats()}


@app.post("/api/skills/search")
async def search_skills(req: SkillSearchRequest) -> dict[str, Any]:
    matches = await runtime().skills.search(req.query, k=req.k)
    return {
        "matches": [
            {
                "title": m.title, "score": round(m.score, 4), "text": m.text[:800],
                "citation": m.citation(), "video_id": m.video_id, "skill_id": m.skill_id,
            }
            for m in matches
        ]
    }


@app.post("/api/skills/activate")
async def activate_skills(req: SkillSearchRequest) -> dict[str, Any]:
    prompt, matches = await runtime().skills.activate(req.query, k=req.k)
    return {
        "activated": bool(matches),
        "injected_prompt": prompt,
        "skills": [{"title": m.title, "score": round(m.score, 4)} for m in matches],
    }


@app.delete("/api/skills/{skill_id}")
async def forget_skill(skill_id: str) -> dict[str, str]:
    await runtime().skills.forget(skill_id)
    return {"status": "forgotten", "skill_id": skill_id}


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------

@app.post("/api/memory")
async def write_memory(req: MemoryWriteRequest) -> dict[str, Any]:
    try:
        kind = MemoryKind(req.kind)
    except ValueError as exc:
        raise HTTPException(422, f"Unknown memory kind '{req.kind}'") from exc
    mid = await runtime().memory.remember(
        MemoryItem(content=req.content, kind=kind, confidence=req.confidence, user_id=req.user_id)
    )
    return {"id": mid, "kind": kind.value}


@app.post("/api/memory/recall")
async def recall_memory(req: MemoryQueryRequest) -> dict[str, Any]:
    return {"memories": await runtime().memory.recall(req.query, k=req.k, user_id=req.user_id)}


@app.post("/api/memory/context")
async def memory_context(req: MemoryQueryRequest) -> dict[str, Any]:
    return {"context": await runtime().memory.twin_context(req.query, user_id=req.user_id, k=req.k)}


@app.get("/api/memory/stats")
async def memory_stats(user_id: str = "default") -> dict[str, Any]:
    return await runtime().memory.stats(user_id)


@app.post("/api/memory/reflect")
async def reflect(req: ReflectRequest) -> dict[str, Any]:
    return await runtime().reflection.reflect(
        task=req.task, outcome=req.outcome, success=req.success, user_id=req.user_id
    )


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------

@app.get("/api/guardian/constitution")
async def get_constitution() -> dict[str, Any]:
    return {
        "principles": [
            {"id": p.id, "title": p.title, "rule": p.rule, "examples": list(p.examples)}
            for p in CONSTITUTION
        ],
        "text": constitution_text(),
    }


@app.post("/api/guardian/check")
async def guardian_check(req: GuardianCheckRequest) -> dict[str, Any]:
    try:
        atype = ActionType(req.action_type)
    except ValueError:
        atype = ActionType.OTHER
    verdict = await runtime().guardian.review(
        ProposedAction(
            type=atype, payload=req.payload,
            description=req.description, target_path=req.target_path,
        )
    )
    return verdict.to_dict()


@app.get("/api/guardian/audit")
async def guardian_audit(limit: int = 50) -> dict[str, Any]:
    return {"entries": runtime().guardian.recent_audit(limit)}


# ---------------------------------------------------------------------------
# Intuition & artifacts
# ---------------------------------------------------------------------------

@app.post("/api/intuition")
async def intuition(req: IntuitionRequest) -> dict[str, Any]:
    s = await runtime().intuition.observe(req.context, user_id=req.user_id)
    return {"suggestion": s.to_dict() if s else None}


@app.get("/api/artifacts")
async def list_artifacts() -> dict[str, Any]:
    root = settings.workspace_dir
    files = [
        {
            "path": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "modified": p.stat().st_mtime,
        }
        for p in sorted(root.rglob("*")) if p.is_file()
    ]
    return {"workspace": str(root), "artifacts": files}


@app.get("/api/artifacts/{path:path}")
async def get_artifact(path: str):
    try:
        target = safe_workspace_path(settings, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, "Artifact not found")
    if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf"}:
        return FileResponse(target)
    return JSONResponse(
        {"path": path, "content": target.read_text(encoding="utf-8", errors="replace")[:200000]}
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
