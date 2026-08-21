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

from .agents.goal_graph import GoalModeEngine
from .agents.intuition import IntuitionEngine
from .agents.tools import ToolBelt
from .config import get_settings
from .core.executor import build_executor, safe_workspace_path
from .core.llm import LLMClient
from .guardian.constitution import CONSTITUTION, constitution_text
from .guardian.guardian import ActionType, Guardian, ProposedAction
from .memory.digital_twin import DigitalTwinMemory, MemoryItem, MemoryKind, ReflectionLoop
from .schemas import (
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

    def engine(self, event_sink: Any = None) -> GoalModeEngine:
        return GoalModeEngine(
            llm=self.llm, memory=self.memory, skills=self.skills,
            guardian=self.guardian, toolbelt=self.toolbelt,
            settings=settings, event_sink=event_sink,
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
    toolbelt = ToolBelt(guardian, build_executor(settings), settings)
    rt = Runtime(
        llm=llm, memory=memory, skills=skills, guardian=guardian, toolbelt=toolbelt,
        intuition=IntuitionEngine(llm, memory, skills, settings),
        reflection=ReflectionLoop(memory, llm),
    )
    logger.info(
        "Vision AI online | llm=%s embeddings=%s executor=%s guardian=%s",
        llm.provider_name, settings.effective_embedding_provider(),
        settings.executor, settings.guardian_mode if settings.guardian_enabled else "off",
    )
    if llm.is_mock:
        logger.warning("Running with the MOCK LLM — set OPENAI_API_KEY for real reasoning.")
    yield
    await llm.aclose()


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
    state = await runtime().engine().run(req.goal, req.user_id)
    return GoalResponse(
        goal=req.goal,
        strategy=state.get("strategy", ""),
        final_answer=state.get("final_answer", ""),
        tasks=state.get("tasks", []),
        events=state.get("events", []),
        activated_skills=state.get("activated_skills", []),
        reflection=state.get("reflection", {}),
        error=state.get("error", ""),
    )


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

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def sink(event: dict[str, Any]) -> None:
            await queue.put(event)

        engine = runtime().engine(event_sink=sink)
        task = asyncio.create_task(engine.run(goal, user_id))
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
        await ws.send_json({
            "type": "final",
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
