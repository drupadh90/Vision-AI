"""Module 3 — Goal Mode: a LangGraph state machine with sub-agent spawning.

    plan ─► execute_wave ─► review ─┬─(retry / more work)─► execute_wave
                                    └─(done or budget spent)─► synthesize ─► reflect ─► END

Key properties
--------------
* The planner emits a **DAG**, not a list. Tasks declare `depends_on`, and the
  scheduler runs every ready task **concurrently** (bounded by a semaphore).
* The Manager reviews each wave and can **re-assign failed tasks**, optionally
  to a different specialist, with feedback attached.
* Cycles, dangling dependencies and runaway plans are rejected at plan time —
  an autonomous agent that deadlocks is worse than one that refuses.
* Every wave streams progress events for the UI.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..core.llm import LLMClient, Message
from ..guardian.guardian import Guardian
from ..memory.digital_twin import DigitalTwinMemory, ReflectionLoop
from ..skills.skill_store import SkillLibrary
from .approvals import ApprovalGate
from .sub_agents import AgentRole, SubAgentOutput, resolve_role, spawn
from .tools import ToolBelt

logger = logging.getLogger("vision.goal")

EventSink = Callable[[dict[str, Any]], Any]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    id: str
    title: str
    description: str
    agent: str = "generalist"
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    attempts: int = 0
    feedback: str = ""
    error: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "agent": self.agent,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "attempts": self.attempts,
            "result": self.result[:4000],
            "error": self.error,
            "feedback": self.feedback,
            "tool_results": self.tool_results,
        }


class GoalState(TypedDict, total=False):
    goal: str
    user_id: str
    run_id: str
    strategy: str
    tasks: list[dict[str, Any]]
    wave: int
    events: list[dict[str, Any]]
    final_answer: str
    reflection: dict[str, Any]
    twin_context: str
    skill_context: str
    activated_skills: list[dict[str, Any]]
    started_at: float
    done: bool
    error: str


PLANNER_SYSTEM = """VISION_ROLE: PLANNER

You are the Manager of Vision AI, decomposing a high-level goal into an
executable DAG of sub-tasks for specialised agents.

Available agents:
- research  : gather and verify information
- coder     : write and run code
- writer    : produce prose and documentation
- analyst   : quantitative analysis and charts
- critic    : review and quality control

Return ONLY this JSON object:
{
  "strategy": "one or two sentences on your approach",
  "tasks": [
    {
      "id": "t1",
      "title": "short imperative title",
      "description": "precise, self-contained instructions for the agent",
      "agent": "research|coder|writer|analyst|critic",
      "depends_on": []
    }
  ]
}

Rules:
- Between 2 and {max_tasks} tasks. Fewer, meatier tasks beat many trivial ones.
- `depends_on` must reference ids defined earlier in the list. NO CYCLES.
- Tasks with no dependency on each other WILL run in parallel — exploit that.
- Every task must be independently actionable from its description alone.
- The final task should consolidate the work into the deliverable."""


REVIEW_SYSTEM = """VISION_ROLE: PLANNER

You are the Manager reviewing a completed sub-task against the overall goal.

Return ONLY:
{
  "accept": true|false,
  "quality": 0.0-1.0,
  "feedback": "if rejecting, exactly what must change; else empty",
  "reassign_to": "research|coder|writer|analyst|critic|null"
}

Accept work that is genuinely usable. Reject only for real defects: unmet
requirements, fabrication, broken logic, or a stub pretending to be complete.
Set `reassign_to` when a different specialist would clearly do better."""


SYNTHESIS_SYSTEM = """VISION_ROLE: PLANNER

You are the Manager producing the final deliverable for the user. Merge the
sub-task outputs into one coherent answer.

- Lead with the result, not a description of your process.
- Use markdown. Preserve concrete artifacts (code, tables, file paths) verbatim.
- If a task failed, state plainly what is missing rather than papering over it.
- Do not narrate the internal task graph unless it helps the user."""


def validate_plan(tasks: list[Task], max_tasks: int) -> list[str]:
    """Return a list of structural problems; empty means the DAG is sound."""
    problems: list[str] = []
    ids = [t.id for t in tasks]
    if not tasks:
        problems.append("plan contains no tasks")
    if len(ids) != len(set(ids)):
        problems.append("duplicate task ids")
    if len(tasks) > max_tasks:
        problems.append(f"too many tasks ({len(tasks)} > {max_tasks})")
    known = set(ids)
    for t in tasks:
        for dep in t.depends_on:
            if dep not in known:
                problems.append(f"task {t.id} depends on unknown task {dep}")
            if dep == t.id:
                problems.append(f"task {t.id} depends on itself")

    # Kahn's algorithm — detect cycles.
    indegree = {t.id: len([d for d in t.depends_on if d in known]) for t in tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep in adj:
                adj[dep].append(t.id)
    queue = [i for i, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if seen != len(tasks):
        problems.append("dependency cycle detected")
    return problems


class GoalModeEngine:
    """Builds and runs the Goal Mode graph."""

    def __init__(
        self,
        llm: LLMClient,
        memory: DigitalTwinMemory,
        skills: SkillLibrary,
        guardian: Guardian,
        toolbelt: ToolBelt | None = None,
        settings: Settings | None = None,
        event_sink: EventSink | None = None,
        checkpointer: Any | None = None,
        approvals: ApprovalGate | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.memory = memory
        self.skills = skills
        self.guardian = guardian
        self.approvals = approvals
        self.toolbelt = toolbelt or ToolBelt(
            guardian, settings=self.settings, approvals=approvals
        )
        self.reflection = ReflectionLoop(memory, llm)
        self.event_sink = event_sink
        self.checkpointer = checkpointer
        self._sem = asyncio.Semaphore(self.settings.max_concurrency)
        self.graph = self._build_graph()

    # -- events ----------------------------------------------------------
    async def _emit(self, state: GoalState, kind: str, **payload: Any) -> None:
        event = {"type": kind, "ts": time.time(), **payload}
        state.setdefault("events", []).append(event)
        if self.event_sink:
            try:
                res = self.event_sink(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # a broken UI socket must not kill the run
                logger.debug("event sink failed", exc_info=True)

    # -- nodes -----------------------------------------------------------
    async def _node_prepare(self, state: GoalState) -> GoalState:
        goal = state["goal"]
        state["started_at"] = time.time()
        await self._emit(state, "goal_started", goal=goal)

        twin = await self.memory.twin_context(goal, user_id=state.get("user_id", "default"))
        skill_ctx, matches = await self.skills.activate(goal)
        state["twin_context"] = twin
        state["skill_context"] = skill_ctx
        state["activated_skills"] = [
            {"title": m.title, "score": round(m.score, 3), "citation": m.citation()}
            for m in matches
        ]
        if matches:
            await self._emit(
                state, "skills_activated",
                skills=[s["title"] for s in state["activated_skills"]],
            )
        return state

    async def _node_plan(self, state: GoalState) -> GoalState:
        goal = state["goal"]
        system = PLANNER_SYSTEM.replace("{max_tasks}", str(self.settings.max_subtasks))
        parts = [f"<GOAL>{goal}</GOAL>"]
        if state.get("twin_context"):
            parts.append(state["twin_context"])
        if state.get("skill_context"):
            parts.append(
                "You have relevant learned skills; plan to exploit them:\n"
                + state["skill_context"][:2500]
            )

        try:
            data = await self.llm.complete_json(
                [Message("system", system), Message("user", "\n\n".join(parts))],
                role="planner",
            )
        except Exception as exc:
            state["error"] = f"Planning failed: {exc}"
            state["tasks"] = []
            await self._emit(state, "error", message=state["error"])
            return state

        tasks = [
            Task(
                id=str(t.get("id") or f"t{i+1}"),
                title=str(t.get("title", "Untitled task")),
                description=str(t.get("description", "")),
                agent=str(t.get("agent", "generalist")),
                depends_on=[str(d) for d in (t.get("depends_on") or [])],
            )
            for i, t in enumerate(data.get("tasks") or [])
        ]
        problems = validate_plan(tasks, self.settings.max_subtasks)
        if problems:
            logger.warning("invalid plan (%s) — repairing to a linear chain", problems)
            await self._emit(state, "plan_repaired", problems=problems)
            for i, t in enumerate(tasks[: self.settings.max_subtasks]):
                t.depends_on = [tasks[i - 1].id] if i else []
            tasks = tasks[: self.settings.max_subtasks]
            if not tasks:
                tasks = [
                    Task(id="t1", title="Complete the goal",
                         description=state["goal"], agent="generalist")
                ]

        state["strategy"] = str(data.get("strategy", ""))
        state["tasks"] = [t.to_dict() for t in tasks]
        state["wave"] = 0
        await self._emit(
            state, "plan_ready", strategy=state["strategy"], tasks=state["tasks"]
        )
        return state

    def _load_tasks(self, state: GoalState) -> list[Task]:
        return [
            Task(
                id=d["id"], title=d["title"], description=d["description"],
                agent=d.get("agent", "generalist"), depends_on=list(d.get("depends_on") or []),
                status=TaskStatus(d.get("status", "pending")), result=d.get("result", ""),
                attempts=int(d.get("attempts", 0)), feedback=d.get("feedback", ""),
                error=d.get("error", ""), tool_results=list(d.get("tool_results") or []),
            )
            for d in state.get("tasks", [])
        ]

    @staticmethod
    def _ready(tasks: list[Task]) -> list[Task]:
        done = {t.id for t in tasks if t.status is TaskStatus.DONE}
        dead = {t.id for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)}
        ready: list[Task] = []
        for t in tasks:
            if t.status is not TaskStatus.PENDING:
                continue
            if any(d in dead for d in t.depends_on):
                t.status = TaskStatus.BLOCKED
                t.error = "upstream task failed"
                continue
            if all(d in done for d in t.depends_on):
                ready.append(t)
        return ready

    async def _node_execute(self, state: GoalState) -> GoalState:
        tasks = self._load_tasks(state)
        by_id = {t.id: t for t in tasks}
        ready = self._ready(tasks)
        state["wave"] = int(state.get("wave", 0)) + 1

        if not ready:
            state["tasks"] = [t.to_dict() for t in tasks]
            return state

        await self._emit(
            state, "wave_started", wave=state["wave"],
            tasks=[{"id": t.id, "title": t.title, "agent": t.agent} for t in ready],
        )

        async def run_one(task: Task) -> tuple[Task, SubAgentOutput]:
            async with self._sem:  # bounded concurrency
                task.status = TaskStatus.RUNNING
                task.attempts += 1
                await self._emit(
                    state, "task_started", task_id=task.id, title=task.title, agent=task.agent
                )
                role: AgentRole = resolve_role(task.agent)
                # Tag the toolbelt so any approval prompt can name the asker.
                self.toolbelt.run_id = state.get("run_id", "")
                self.toolbelt.task_id = task.id
                agent = spawn(role, self.llm, self.toolbelt)
                dep_ctx = "\n\n".join(
                    f"### Output of '{by_id[d].title}' ({d})\n{by_id[d].result[:3000]}"
                    for d in task.depends_on
                    if d in by_id and by_id[d].result
                )
                out = await agent.run(
                    task_id=task.id,
                    title=task.title,
                    description=task.description,
                    goal=state["goal"],
                    dependency_context=dep_ctx,
                    twin_context=state.get("twin_context", ""),
                    skill_context=state.get("skill_context", ""),
                    feedback=task.feedback,
                )
                return task, out

        for coro in asyncio.as_completed([run_one(t) for t in ready]):
            task, out = await coro
            if out.ok and out.content.strip():
                task.result = out.content
                task.tool_results = out.tool_results
                task.status = TaskStatus.DONE  # provisional; the Manager reviews next
                await self._emit(
                    state, "task_completed", task_id=task.id, title=task.title,
                    agent=task.agent, preview=out.content[:400],
                    tool_results=out.tool_results,
                )
            else:
                task.error = out.error or "empty output"
                task.status = TaskStatus.FAILED
                await self._emit(
                    state, "task_failed", task_id=task.id, title=task.title, error=task.error
                )

        state["tasks"] = [t.to_dict() for t in tasks]
        return state

    async def _node_review(self, state: GoalState) -> GoalState:
        """Manager reviews the wave, rejecting weak work and re-assigning it."""
        tasks = self._load_tasks(state)
        max_retries = self.settings.max_task_retries

        just_done = [
            t for t in tasks
            if t.status is TaskStatus.DONE and t.result and not t.feedback.startswith("ACCEPTED")
        ]
        for task in just_done:
            try:
                data = await self.llm.complete_json(
                    [
                        Message("system", REVIEW_SYSTEM),
                        Message(
                            "user",
                            f"<GOAL>{state['goal']}</GOAL>\n"
                            f"<TASK>{task.title}\n{task.description}</TASK>\n"
                            f"<OUTPUT>{task.result[:5000]}</OUTPUT>",
                        ),
                    ],
                    role="planner",
                    temperature=0.0,
                )
            except Exception as exc:
                logger.warning("review failed for %s: %s", task.id, exc)
                task.feedback = "ACCEPTED (review unavailable)"
                continue

            if data.get("accept", True):
                task.feedback = f"ACCEPTED (quality {data.get('quality', 0.7)})"
                continue

            if task.attempts > max_retries:
                task.feedback = f"ACCEPTED after {task.attempts} attempts (retry budget spent)"
                await self._emit(state, "retry_budget_spent", task_id=task.id)
                continue

            task.status = TaskStatus.PENDING  # re-queue for another wave
            task.feedback = str(data.get("feedback", "Improve quality and completeness."))
            reassign = data.get("reassign_to")
            if reassign and str(reassign).lower() not in ("null", "none", ""):
                old, task.agent = task.agent, str(reassign).lower()
                await self._emit(
                    state, "task_reassigned", task_id=task.id, from_agent=old, to_agent=task.agent
                )
            await self._emit(
                state, "task_rejected", task_id=task.id, title=task.title, feedback=task.feedback
            )

        # Retry outright failures too, while budget remains.
        for task in tasks:
            if task.status is TaskStatus.FAILED and task.attempts <= max_retries:
                task.status = TaskStatus.PENDING
                task.feedback = f"Previous attempt errored: {task.error}. Try a different approach."

        state["tasks"] = [t.to_dict() for t in tasks]
        return state

    def _should_continue(self, state: GoalState) -> str:
        if state.get("error"):
            return "synthesize"
        elapsed = time.time() - float(state.get("started_at", time.time()))
        if elapsed > self.settings.goal_wall_clock_seconds:
            logger.warning("goal wall clock exceeded")
            return "synthesize"
        tasks = self._load_tasks(state)
        if any(t.status is TaskStatus.PENDING for t in tasks) and self._ready(tasks):
            return "execute"
        if int(state.get("wave", 0)) >= self.settings.max_subtasks * 2:
            return "synthesize"  # hard loop guard
        return "synthesize"

    async def _node_synthesize(self, state: GoalState) -> GoalState:
        tasks = self._load_tasks(state)
        completed = [t for t in tasks if t.status is TaskStatus.DONE]
        failed = [t for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]

        if not completed:
            state["final_answer"] = (
                "I could not complete this goal.\n\n"
                + (f"Error: {state['error']}\n" if state.get("error") else "")
                + "\n".join(f"- {t.title}: {t.error or 'no output'}" for t in failed)
            )
            state["done"] = True
            await self._emit(state, "goal_failed", reason=state.get("error", "all tasks failed"))
            return state

        body = "\n\n".join(
            f"## {t.title} (agent: {t.agent})\n{t.result[:6000]}" for t in completed
        )
        parts = [f"<GOAL>{state['goal']}</GOAL>", f"<COMPLETED_WORK>\n{body}\n</COMPLETED_WORK>"]
        if failed:
            parts.append(
                "<FAILED_TASKS>\n"
                + "\n".join(f"- {t.title}: {t.error}" for t in failed)
                + "\n</FAILED_TASKS>"
            )
        if state.get("twin_context"):
            parts.append(state["twin_context"])

        try:
            resp = await self.llm.complete(
                [Message("system", SYNTHESIS_SYSTEM), Message("user", "\n\n".join(parts))],
                role="planner",
                max_tokens=self.settings.llm_max_tokens,
            )
            state["final_answer"] = resp.text
        except Exception as exc:
            state["final_answer"] = f"# Partial results\n\n{body}\n\n(Synthesis failed: {exc})"

        state["done"] = True
        await self._emit(
            state, "goal_completed",
            completed=len(completed), failed=len(failed),
            answer_preview=state["final_answer"][:400],
        )
        return state

    async def _node_reflect(self, state: GoalState) -> GoalState:
        tasks = self._load_tasks(state)
        success = bool(state.get("final_answer")) and not state.get("error")
        # Self-improvement is valuable but strictly secondary: never let it
        # discard a deliverable the user is already waiting on.
        try:
            data = await self.reflection.reflect(
                task=state["goal"],
                outcome=state.get("final_answer", "")[:5000],
                success=success,
                user_id=state.get("user_id", "default"),
                extra_context=(
                    f"{len(tasks)} sub-tasks across {state.get('wave', 0)} waves; "
                    f"agents used: {sorted({t.agent for t in tasks})}"
                ),
            )
        except Exception as exc:
            logger.warning("reflection node failed, continuing: %s", exc)
            data = {"error": str(exc), "stored": 0}
        state["reflection"] = data
        await self._emit(state, "reflection_written", lesson=data.get("lesson"), stored=data.get("stored", 0))
        return state

    # -- graph -----------------------------------------------------------
    def _build_graph(self) -> Any:
        g: StateGraph = StateGraph(GoalState)
        g.add_node("prepare", self._node_prepare)
        g.add_node("plan", self._node_plan)
        g.add_node("execute", self._node_execute)
        g.add_node("review", self._node_review)
        g.add_node("synthesize", self._node_synthesize)
        g.add_node("reflect", self._node_reflect)

        g.set_entry_point("prepare")
        g.add_edge("prepare", "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "review")
        g.add_conditional_edges(
            "review", self._should_continue,
            {"execute": "execute", "synthesize": "synthesize"},
        )
        g.add_edge("synthesize", "reflect")
        g.add_edge("reflect", END)
        # With a checkpointer attached, every node transition is persisted, so a
        # run that dies mid-flight can be resumed from its last completed node.
        return g.compile(checkpointer=self.checkpointer) if self.checkpointer else g.compile()

    def _config(self, run_id: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {"recursion_limit": self.settings.max_subtasks * 6 + 20}
        if self.checkpointer is not None:
            # thread_id is the durable handle used to resume this exact run.
            cfg["configurable"] = {"thread_id": run_id}
        return cfg

    async def run(self, goal: str, user_id: str = "default", run_id: str = "") -> GoalState:
        rid = run_id or uuid.uuid4().hex[:16]
        initial: GoalState = {
            "goal": goal, "user_id": user_id, "run_id": rid, "tasks": [],
            "events": [], "wave": 0, "done": False,
        }
        result = await self.graph.ainvoke(initial, config=self._config(rid))
        return dict(result)  # type: ignore[return-value]

    async def resume(self, run_id: str) -> GoalState:
        """Continue an interrupted run from its last persisted checkpoint."""
        if self.checkpointer is None:
            raise RuntimeError("Cannot resume: no checkpointer configured.")
        config = self._config(run_id)
        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            raise ValueError(f"No checkpoint found for run '{run_id}'.")
        logger.info("resuming run %s (next node: %s)", run_id, snapshot.next)
        # Passing None replays from the checkpoint instead of restarting.
        result = await self.graph.ainvoke(None, config=config)
        return dict(result)  # type: ignore[return-value]

    async def get_state(self, run_id: str) -> dict[str, Any] | None:
        if self.checkpointer is None:
            return None
        snapshot = await self.graph.aget_state(self._config(run_id))
        if not snapshot.values:
            return None
        return {"values": dict(snapshot.values), "next": list(snapshot.next)}
