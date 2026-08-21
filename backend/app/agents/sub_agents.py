"""Specialised sub-agents spawned by the Manager in Goal Mode.

Each role has its own system prompt, tool permissions and model. They are
spawned dynamically per task, run concurrently when the DAG allows, and every
one of them receives Digital Twin context + activated YouTube skills — so a
freshly spawned Coder Agent already "knows" what the user likes and what the
tutorials taught.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.llm import LLMClient, Message
from .tools import ToolBelt, ToolResult

logger = logging.getLogger("vision.subagents")


class AgentRole(str, Enum):
    RESEARCH = "research"
    CODER = "coder"
    WRITER = "writer"
    ANALYST = "analyst"
    CRITIC = "critic"
    GENERALIST = "generalist"


@dataclass
class AgentSpec:
    role: AgentRole
    system_prompt: str
    tools: tuple[str, ...] = ()
    temperature: float = 0.2


AGENT_REGISTRY: dict[AgentRole, AgentSpec] = {
    AgentRole.RESEARCH: AgentSpec(
        role=AgentRole.RESEARCH,
        system_prompt=(
            "You are the Research Agent of Vision AI. You gather, verify and "
            "synthesise information. Distinguish clearly between what you know, "
            "what you inferred, and what you could not verify. Never invent "
            "citations, statistics or URLs. Output tight, structured findings "
            "with an explicit confidence note on anything uncertain."
        ),
        tools=("read_file", "write_file"),
        temperature=0.3,
    ),
    AgentRole.CODER: AgentSpec(
        role=AgentRole.CODER,
        system_prompt=(
            "You are the Coder Agent of Vision AI. You write correct, minimal, "
            "readable code. Prefer the standard library. Handle errors "
            "explicitly. When you produce a file, give the full contents — never "
            "an elided fragment. Verify your work by running it when a tool is "
            "available, and report failures honestly rather than claiming success."
        ),
        tools=("python", "shell", "write_file", "read_file"),
        temperature=0.1,
    ),
    AgentRole.WRITER: AgentSpec(
        role=AgentRole.WRITER,
        system_prompt=(
            "You are the Writer Agent of Vision AI. You turn raw material into "
            "clear, well-structured prose in the user's voice. Lead with the "
            "conclusion. Cut filler, hedging and throat-clearing. Use markdown "
            "headings and lists where they aid scanning, not for decoration."
        ),
        tools=("write_file", "read_file"),
        temperature=0.4,
    ),
    AgentRole.ANALYST: AgentSpec(
        role=AgentRole.ANALYST,
        system_prompt=(
            "You are the Analyst Agent of Vision AI. You quantify things. Show "
            "your reasoning, state assumptions explicitly, and visualise results "
            "with the make_chart tool when a comparison or trend is involved."
        ),
        tools=("python", "make_chart", "write_file", "read_file"),
        temperature=0.15,
    ),
    AgentRole.CRITIC: AgentSpec(
        role=AgentRole.CRITIC,
        system_prompt=(
            "You are the Critic Agent of Vision AI. You find what is wrong before "
            "the user does: factual errors, missed requirements, broken logic, "
            "security issues. Be specific and actionable. If the work is genuinely "
            "good, say so plainly instead of inventing nitpicks."
        ),
        tools=("read_file",),
        temperature=0.2,
    ),
    AgentRole.GENERALIST: AgentSpec(
        role=AgentRole.GENERALIST,
        system_prompt=(
            "You are a Generalist Agent of Vision AI. Complete the assigned task "
            "pragmatically and report the result plainly."
        ),
        tools=("read_file", "write_file", "python"),
        temperature=0.25,
    ),
}


def resolve_role(name: str) -> AgentRole:
    try:
        return AgentRole(name.strip().lower())
    except ValueError:
        return AgentRole.GENERALIST


@dataclass
class SubAgentOutput:
    role: AgentRole
    task_id: str
    content: str
    ok: bool = True
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    error: str | None = None


class SubAgent:
    """A single specialised worker."""

    def __init__(
        self,
        spec: AgentSpec,
        llm: LLMClient,
        toolbelt: ToolBelt | None = None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.toolbelt = toolbelt

    async def run(
        self,
        *,
        task_id: str,
        title: str,
        description: str,
        goal: str,
        dependency_context: str = "",
        twin_context: str = "",
        skill_context: str = "",
        feedback: str = "",
    ) -> SubAgentOutput:
        system_parts = [self.spec.system_prompt]
        if twin_context:
            system_parts.append(twin_context)
        if skill_context:
            system_parts.append(skill_context)
        if self.toolbelt and self.spec.tools:
            allowed = [t for t in self.spec.tools if t in self.toolbelt.available()]
            if allowed:
                system_parts.append(
                    "## Tools\nYou may request tools by emitting a JSON block:\n"
                    '```tool\n{"tool": "write_file", "args": {"path": "out.md", "content": "..."}}\n```\n'
                    f"Available to you: {', '.join(allowed)}.\n"
                    "Every call is screened by the Guardian against the constitution. "
                    "Emit at most one tool block per reply, then continue with your answer."
                )

        user_parts = [
            f"<GOAL>{goal}</GOAL>",
            f"<TASK id=\"{task_id}\">{title}\n\n{description}</TASK>",
        ]
        if dependency_context:
            user_parts.append(f"<UPSTREAM_RESULTS>\n{dependency_context}\n</UPSTREAM_RESULTS>")
        if feedback:
            user_parts.append(
                f"<MANAGER_FEEDBACK>Your previous attempt was rejected.\n{feedback}\n"
                "Address this specifically.</MANAGER_FEEDBACK>"
            )
        user_parts.append(
            "Produce the deliverable for this task only. Be concrete and complete."
        )

        messages = [
            Message("system", "\n\n".join(system_parts)),
            Message("user", "\n\n".join(user_parts)),
        ]
        try:
            resp = await self.llm.complete(
                messages, role="worker", temperature=self.spec.temperature
            )
        except Exception as exc:
            logger.exception("sub-agent %s failed", self.spec.role.value)
            return SubAgentOutput(
                role=self.spec.role, task_id=task_id, content="", ok=False, error=str(exc)
            )

        content = resp.text
        tool_results: list[dict[str, Any]] = []
        if self.toolbelt:
            content, tool_results = await self._run_tools(content)

        return SubAgentOutput(
            role=self.spec.role,
            task_id=task_id,
            content=content,
            ok=True,
            tool_results=tool_results,
            tokens=resp.total_tokens,
        )

    async def _run_tools(self, content: str) -> tuple[str, list[dict[str, Any]]]:
        import json as _json
        import re

        results: list[dict[str, Any]] = []
        blocks = re.findall(r"```tool\s*(.*?)```", content, re.DOTALL)
        for block in blocks[:3]:  # bound tool usage per turn
            try:
                spec = _json.loads(block.strip())
            except _json.JSONDecodeError:
                continue
            name = spec.get("tool", "")
            if name not in self.spec.tools:
                results.append({"tool": name, "ok": False, "output": "Tool not permitted for this role."})
                continue
            assert self.toolbelt is not None
            res: ToolResult = await self.toolbelt.call(name, **(spec.get("args") or {}))
            results.append(res.to_dict())
            content += f"\n\n**Tool `{name}`** → {res.output[:1200]}"
        return content, results


def spawn(role: AgentRole, llm: LLMClient, toolbelt: ToolBelt | None = None) -> SubAgent:
    """Dynamically spawn a specialised sub-agent."""
    spec = AGENT_REGISTRY.get(role, AGENT_REGISTRY[AgentRole.GENERALIST])
    logger.info("spawning %s agent", spec.role.value)
    return SubAgent(spec, llm, toolbelt)
