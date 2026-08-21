"""Guardian-gated toolbelt.

Every tool call funnels through `ToolBelt.call()`, which asks the Guardian for
a verdict *before* anything happens. There is deliberately no unguarded path to
the executor or the filesystem.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings, get_settings
from ..core.executor import BaseExecutor, build_executor, safe_workspace_path
from ..guardian.guardian import ActionType, Decision, Guardian, ProposedAction

logger = logging.getLogger("vision.tools")


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: str
    blocked: bool = False
    verdict: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output[:8000],
            "blocked": self.blocked,
            "verdict": self.verdict,
            "artifacts": self.artifacts,
            "duration_s": round(self.duration_s, 3),
        }


class ToolBelt:
    def __init__(
        self,
        guardian: Guardian,
        executor: BaseExecutor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.guardian = guardian
        self.executor = executor or build_executor(self.settings)
        self._tools: dict[str, Callable[..., Any]] = {
            "shell": self._shell,
            "python": self._python,
            "write_file": self._write_file,
            "read_file": self._read_file,
            "make_chart": self._make_chart,
        }

    def available(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, tool: str, **kwargs: Any) -> ToolResult:
        start = time.monotonic()
        if tool not in self._tools:
            return ToolResult(tool=tool, ok=False, output=f"Unknown tool '{tool}'.")

        action = self._describe(tool, kwargs)
        verdict = await self.guardian.review(action)
        if verdict.decision is not Decision.ALLOW:
            msg = (
                f"BLOCKED by Guardian ({verdict.layer}): {verdict.reason}"
                + (f"\nSafe alternative: {verdict.safe_alternative}" if verdict.safe_alternative else "")
            )
            return ToolResult(
                tool=tool, ok=False, output=msg, blocked=True,
                verdict=verdict.to_dict(), duration_s=time.monotonic() - start,
            )
        try:
            result: ToolResult = await self._tools[tool](**kwargs)
        except Exception as exc:
            logger.exception("tool %s failed", tool)
            result = ToolResult(tool=tool, ok=False, output=f"{type(exc).__name__}: {exc}")
        result.verdict = verdict.to_dict()
        result.duration_s = time.monotonic() - start
        return result

    def _describe(self, tool: str, kwargs: dict[str, Any]) -> ProposedAction:
        if tool == "shell":
            return ProposedAction(
                ActionType.SHELL, str(kwargs.get("command", "")),
                description=str(kwargs.get("reason", "run a shell command")),
            )
        if tool in ("python", "make_chart"):
            return ProposedAction(
                ActionType.PYTHON, str(kwargs.get("code", "")) or json.dumps(kwargs)[:1000],
                description=str(kwargs.get("reason", f"run {tool}")),
            )
        if tool == "write_file":
            rel = str(kwargs.get("path", ""))
            return ProposedAction(
                ActionType.FILE_WRITE, f"write {rel}: {str(kwargs.get('content',''))[:500]}",
                description=str(kwargs.get("reason", "write a file")),
                target_path=str(self.settings.workspace_dir / rel),
            )
        if tool == "read_file":
            rel = str(kwargs.get("path", ""))
            return ProposedAction(
                ActionType.FILE_READ, f"read {rel}",
                description="read a workspace file",
                target_path=str(self.settings.workspace_dir / rel),
            )
        return ProposedAction(ActionType.OTHER, json.dumps(kwargs)[:1000], description=tool)

    # -- implementations -------------------------------------------------
    async def _shell(self, command: str, reason: str = "") -> ToolResult:
        res = await self.executor.run_shell(command)
        return ToolResult(tool="shell", ok=res.ok, output=res.summary())

    async def _python(self, code: str, reason: str = "") -> ToolResult:
        res = await self.executor.run_python(code)
        return ToolResult(tool="python", ok=res.ok, output=res.summary())

    async def _write_file(self, path: str, content: str, reason: str = "") -> ToolResult:
        target = safe_workspace_path(self.settings, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = target.relative_to(self.settings.workspace_dir)
        return ToolResult(
            tool="write_file", ok=True,
            output=f"Wrote {len(content)} chars to {rel}", artifacts=[str(rel)],
        )

    async def _read_file(self, path: str, max_chars: int = 20000) -> ToolResult:
        target = safe_workspace_path(self.settings, path)
        if not target.exists():
            return ToolResult(tool="read_file", ok=False, output=f"No such file: {path}")
        return ToolResult(
            tool="read_file", ok=True,
            output=target.read_text(encoding="utf-8", errors="replace")[:max_chars],
        )

    async def _make_chart(
        self,
        *,
        labels: list[str],
        values: list[float],
        title: str = "chart",
        kind: str = "bar",
        filename: str = "chart.png",
        reason: str = "",
    ) -> ToolResult:
        """Multi-modal output: render a chart to a PNG in the workspace."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ToolResult(tool="make_chart", ok=False, output="matplotlib is not installed.")

        target = safe_workspace_path(self.settings, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        if kind == "line":
            ax.plot(labels, values, marker="o", color="#6366f1")
        elif kind == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        else:
            ax.bar(labels, values, color="#6366f1")
        ax.set_title(title)
        if kind != "pie":
            ax.grid(axis="y", alpha=0.25)
            fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(target)
        plt.close(fig)
        rel = target.relative_to(self.settings.workspace_dir)
        return ToolResult(
            tool="make_chart", ok=True, output=f"Chart saved to {rel}", artifacts=[str(rel)]
        )
