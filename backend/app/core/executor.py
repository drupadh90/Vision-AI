"""Sandboxed execution for Goal Mode tool calls.

Nothing here runs until `Guardian.review()` has returned ALLOW — see
`tools.py`, which is the only place these executors should be called from.

Backends
--------
`docker`     : ephemeral container, no network by default, read-only rootfs,
               dropped capabilities, pids/memory/cpu caps, workspace bind-mount.
`subprocess` : workspace-cwd child process. Weaker isolation — explicitly
               opt-in, and loudly logged.
`disabled`   : refuses everything (safe default for hosted deployments).

If `docker` is configured but the daemon is unreachable, we do NOT silently
fall back to running commands on the host. We fail closed and say why.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings

logger = logging.getLogger("vision.executor")


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    backend: str = "none"
    artifacts: list[str] = field(default_factory=list)

    def summary(self, limit: int = 2000) -> str:
        parts = [f"exit={self.exit_code} backend={self.backend} ({self.duration_s:.2f}s)"]
        if self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout[:limit]}")
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr[:limit]}")
        return "\n".join(parts)


class ExecutorUnavailable(RuntimeError):
    pass


class BaseExecutor:
    name = "base"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.workspace = self.settings.workspace_dir
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def run_shell(self, command: str) -> ExecResult:  # pragma: no cover
        raise NotImplementedError

    async def run_python(self, code: str) -> ExecResult:
        script = self.workspace / f"_vision_{uuid.uuid4().hex[:8]}.py"
        script.write_text(code, encoding="utf-8")
        try:
            return await self.run_shell(f"python {shlex.quote(script.name)}")
        finally:
            script.unlink(missing_ok=True)


class DisabledExecutor(BaseExecutor):
    name = "disabled"

    async def run_shell(self, command: str) -> ExecResult:
        return ExecResult(
            ok=False,
            stderr="Execution is disabled (VISION_EXECUTOR=disabled).",
            backend=self.name,
        )


class SubprocessExecutor(BaseExecutor):
    """Workspace-jailed child process. Isolation is best-effort only."""

    name = "subprocess"

    async def run_shell(self, command: str) -> ExecResult:
        start = time.monotonic()
        logger.warning("subprocess executor running (weak isolation): %s", command[:120])
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.workspace)},
            )
        except OSError as exc:
            return ExecResult(ok=False, stderr=str(exc), backend=self.name)
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=self.settings.exec_timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(
                ok=False,
                stderr=f"Timed out after {self.settings.exec_timeout_seconds}s.",
                exit_code=124,
                duration_s=time.monotonic() - start,
                backend=self.name,
            )
        return ExecResult(
            ok=proc.returncode == 0,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            exit_code=proc.returncode,
            duration_s=time.monotonic() - start,
            backend=self.name,
        )


class DockerExecutor(BaseExecutor):
    """Ephemeral, network-less container per command."""

    name = "docker"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._checked = False
        self._available = False
        self._reason = ""

    async def _probe(self) -> bool:
        if self._checked:
            return self._available
        self._checked = True
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info", "--format", "{{.ServerVersion}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=15)
            self._available = proc.returncode == 0
            self._reason = err.decode("utf-8", "replace")[:300] if not self._available else ""
            if self._available:
                logger.info("Docker sandbox available (server %s)", out.decode().strip())
        except (FileNotFoundError, asyncio.TimeoutError, OSError) as exc:
            self._available = False
            self._reason = f"{type(exc).__name__}: {exc}"
        if not self._available:
            logger.error("Docker sandbox unavailable: %s", self._reason)
        return self._available

    def _docker_args(self, command: str) -> list[str]:
        s = self.settings
        return [
            "docker", "run", "--rm", "--interactive=false",
            "--network", s.exec_network,
            "--memory", s.exec_memory_limit,
            "--cpus", str(s.exec_cpu_limit),
            "--pids-limit", "256",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",                       # rootfs immutable...
            "--tmpfs", "/tmp:rw,size=64m",       # ...except scratch
            "--user", "1000:1000",
            "-v", f"{self.workspace}:/workspace:rw",
            "-w", "/workspace",
            s.docker_image,
            "sh", "-c", command,
        ]

    async def run_shell(self, command: str) -> ExecResult:
        if not await self._probe():
            return ExecResult(
                ok=False,
                stderr=(
                    "Docker sandbox unavailable — refusing to execute on the host. "
                    f"Reason: {self._reason}\n"
                    "Start Docker, or set VISION_EXECUTOR=subprocess to accept "
                    "weaker isolation, or VISION_EXECUTOR=disabled."
                ),
                backend=self.name,
            )
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *self._docker_args(command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=self.settings.exec_timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(
                ok=False,
                stderr=f"Container timed out after {self.settings.exec_timeout_seconds}s.",
                exit_code=124,
                duration_s=time.monotonic() - start,
                backend=self.name,
            )
        return ExecResult(
            ok=proc.returncode == 0,
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            exit_code=proc.returncode,
            duration_s=time.monotonic() - start,
            backend=self.name,
        )


def build_executor(settings: Settings | None = None) -> BaseExecutor:
    s = settings or get_settings()
    return {
        "docker": DockerExecutor,
        "subprocess": SubprocessExecutor,
        "disabled": DisabledExecutor,
    }[s.executor](s)


def safe_workspace_path(settings: Settings, relative: str) -> Path:
    """Resolve a user/agent-supplied path, guaranteeing it stays in the jail."""
    workspace = settings.workspace_dir.resolve()
    candidate = (workspace / relative).resolve()
    if workspace != candidate and workspace not in candidate.parents:
        raise ValueError(f"Path escapes workspace jail: {relative}")
    return candidate
