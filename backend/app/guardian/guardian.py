"""Module 4 — the Pre-Execution Filter (Constitutional AI).

Every side-effecting action in Goal Mode (shell command, code execution, file
write, network call) must be cleared here first. Defence in depth, cheapest
layer first:

    Layer 0  path jail       — writes/reads must stay inside the workspace
    Layer 1  pattern filter  — deterministic regex catalogue of catastrophic ops
    Layer 2  Guardian LLM    — semantic judgement against the constitution
    Layer 3  alternative     — on denial, demand a safe alternative

Layer 1 is deliberately *not* delegated to the model: `rm -rf /` must be
blocked even if the LLM is unavailable, jailbroken or hallucinating.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..core.llm import LLMClient, Message
from .constitution import constitution_text

logger = logging.getLogger("vision.guardian")


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


class ActionType(str, Enum):
    SHELL = "shell"
    PYTHON = "python"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    NETWORK = "network"
    OTHER = "other"


@dataclass
class ProposedAction:
    type: ActionType
    payload: str
    description: str = ""
    target_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    decision: Decision
    reason: str
    layer: str
    violated_principles: list[str] = field(default_factory=list)
    safe_alternative: str | None = None
    confidence: float = 1.0
    checked_at: float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "layer": self.layer,
            "violated_principles": self.violated_principles,
            "safe_alternative": self.safe_alternative,
            "confidence": self.confidence,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# Layer 1 — deterministic catastrophic-pattern catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockRule:
    pattern: re.Pattern[str]
    principle: str
    reason: str
    alternative: str


def _rx(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


HARD_BLOCKS: tuple[BlockRule, ...] = (
    BlockRule(
        _rx(
            r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+"
            r"(~|\$HOME|/etc|/usr|/var|/bin|/boot|/lib|/sys|/proc|/)"
            r"([/\w.*-]*)\s*(\s|$|;|&|\|)"
        ),
        "P3_system_integrity",
        "Recursive deletion of a system or home root directory.",
        "Delete only specific files inside the sandboxed workspace directory.",
    ),
    BlockRule(
        _rx(r"\bmkfs(\.\w+)?\b|\bfdisk\b|\bparted\b"),
        "P3_system_integrity",
        "Filesystem/partition manipulation destroys data irreversibly.",
        "Work with a file-backed image or a temp directory instead.",
    ),
    BlockRule(
        _rx(r"\bdd\s+if=.*\bof=/dev/(sd|nvme|hd|disk)"),
        "P3_system_integrity",
        "Raw write to a block device.",
        "Write to a regular file inside the workspace.",
    ),
    BlockRule(
        _rx(r":\(\)\s*\{.*\}\s*;?\s*:|\bfork\s*bomb\b"),
        "P3_system_integrity",
        "Fork bomb / resource exhaustion.",
        "Run a bounded loop with an explicit iteration limit.",
    ),
    BlockRule(
        _rx(r"\b(shutdown|reboot|halt|poweroff|init\s+0)\b"),
        "P3_system_integrity",
        "Host power-state change.",
        "Stop only the specific process you started.",
    ),
    BlockRule(
        _rx(r"/etc/(passwd|shadow|sudoers)|\bchmod\s+(-R\s+)?777\s+/(\s|$)"),
        "P3_system_integrity",
        "Modification of critical system authentication files or permissions.",
        "Keep configuration inside the workspace.",
    ),
    BlockRule(
        _rx(r"(id_rsa|id_ed25519|\.ssh/|\.aws/credentials|\.env)\b[^\n]*\|[^\n]*\b(curl|wget|nc|netcat)\b"),
        "P4_secrets",
        "Piping credential material to a network utility (exfiltration).",
        "Never transmit secrets; reference them via environment variables locally.",
    ),
    BlockRule(
        _rx(r"\bcurl\b[^\n|;]*\|\s*(sudo\s+)?(ba)?sh\b|\bwget\b[^\n|;]*\|\s*(sudo\s+)?(ba)?sh\b"),
        "P1_no_harm",
        "Piping a remote script straight into a shell executes unreviewed code.",
        "Download the script to the workspace, inspect it, then run it explicitly.",
    ),
    BlockRule(
        _rx(r"\b(nmap|masscan|sqlmap|metasploit|msfconsole|hydra|john|hashcat|aircrack)\b"),
        "P2_no_unauthorized_access",
        "Offensive security tooling aimed at third-party systems.",
        "If this is authorized security testing, do it in your own lab with written scope.",
    ),
    BlockRule(
        _rx(r"\bgit\s+push\b[^\n]*\s(--force|-f)\b"),
        "P6_user_authority",
        "Force push can destroy shared history.",
        "Push to a feature branch, or use --force-with-lease after explicit approval.",
    ),
    BlockRule(
        _rx(r"\bsudo\b|\bsu\s+root\b|\bdoas\b"),
        "P6_user_authority",
        "Privilege escalation is outside the agent's mandate.",
        "Perform the operation unprivileged inside the workspace.",
    ),
    BlockRule(
        _rx(r"\b(keylogger|ransomware|botnet|rootkit|backdoor|cryptolocker)\b"),
        "P1_no_harm",
        "Explicitly malicious software.",
        "Describe the defensive goal instead; I can help with detection or hardening.",
    ),
)

# Actions that are not catastrophic but must never be auto-approved silently.
APPROVAL_PATTERNS: tuple[BlockRule, ...] = (
    BlockRule(
        _rx(r"\b(pip|npm|apt|apt-get|yum|brew)\s+(install|add)\b"),
        "P6_user_authority",
        "Installing packages changes the environment.",
        "Confirm the package list before installing.",
    ),
    BlockRule(
        _rx(r"\b(send|post)\b[^\n]*\b(email|slack|tweet|webhook)\b|\bsmtplib\b"),
        "P6_user_authority",
        "Outbound communication on the user's behalf.",
        "Draft the message and show it to the user for approval.",
    ),
)


GUARDIAN_SYSTEM = f"""VISION_ROLE: GUARDIAN

You are the Guardian of Vision AI: an uncompromising safety reviewer. An
autonomous agent wants to execute an action. Judge it against the constitution.

{constitution_text()}

Reply with ONLY this JSON object:
{{
  "decision": "allow" | "deny" | "needs_approval",
  "confidence": 0.0-1.0,
  "violated_principles": ["P#_id", ...],
  "reason": "one or two sentences of justification",
  "safe_alternative": "concrete alternative achieving the legitimate goal, or null"
}}

Guidance:
- "allow" for ordinary development work: reading/writing project files, running
  tests, computing, generating charts, calling documented public APIs.
- "needs_approval" for legal-but-consequential actions: installing software,
  spending money, sending messages, irreversible external writes.
- "deny" for anything violating the constitution.
- Judge the ACTUAL action, not hypothetical misuse. Do not be paranoid about
  benign development work — false denials make the agent useless."""


class Guardian:
    """The pre-execution filter."""

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.audit_log: list[dict[str, Any]] = []

    # -- Layer 0 ---------------------------------------------------------
    def _check_path_jail(self, action: ProposedAction) -> Verdict | None:
        if not action.target_path:
            return None
        workspace = self.settings.workspace_dir.resolve()
        try:
            target = Path(action.target_path).expanduser().resolve()
        except (OSError, RuntimeError):
            return Verdict(
                decision=Decision.DENY,
                reason=f"Unresolvable target path: {action.target_path}",
                layer="path_jail",
                violated_principles=["P3_system_integrity"],
            )
        if workspace not in target.parents and target != workspace:
            return Verdict(
                decision=Decision.DENY,
                reason=(
                    f"Path '{target}' is outside the sandboxed workspace "
                    f"('{workspace}'). Filesystem access is jailed."
                ),
                layer="path_jail",
                violated_principles=["P3_system_integrity"],
                safe_alternative=f"Write to {workspace / target.name} instead.",
            )
        return None

    # -- Layer 1 ---------------------------------------------------------
    def _check_patterns(self, action: ProposedAction) -> Verdict | None:
        text = f"{action.payload}\n{action.description}"
        for rule in HARD_BLOCKS:
            if rule.pattern.search(text):
                return Verdict(
                    decision=Decision.DENY,
                    reason=rule.reason,
                    layer="pattern_filter",
                    violated_principles=[rule.principle],
                    safe_alternative=rule.alternative,
                )
        for rule in APPROVAL_PATTERNS:
            if rule.pattern.search(text):
                return Verdict(
                    decision=Decision.NEEDS_APPROVAL,
                    reason=rule.reason,
                    layer="pattern_filter",
                    violated_principles=[rule.principle],
                    safe_alternative=rule.alternative,
                    confidence=0.8,
                )
        return None

    # -- Layer 2 ---------------------------------------------------------
    async def _check_llm(self, action: ProposedAction) -> Verdict:
        prompt = (
            f"<ACTION type=\"{action.type.value}\">\n{action.payload[:4000]}\n</ACTION>\n"
            f"<INTENT>{action.description or 'not stated'}</INTENT>\n"
            f"<CONTEXT>{action.metadata}</CONTEXT>"
        )
        try:
            data = await self.llm.complete_json(
                [Message("system", GUARDIAN_SYSTEM), Message("user", prompt)],
                role="guardian",
                temperature=0.0,
            )
        except Exception as exc:
            # Fail CLOSED: if the Guardian cannot judge, the action does not run.
            logger.error("Guardian LLM failed, failing closed: %s", exc)
            return Verdict(
                decision=Decision.NEEDS_APPROVAL,
                reason=f"Guardian review unavailable ({exc}). Escalating to human approval.",
                layer="guardian_llm_error",
                confidence=0.0,
            )
        try:
            decision = Decision(str(data.get("decision", "deny")).lower())
        except ValueError:
            decision = Decision.NEEDS_APPROVAL
        return Verdict(
            decision=decision,
            reason=str(data.get("reason", "No reason supplied.")),
            layer="guardian_llm",
            violated_principles=[str(p) for p in (data.get("violated_principles") or [])],
            safe_alternative=data.get("safe_alternative"),
            confidence=float(data.get("confidence", 0.5) or 0.5),
        )

    # -- public ----------------------------------------------------------
    async def review(self, action: ProposedAction) -> Verdict:
        if not self.settings.guardian_enabled:
            verdict = Verdict(
                decision=Decision.ALLOW,
                reason="Guardian disabled by configuration.",
                layer="disabled",
            )
            return self._record(action, verdict)

        verdict = self._check_path_jail(action) or self._check_patterns(action)

        if verdict is None:
            if self.settings.guardian_llm_review:
                verdict = await self._check_llm(action)
            else:
                verdict = Verdict(
                    decision=Decision.ALLOW,
                    reason="Passed deterministic filters; LLM review disabled.",
                    layer="pattern_filter",
                )
        elif verdict.decision is Decision.DENY:
            # Hard blocks short-circuit; no need to spend a Guardian call.
            pass

        if self.settings.guardian_mode == "monitor" and verdict.decision is Decision.DENY:
            logger.warning("MONITOR MODE: would have denied — %s", verdict.reason)
            verdict = Verdict(
                decision=Decision.ALLOW,
                reason=f"[monitor mode] violation logged but allowed: {verdict.reason}",
                layer=verdict.layer,
                violated_principles=verdict.violated_principles,
                confidence=verdict.confidence,
            )
        return self._record(action, verdict)

    def _record(self, action: ProposedAction, verdict: Verdict) -> Verdict:
        entry = {
            "action_type": action.type.value,
            "payload": action.payload[:500],
            "description": action.description,
            **verdict.to_dict(),
        }
        self.audit_log.append(entry)
        log = logger.info if verdict.allowed else logger.warning
        log("guardian[%s] %s: %s", verdict.layer, verdict.decision.value, verdict.reason)
        return verdict

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.audit_log[-limit:]
