"""Module 5 — Proactive Intuition.

Hermes waits to be told. Vision AI watches the context and speaks up — but on a
strict budget, because an assistant that interrupts constantly is worse than
one that stays quiet.

Discipline applied here:
* a cooldown between suggestions;
* a confidence floor;
* de-duplication against recently-made suggestions;
* cheap deterministic triggers first, LLM only when something looks promising.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..core.llm import LLMClient, Message
from ..memory.digital_twin import DigitalTwinMemory
from ..skills.skill_store import SkillLibrary

logger = logging.getLogger("vision.intuition")


@dataclass
class Suggestion:
    text: str
    confidence: float
    rationale: str = ""
    kind: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion": self.text,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "kind": self.kind,
            "metadata": self.metadata,
        }


INTUITION_SYSTEM = """VISION_ROLE: INTUITION

You are the intuition of Vision AI. Given recent context, decide whether to
proactively offer ONE genuinely useful action. Silence is the correct answer
most of the time — only speak when the value is obvious.

Return ONLY:
{
  "should_speak": true|false,
  "confidence": 0.0-1.0,
  "suggestion": "one sentence, addressed to the user, offering a concrete action",
  "rationale": "why this is worth interrupting for"
}

Never suggest something the user has just declined. Never state the obvious.
Never pad. If in doubt, should_speak = false."""


CODE_HINT = re.compile(
    r"```|\bimport\s+\w+|\bdef\s+\w+\(|\bclass\s+\w+|\bnpm\b|\bpip install\b|"
    r"\.py\b|\.ts\b|\.tsx\b|\bgit\b|traceback|exception|stack trace",
    re.IGNORECASE,
)
FRUSTRATION = re.compile(
    r"\b(not working|doesn'?t work|broken|stuck|failed again|no idea|confused|"
    r"why isn'?t|still failing)\b",
    re.IGNORECASE,
)
REPETITION = re.compile(r"\b(again|same error|still)\b", re.IGNORECASE)


class IntuitionEngine:
    def __init__(
        self,
        llm: LLMClient,
        memory: DigitalTwinMemory,
        skills: SkillLibrary,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.memory = memory
        self.skills = skills
        self._last_spoke: float = 0.0
        self._recent: list[str] = []

    def _cooling_down(self) -> bool:
        return (time.time() - self._last_spoke) < self.settings.proactive_cooldown_seconds

    def _is_duplicate(self, text: str) -> bool:
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        return any(
            norm == r or (len(norm) > 20 and (norm in r or r in norm)) for r in self._recent
        )

    def _register(self, text: str) -> None:
        self._last_spoke = time.time()
        self._recent.append(re.sub(r"\W+", " ", text.lower()).strip())
        self._recent = self._recent[-10:]

    async def observe(self, context: str, *, user_id: str = "default") -> Suggestion | None:
        """Consider the context; return a suggestion or None (usually None)."""
        if not self.settings.proactive_enabled or self._cooling_down() or not context.strip():
            return None

        # --- cheap deterministic trigger: do we have a relevant learned skill?
        if CODE_HINT.search(context):
            matches = await self.skills.search(context[-1500:], k=1)
            if matches and matches[0].score >= max(
                self.settings.skill_match_threshold, 0.35
            ):
                m = matches[0]
                text = (
                    f"I've learned \"{m.title}\" from a video that looks relevant here — "
                    "want me to apply it?"
                )
                if not self._is_duplicate(text):
                    self._register(text)
                    return Suggestion(
                        text=text,
                        confidence=min(0.95, 0.55 + m.score),
                        rationale=f"Skill DB match at {m.score:.2f} relevance.",
                        kind="skill_offer",
                        metadata={"skill_id": m.skill_id, "citation": m.citation()},
                    )

        # --- frustration trigger: offer to take the whole thing over
        if FRUSTRATION.search(context) and REPETITION.search(context):
            text = (
                "This looks like it's been fighting you for a while — want me to take "
                "it into Goal Mode and work the problem end to end?"
            )
            if not self._is_duplicate(text):
                self._register(text)
                return Suggestion(
                    text=text, confidence=0.7,
                    rationale="Repeated failure signals detected.", kind="goal_mode_offer",
                )

        # --- otherwise ask the model, primed with what we know about the user
        twin = await self.memory.twin_context(context[-1000:], user_id=user_id, k=4)
        try:
            data = await self.llm.complete_json(
                [
                    Message("system", INTUITION_SYSTEM),
                    Message(
                        "user",
                        f"<CONTEXT>{context[-3000:]}</CONTEXT>\n\n{twin}".strip(),
                    ),
                ],
                role="reflection",
                temperature=0.5,
            )
        except Exception as exc:
            logger.debug("intuition LLM failed: %s", exc)
            return None

        if not data.get("should_speak"):
            return None
        confidence = float(data.get("confidence", 0.0) or 0.0)
        text = str(data.get("suggestion", "")).strip()
        if not text or confidence < self.settings.proactive_min_confidence:
            return None
        if self._is_duplicate(text):
            return None
        self._register(text)
        return Suggestion(
            text=text, confidence=confidence,
            rationale=str(data.get("rationale", "")), kind="llm",
        )
