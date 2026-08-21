"""Module 1 — the Digital Twin self-improvement engine.

Vision AI's differentiator over Hermes: it does not merely *recall* the past,
it *converges on the user*. Three cooperating pieces:

1. `DigitalTwinMemory` — a typed, persistent Chroma-backed store of episodes,
   preferences, behavioural patterns and lessons.
2. `ReflectionLoop`    — runs after every task, critiques the agent's own
   performance, and writes durable lessons + preference deltas back to memory.
3. `twin_context()`    — renders the highest-signal memories into a system
   prompt block so *future* runs behave more like the user.

Preferences carry a confidence that is reinforced when re-observed, so beliefs
about the user strengthen with evidence instead of being overwritten.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..config import Settings, get_settings
from ..core.embeddings import BaseEmbedder, build_embedder
from ..core.llm import LLMClient, Message
from .vector_store import VectorStore

logger = logging.getLogger("vision.twin")


class MemoryKind(str, Enum):
    EPISODE = "episode"          # something that happened
    PREFERENCE = "preference"    # a durable fact about how the user likes things
    PATTERN = "pattern"          # recurring behaviour
    LESSON = "lesson"            # self-critique output
    FEEDBACK = "feedback"        # explicit user correction (highest signal)


@dataclass
class MemoryItem:
    content: str
    kind: MemoryKind = MemoryKind.EPISODE
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = "default"
    confidence: float = 0.5
    importance: float = 0.5
    source: str = "runtime"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def to_metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("content", None)
        d["kind"] = self.kind.value
        d["tags"] = ", ".join(self.tags)
        return d


REFLECTION_SYSTEM = """VISION_ROLE: REFLECTION

You are the reflective conscience of Vision AI, an agent whose purpose is to
become a faithful second version of its user. You have just finished a task.
Critique your own performance honestly and mine the interaction for durable
signal about *this specific user*.

Return ONLY a JSON object:
{
  "summary": "2-3 sentence account of what happened and how well it went",
  "what_worked": ["..."],
  "what_failed": ["..."],
  "user_preferences": [
     {"observation": "a durable, reusable fact about the user", "confidence": 0.0-1.0}
  ],
  "behavioral_adjustments": ["concrete change to make next time"],
  "lesson": "one sentence, imperative, that will be injected into future prompts"
}

Rules:
- Preferences must be about the USER's taste/workflow, not about this one task.
- Be specific. "User prefers X over Y" beats "user likes good output".
- If there is no real signal, return an empty preferences list. Do not invent."""


class DigitalTwinMemory:
    """Persistent, self-reinforcing long-term memory."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.store = store or VectorStore(
            self.settings.chroma_path, self.settings.twin_collection
        )

    async def remember(self, item: MemoryItem) -> str:
        vec = await self.embedder.embed_one(item.content)
        await self.store.add(
            ids=[item.id],
            documents=[item.content],
            embeddings=[vec],
            metadatas=[item.to_metadata()],
        )
        logger.info("twin: stored %s (%s)", item.kind.value, item.content[:60])
        return item.id

    async def remember_many(self, items: list[MemoryItem]) -> int:
        if not items:
            return 0
        vecs = await self.embedder.embed([i.content for i in items])
        return await self.store.add(
            ids=[i.id for i in items],
            documents=[i.content for i in items],
            embeddings=vecs,
            metadatas=[i.to_metadata() for i in items],
        )

    async def recall(
        self,
        query: str,
        *,
        k: int = 6,
        kinds: list[MemoryKind] | None = None,
        user_id: str = "default",
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        # NOTE: cosine similarity is in [-1, 1]. Defaulting the floor to 0.0
        # silently hides real memories that merely phrase things differently,
        # so the default is "no floor" and callers opt in to filtering.
        floor = -1.0 if min_score is None else min_score
        vec = await self.embedder.embed_one(query)
        where: dict[str, Any] = {"user_id": user_id}
        if kinds:
            where = {
                "$and": [
                    {"user_id": user_id},
                    {"kind": {"$in": [k_.value for k_ in kinds]}},
                ]
            }
        # Over-fetch, then re-rank by relevance * confidence * recency.
        raw = await self.store.query(embedding=vec, n_results=max(k * 3, 12), where=where)
        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in raw:
            if rec.score < floor:
                continue
            meta = rec.metadata
            confidence = float(meta.get("confidence", 0.5))
            importance = float(meta.get("importance", 0.5))
            age_days = max((now - float(meta.get("created_at", now))) / 86400.0, 0.0)
            recency = 0.5 ** (age_days / 30.0)  # 30-day half life
            final = rec.score * (0.55 + 0.45 * confidence) * (0.7 + 0.3 * importance)
            final *= 0.8 + 0.2 * recency
            scored.append(
                (
                    final,
                    {
                        "id": rec.id,
                        "content": rec.document,
                        "kind": meta.get("kind", "episode"),
                        "confidence": confidence,
                        "relevance": round(rec.score, 4),
                        "score": round(final, 4),
                        "created_at": meta.get("created_at"),
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    async def reinforce_preference(
        self, observation: str, confidence: float, user_id: str = "default"
    ) -> str:
        """Strengthen an existing belief if we've seen it before, else create it.

        This is what makes the Twin *converge*: repeated observations push
        confidence toward 1.0 rather than creating duplicate memories.
        """
        similar = await self.recall(
            observation, k=1, kinds=[MemoryKind.PREFERENCE], user_id=user_id
        )
        if similar and similar[0]["relevance"] >= 0.82:
            existing = similar[0]
            old = float(existing["confidence"])
            merged = min(0.99, old + (1.0 - old) * max(confidence, 0.3) * 0.5)
            item = MemoryItem(
                id=existing["id"],  # same id -> upsert, no duplicate
                content=existing["content"],
                kind=MemoryKind.PREFERENCE,
                user_id=user_id,
                confidence=merged,
                importance=0.8,
                source="reinforced",
                hits=1,
            )
            await self.remember(item)
            logger.info("twin: reinforced preference %.2f -> %.2f", old, merged)
            return item.id
        return await self.remember(
            MemoryItem(
                content=observation,
                kind=MemoryKind.PREFERENCE,
                user_id=user_id,
                confidence=confidence,
                importance=0.8,
                source="reflection",
            )
        )

    async def twin_context(self, query: str, *, user_id: str = "default", k: int = 6) -> str:
        """Render the Twin's beliefs as an injectable system-prompt block."""
        prefs = await self.recall(
            query, k=k, kinds=[MemoryKind.PREFERENCE, MemoryKind.FEEDBACK], user_id=user_id
        )
        lessons = await self.recall(
            query, k=k, kinds=[MemoryKind.LESSON, MemoryKind.PATTERN], user_id=user_id
        )
        if not prefs and not lessons:
            return ""
        lines = ["## Digital Twin context (learned from this user)"]
        if prefs:
            lines.append("\n### Known preferences")
            lines += [
                f"- {p['content']} (confidence {p['confidence']:.2f})" for p in prefs
            ]
        if lessons:
            lines.append("\n### Lessons from past runs")
            lines += [f"- {l['content']}" for l in lessons]
        lines.append(
            "\nAct in accordance with these. They are evidence about who the user is."
        )
        return "\n".join(lines)

    async def stats(self, user_id: str = "default") -> dict[str, Any]:
        total = await self.store.count()
        counts: dict[str, int] = {}
        for kind in MemoryKind:
            recs = await self.store.get_where(
                {"$and": [{"user_id": user_id}, {"kind": kind.value}]}, limit=1000
            )
            counts[kind.value] = len(recs)
        return {"total_memories": total, "by_kind": counts, "user_id": user_id}


class ReflectionLoop:
    """Post-task self-critique that writes back into the Twin."""

    def __init__(self, memory: DigitalTwinMemory, llm: LLMClient) -> None:
        self.memory = memory
        self.llm = llm

    async def reflect(
        self,
        *,
        task: str,
        outcome: str,
        success: bool,
        user_id: str = "default",
        extra_context: str = "",
    ) -> dict[str, Any]:
        prompt = (
            f"<TASK>{task}</TASK>\n\n"
            f"<OUTCOME success={str(success).lower()}>{outcome[:6000]}</OUTCOME>\n"
        )
        if extra_context:
            prompt += f"\n<CONTEXT>{extra_context[:2000]}</CONTEXT>\n"

        try:
            data = await self.llm.complete_json(
                [Message("system", REFLECTION_SYSTEM), Message("user", prompt)],
                role="reflection",
            )
        except Exception as exc:  # reflection must never break the main flow
            logger.warning("reflection failed: %s", exc)
            return {"error": str(exc), "stored": 0}

        # Persistence below is best-effort: a memory-layer failure must never
        # fail the task that just succeeded.
        try:
            stored = await self._persist_reflection(data, task, outcome, success, user_id)
        except Exception as exc:
            logger.warning("reflection persistence failed: %s", exc)
            return {**data, "stored": 0, "persist_error": str(exc)}

        logger.info("reflection: wrote %d memories", stored)
        return {**data, "stored": stored}

    async def _persist_reflection(
        self,
        data: dict[str, Any],
        task: str,
        outcome: str,
        success: bool,
        user_id: str,
    ) -> int:
        stored = 0
        lesson = (data.get("lesson") or "").strip()
        if lesson:
            await self.memory.remember(
                MemoryItem(
                    content=lesson,
                    kind=MemoryKind.LESSON,
                    user_id=user_id,
                    confidence=0.7 if success else 0.85,
                    importance=0.75,
                    source="reflection",
                    tags=["success" if success else "failure"],
                )
            )
            stored += 1

        for pref in data.get("user_preferences") or []:
            obs = (pref or {}).get("observation", "").strip()
            if not obs:
                continue
            await self.memory.reinforce_preference(
                obs, float(pref.get("confidence", 0.5)), user_id=user_id
            )
            stored += 1

        adjustments = data.get("behavioral_adjustments") or []
        if adjustments:
            await self.memory.remember(
                MemoryItem(
                    content="Behavioural adjustments: " + "; ".join(map(str, adjustments)),
                    kind=MemoryKind.PATTERN,
                    user_id=user_id,
                    confidence=0.65,
                    importance=0.7,
                    source="reflection",
                )
            )
            stored += 1

        await self.memory.remember(
            MemoryItem(
                content=f"Task: {task}\nResult: {data.get('summary', outcome[:400])}",
                kind=MemoryKind.EPISODE,
                user_id=user_id,
                confidence=0.9,
                importance=0.5 if success else 0.8,
                source="episode",
                tags=["success" if success else "failure"],
            )
        )
        stored += 1
        return stored


def summarize_reflection(data: dict[str, Any]) -> str:
    """Human-readable one-liner for logs and the UI."""
    if "error" in data:
        return f"Reflection unavailable: {data['error']}"
    return json.dumps(
        {
            "lesson": data.get("lesson"),
            "preferences_learned": len(data.get("user_preferences") or []),
            "memories_written": data.get("stored", 0),
        }
    )
