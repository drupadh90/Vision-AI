"""Module 2 (part 2) — the Skill Vector DB and Skill Activation.

Ingested videos become *skills*: retrievable, timestamped procedural knowledge.
Skill Activation is the payoff — before acting, Vision AI asks "have I watched
something that teaches this?" and, if so, injects that knowledge into the
system prompt so it performs the task like someone who just did the tutorial.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..core.embeddings import BaseEmbedder, build_embedder
from ..memory.vector_store import VectorStore
from .youtube_ingest import (
    IngestError,
    Transcript,
    VideoMetadata,
    YouTubeIngestor,
    chunk_transcript,
    extract_video_id,
    format_timestamp,
    normalize_url,
)

logger = logging.getLogger("vision.skills")


@dataclass
class SkillChunk:
    text: str
    skill_id: str
    video_id: str
    title: str
    start: float = 0.0
    end: float = 0.0
    index: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def deep_link(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(self.start)}s"


@dataclass
class SkillMatch:
    text: str
    title: str
    video_id: str
    score: float
    start: float
    end: float
    skill_id: str

    def citation(self) -> str:
        return (
            f"[{self.title} @ {format_timestamp(self.start)}]"
            f"(https://www.youtube.com/watch?v={self.video_id}&t={int(self.start)}s)"
        )


@dataclass
class IngestReport:
    skill_id: str
    video_id: str
    title: str
    channel: str
    duration: int
    chunks: int
    characters: int
    method: str
    url: str
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SkillLibrary:
    """Persistent store of everything Vision AI has learned from video."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
        ingestor: YouTubeIngestor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.store = store or VectorStore(
            self.settings.chroma_path, self.settings.skill_collection
        )
        self.ingestor = ingestor or YouTubeIngestor(self.settings)

    # -- ingestion -------------------------------------------------------
    async def learn_from_youtube(
        self, url: str, *, skill_name: str | None = None, force: bool = False
    ) -> IngestReport:
        started = time.monotonic()
        video_id = extract_video_id(url)
        canonical = normalize_url(url)

        if not force:
            existing = await self.store.get_where({"video_id": video_id}, limit=1)
            if existing:
                meta = existing[0].metadata
                logger.info("skill already learned: %s", video_id)
                return IngestReport(
                    skill_id=str(meta.get("skill_id", video_id)),
                    video_id=video_id,
                    title=str(meta.get("title", "")),
                    channel=str(meta.get("channel", "")),
                    duration=int(meta.get("duration", 0) or 0),
                    chunks=len(await self.store.get_where({"video_id": video_id}, limit=10000)),
                    characters=0,
                    method="cached",
                    url=canonical,
                    elapsed_s=time.monotonic() - started,
                )

        meta_obj: VideoMetadata = await self.ingestor.fetch_metadata(canonical)
        transcript: Transcript = await self.ingestor.get_transcript(canonical, video_id)
        if not transcript.text.strip():
            raise IngestError(f"Empty transcript for {canonical}")

        raw_chunks = chunk_transcript(
            transcript,
            max_chars=self.settings.skill_chunk_chars,
            overlap=self.settings.skill_chunk_overlap,
        )
        skill_id = uuid.uuid4().hex[:12]
        title = skill_name or meta_obj.title

        chunks = [
            SkillChunk(
                text=c["text"],
                skill_id=skill_id,
                video_id=video_id,
                title=title,
                start=c.get("start", 0.0),
                end=c.get("end", 0.0),
                index=i,
            )
            for i, c in enumerate(raw_chunks)
        ]
        await self._persist(chunks, meta_obj, transcript.method, title, canonical)

        report = IngestReport(
            skill_id=skill_id,
            video_id=video_id,
            title=title,
            channel=meta_obj.channel,
            duration=meta_obj.duration,
            chunks=len(chunks),
            characters=len(transcript.text),
            method=transcript.method,
            url=canonical,
            elapsed_s=time.monotonic() - started,
        )
        logger.info("learned skill '%s' (%d chunks, %s)", title, len(chunks), transcript.method)
        return report

    async def _persist(
        self,
        chunks: list[SkillChunk],
        meta: VideoMetadata,
        method: str,
        title: str,
        url: str,
    ) -> None:
        if not chunks:
            return
        # Prefixing the title sharpens retrieval: the topic is in every vector.
        vectors = await self.embedder.embed([f"{title}. {c.text}" for c in chunks])
        await self.store.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "skill_id": c.skill_id,
                    "video_id": c.video_id,
                    "title": title,
                    "channel": meta.channel,
                    "duration": meta.duration,
                    "start": c.start,
                    "end": c.end,
                    "index": c.index,
                    "method": method,
                    "url": url,
                    "learned_at": time.time(),
                }
                for c in chunks
            ],
        )

    async def add_text_skill(self, title: str, text: str, source: str = "manual") -> IngestReport:
        """Teach a skill from raw text (docs, notes) — same retrieval path."""
        skill_id = uuid.uuid4().hex[:12]
        raw = chunk_transcript(
            Transcript(text=text, method=source),
            max_chars=self.settings.skill_chunk_chars,
            overlap=self.settings.skill_chunk_overlap,
        )
        chunks = [
            SkillChunk(
                text=c["text"], skill_id=skill_id, video_id=f"text:{skill_id}",
                title=title, index=i,
            )
            for i, c in enumerate(raw)
        ]
        await self._persist(
            chunks, VideoMetadata(video_id=f"text:{skill_id}", title=title), source, title, source
        )
        return IngestReport(
            skill_id=skill_id, video_id=f"text:{skill_id}", title=title, channel="",
            duration=0, chunks=len(chunks), characters=len(text), method=source, url=source,
        )

    # -- activation ------------------------------------------------------
    async def search(self, query: str, *, k: int = 5, min_score: float | None = None) -> list[SkillMatch]:
        threshold = self.settings.skill_match_threshold if min_score is None else min_score
        vec = await self.embedder.embed_one(query)
        records = await self.store.query(embedding=vec, n_results=k * 2)
        matches = [
            SkillMatch(
                text=r.document,
                title=str(r.metadata.get("title", "Unknown skill")),
                video_id=str(r.metadata.get("video_id", "")),
                score=r.score,
                start=float(r.metadata.get("start", 0) or 0),
                end=float(r.metadata.get("end", 0) or 0),
                skill_id=str(r.metadata.get("skill_id", "")),
            )
            for r in records
            if r.score >= threshold
        ]
        return matches[:k]

    async def activate(self, task: str, *, k: int = 4) -> tuple[str, list[SkillMatch]]:
        """Skill Activation: render retrieved know-how as a prompt injection."""
        matches = await self.search(task, k=k)
        if not matches:
            return "", []
        lines = [
            "## Activated skills (learned from video)",
            "",
            "You have previously watched and internalised the following material. "
            "Apply it as though you had just practised it. Cite the timestamp when "
            "you rely on a specific step.",
            "",
        ]
        for i, m in enumerate(matches, 1):
            lines.append(f"### Skill {i}: {m.title} (relevance {m.score:.2f})")
            lines.append(f"Source: {m.citation()}")
            lines.append(m.text.strip())
            lines.append("")
        return "\n".join(lines), matches

    # -- inventory -------------------------------------------------------
    async def list_skills(self) -> list[dict[str, Any]]:
        records = await self.store.get_where({"index": 0}, limit=500)
        skills = [
            {
                "skill_id": r.metadata.get("skill_id"),
                "title": r.metadata.get("title"),
                "video_id": r.metadata.get("video_id"),
                "channel": r.metadata.get("channel"),
                "duration": r.metadata.get("duration"),
                "method": r.metadata.get("method"),
                "url": r.metadata.get("url"),
                "learned_at": r.metadata.get("learned_at"),
                "preview": r.document[:200],
            }
            for r in records
        ]
        skills.sort(key=lambda s: s.get("learned_at") or 0, reverse=True)
        return skills

    async def forget(self, skill_id: str) -> None:
        await self.store.delete(where={"skill_id": skill_id})

    async def stats(self) -> dict[str, Any]:
        return {
            "total_chunks": await self.store.count(),
            "total_skills": len(await self.list_skills()),
        }
