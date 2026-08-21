"""Thin async wrapper over ChromaDB used by both the Twin and the Skill DB.

Design notes
------------
* Chroma's Python client is synchronous; every call is pushed to a worker
  thread so it never blocks the FastAPI event loop.
* We always pass our own embeddings (`embedding_function=None`) so Chroma never
  tries to download an ONNX model at import time — critical for offline boot.
* Cosine space, because both embedders emit L2-normalised vectors.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("vision.vectorstore")


@dataclass
class VectorRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float = 0.0  # cosine similarity in [-1, 1]; higher is better


def _sanitize(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values."""
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = ", ".join(str(x) for x in v)
        else:
            clean[k] = str(v)
    return clean


class VectorStore:
    """One Chroma collection, async-safe."""

    def __init__(self, path: Path, collection: str) -> None:
        self.path = Path(path)
        self.collection_name = collection
        self._collection: Any | None = None
        self._lock = asyncio.Lock()

    def _ensure_sync(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb  # imported lazily: keeps startup fast

        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.path))
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma collection '%s' ready at %s", self.collection_name, self.path)
        return self._collection

    async def _collection_handle(self) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._ensure_sync)

    async def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        col = await self._collection_handle()
        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=[_sanitize(m) for m in metadatas],
        )
        return len(ids)

    async def query(
        self,
        *,
        embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        col = await self._collection_handle()
        total = await asyncio.to_thread(col.count)
        if total == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(n_results, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = await asyncio.to_thread(col.query, **kwargs)

        out: list[VectorRecord] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, _id in enumerate(ids):
            distance = dists[i] if i < len(dists) else 1.0
            out.append(
                VectorRecord(
                    id=_id,
                    document=docs[i] if i < len(docs) else "",
                    metadata=dict(metas[i] or {}) if i < len(metas) else {},
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return out

    async def get_where(self, where: dict[str, Any], limit: int = 100) -> list[VectorRecord]:
        col = await self._collection_handle()
        res = await asyncio.to_thread(
            col.get, where=where, limit=limit, include=["documents", "metadatas"]
        )
        return [
            VectorRecord(
                id=_id,
                document=(res.get("documents") or [])[i] or "",
                metadata=dict((res.get("metadatas") or [])[i] or {}),
            )
            for i, _id in enumerate(res.get("ids") or [])
        ]

    async def count(self) -> int:
        col = await self._collection_handle()
        return int(await asyncio.to_thread(col.count))

    async def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        col = await self._collection_handle()
        kwargs: dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if kwargs:
            await asyncio.to_thread(col.delete, **kwargs)
