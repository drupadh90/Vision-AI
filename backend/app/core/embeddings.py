"""Embedding backends for the Digital Twin and Skill vector stores.

Two implementations:

* `OpenAIEmbedder`  - production quality, needs `OPENAI_API_KEY`.
* `HashEmbedder`    - deterministic, dependency-free, offline. Hashed bag of
  character n-grams + word unigrams, L2-normalised. Not semantically great,
  but it is stable, fast, needs no model download, and makes retrieval tests
  reproducible. It is the default so a fresh clone just works.

Both return L2-normalised vectors, so cosine distance in Chroma behaves.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod

import httpx

from ..config import Settings, get_settings

logger = logging.getLogger("vision.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BaseEmbedder(ABC):
    name: str = "base"
    dim: int = 1536

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def aclose(self) -> None:
        return None


# Damping these stops queries like "how to bake bread" from matching a video
# about editing purely on shared function words. Measured effect on the
# relevant-vs-irrelevant separation gap: -0.09 -> +0.08.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those is are was were be
    been being am do does did doing have has had having i you he she it we they
    me him her them my your his its our their to of in on at by for with about
    into over after under from up down out off again further once here there
    when where why how all any both each few more most other some such no nor
    not only own same so too very can will just should now what which who whom
    get got make makes made use used using want need like""".split()
)
_STOPWORD_WEIGHT = 0.15
_BIGRAM_WEIGHT = 0.5
_CHARGRAM_WEIGHT = 0.10


class HashEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-features embedder.

    Features: sublinear-weighted word unigrams (stopwords damped), bigrams for
    word order, and word-internal char 4-grams for typo tolerance.
    """

    name = "hash"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _TOKEN_RE.findall(text.lower())
        if not words:
            return vec
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1

        def bump(feature: str, weight: float) -> None:
            h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * weight

        for token, count in counts.items():
            tf = 1.0 + math.log(count)  # sublinear: long docs don't dominate
            if token in _STOPWORDS:
                tf *= _STOPWORD_WEIGHT
            bump(f"w:{token}", tf)
            if len(token) >= 5 and token not in _STOPWORDS:
                for i in range(len(token) - 3):
                    bump(f"c:{token[i : i + 4]}", _CHARGRAM_WEIGHT)

        for a, b in zip(words, words[1:]):
            if a in _STOPWORDS and b in _STOPWORDS:
                continue
            bump(f"b:{a}_{b}", _BIGRAM_WEIGHT)

        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dim = settings.embedding_dim
        base = settings.openai_base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(
            base_url=base.rstrip("/"),
            timeout=settings.llm_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):  # stay under request-size limits
            batch = texts[i : i + 64]
            resp = await self._client.post(
                "/embeddings", json={"model": self.settings.embedding_model, "input": batch}
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            out.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


def build_embedder(settings: Settings | None = None) -> BaseEmbedder:
    s = settings or get_settings()
    resolved = s.effective_embedding_provider()
    if resolved != s.embedding_provider:
        logger.warning("Embedding provider downgraded to 'hash' (no OPENAI_API_KEY).")
    return OpenAIEmbedder(s) if resolved == "openai" else HashEmbedder()
