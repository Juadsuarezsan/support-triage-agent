"""Similar-ticket retriever — finds past resolved tickets by semantic similarity.

Supports Qdrant (production) and in-memory cosine over sentence-transformer
embeddings (offline / tests).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol, Sequence

from loguru import logger

from src.api.schemas import SimilarTicket


class Encoder(Protocol):
    async def encode(self, texts: list[str]) -> list[list[float]]: ...


@lru_cache(maxsize=1)
def _hf_encoder():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer all-mpnet-base-v2 (~420 MB)")
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


class HFEncoder:
    async def encode(self, texts: list[str]) -> list[list[float]]:
        return _hf_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


class InMemoryTicketStore:
    """In-memory similar-ticket store for tests + offline demos."""

    def __init__(self, encoder: Encoder | None = None) -> None:
        self.encoder = encoder or HFEncoder()
        self._tickets: list[dict] = []
        self._embeds: list[list[float]] = []

    async def index(self, tickets: list[dict]) -> int:
        bodies = [t["body"] for t in tickets]
        if not bodies:
            return 0
        embeds = await self.encoder.encode(bodies)
        self._tickets.extend(tickets)
        self._embeds.extend(embeds)
        return len(tickets)

    async def search(self, query: str, k: int = 5) -> list[SimilarTicket]:
        if not self._tickets:
            return []
        [q_emb] = await self.encoder.encode([query])
        scored = sorted(
            (
                (cosine(q_emb, e), t)
                for e, t in zip(self._embeds, self._tickets)
            ),
            key=lambda x: x[0], reverse=True,
        )[:k]
        out: list[SimilarTicket] = []
        for sim, t in scored:
            out.append(SimilarTicket(
                ticket_id=str(t.get("ticket_id", t.get("id", "?"))),
                intent=str(t.get("intent", "")),
                body_snippet=str(t["body"])[:200],
                resolution_snippet=str(t.get("resolution", ""))[:200],
                similarity=float(sim),
            ))
        return out

    def count(self) -> int:
        return len(self._tickets)


class DeterministicEncoder:
    """Hash-bag-of-tokens encoder. No model download — for unit tests only."""

    DIM = 256

    async def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import re
        TOKEN = re.compile(r"[a-zA-Z]+")
        out = []
        for text in texts:
            tokens = TOKEN.findall(text.lower()) or ["<empty>"]
            acc = [0.0] * self.DIM
            for tok in tokens:
                d = hashlib.sha256(tok.encode()).digest()
                raw = (d * ((self.DIM // len(d)) + 1))[: self.DIM]
                for i, b in enumerate(raw):
                    acc[i] += (b - 128) / 128.0
            out.append([x / len(tokens) for x in acc])
        return out
