"""Similar-ticket retrieval: protocol, in-memory store and helpers.

The production store lives in :mod:`src.retrieval.qdrant_store`; this module
holds the :class:`TicketStore` protocol both implementations satisfy and the
in-memory cosine store used by tests, CI and the offline evaluation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from src.api.schemas import SimilarTicket
from src.retrieval.encoders import Encoder

#: A resolved ticket as stored on disk (``data/eval/seed_tickets.jsonl``).
TicketDoc = dict[str, Any]


class TicketStore(Protocol):
    """Index resolved tickets and retrieve the most similar ones."""

    name: str

    async def index(self, tickets: list[TicketDoc]) -> int:
        """Embed and store ``tickets``; return how many were indexed."""
        ...

    async def search(self, query: str, k: int = 5) -> list[SimilarTicket]:
        """Return up to ``k`` tickets most similar to ``query``."""
        ...

    async def count(self) -> int:
        """Number of indexed tickets."""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 when either is null)."""
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def to_similar_ticket(doc: TicketDoc, similarity: float) -> SimilarTicket:
    """Project a stored ticket document onto the API schema."""
    return SimilarTicket(
        ticket_id=str(doc.get("ticket_id", doc.get("id", "?"))),
        intent=str(doc.get("intent", "")),
        body_snippet=str(doc.get("body", ""))[:200],
        resolution_snippet=str(doc.get("resolution", ""))[:200],
        similarity=round(float(similarity), 6),
    )


def load_jsonl(path: Path) -> list[TicketDoc]:
    """Read a JSON-lines file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class InMemoryTicketStore:
    """Cosine search over embeddings held in process memory.

    Args:
        encoder: Encoder used for both indexing and queries.
    """

    name = "memory"

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self._tickets: list[TicketDoc] = []
        self._embeds: list[list[float]] = []

    async def index(self, tickets: list[TicketDoc]) -> int:
        """Embed ``tickets`` in one batch and append them to the index."""
        bodies = [str(t["body"]) for t in tickets]
        if not bodies:
            return 0
        embeds = await self.encoder.encode(bodies)
        self._tickets.extend(tickets)
        self._embeds.extend(embeds)
        return len(tickets)

    async def search(self, query: str, k: int = 5) -> list[SimilarTicket]:
        """Rank every indexed ticket by cosine similarity to ``query``."""
        if not self._tickets or k <= 0:
            return []
        [q_emb] = await self.encoder.encode([query])
        scored = sorted(
            ((cosine(q_emb, e), t) for e, t in zip(self._embeds, self._tickets, strict=True)),
            key=lambda pair: pair[0],
            reverse=True,
        )[:k]
        return [to_similar_ticket(t, sim) for sim, t in scored]

    async def count(self) -> int:
        """Number of indexed tickets."""
        return len(self._tickets)
