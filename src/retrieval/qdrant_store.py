"""Qdrant-backed implementation of :class:`~src.retrieval.similar_tickets.TicketStore`."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from src.api.schemas import SimilarTicket
from src.errors import RetrievalError
from src.observability import trace_logger
from src.retrieval.encoders import Encoder
from src.retrieval.similar_tickets import TicketDoc, to_similar_ticket

#: Namespace used to derive stable point UUIDs from ticket IDs.
_NS = uuid.UUID("6f1c2b4e-6c0a-4b8e-9a3e-1d2f3a4b5c6d")


def point_id_for(ticket_id: str) -> str:
    """Deterministic UUID5 for a ticket so re-indexing upserts instead of duplicating."""
    return str(uuid.uuid5(_NS, ticket_id))


class QdrantTicketStore:
    """Similar-ticket store backed by a Qdrant collection.

    Args:
        encoder: Encoder for both indexing and queries.
        client: An :class:`qdrant_client.AsyncQdrantClient`.
        collection: Collection name.
        batch_size: Points per upsert request.
    """

    name = "qdrant"

    def __init__(
        self,
        encoder: Encoder,
        client: AsyncQdrantClient,
        collection: str,
        batch_size: int = 64,
    ) -> None:
        self.encoder = encoder
        self._client = client
        self.collection = collection
        self.batch_size = batch_size
        self._ready = False

    async def ensure_collection(self, dim: int) -> None:
        """Create the collection with cosine distance if it does not exist."""
        if self._ready:
            return
        try:
            exists = await self._client.collection_exists(self.collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
                trace_logger().info("created qdrant collection {} dim={}", self.collection, dim)
        except Exception as exc:  # qdrant raises several unrelated types
            raise RetrievalError(f"qdrant collection setup failed: {exc}") from exc
        self._ready = True

    async def index(self, tickets: list[TicketDoc]) -> int:
        """Embed ``tickets`` and upsert them in batches."""
        if not tickets:
            return 0
        vectors = await self.encoder.encode([str(t["body"]) for t in tickets])
        await self.ensure_collection(len(vectors[0]))
        points = [
            qm.PointStruct(
                id=point_id_for(str(t.get("ticket_id", t.get("id", i)))),
                vector=vec,
                payload={
                    "ticket_id": str(t.get("ticket_id", t.get("id", i))),
                    "intent": str(t.get("intent", "")),
                    "body": str(t.get("body", "")),
                    "resolution": str(t.get("resolution", "")),
                },
            )
            for i, (t, vec) in enumerate(zip(tickets, vectors, strict=True))
        ]
        try:
            for start in range(0, len(points), self.batch_size):
                await self._client.upsert(
                    collection_name=self.collection, points=points[start : start + self.batch_size]
                )
        except Exception as exc:
            raise RetrievalError(f"qdrant upsert failed: {exc}") from exc
        return len(points)

    async def search(self, query: str, k: int = 5) -> list[SimilarTicket]:
        """Return the ``k`` nearest tickets by cosine similarity."""
        if k <= 0:
            return []
        [vector] = await self.encoder.encode([query])
        await self.ensure_collection(len(vector))
        try:
            hits: list[Any] = await self._client.search(
                collection_name=self.collection, query_vector=vector, limit=k, with_payload=True
            )
        except Exception as exc:
            raise RetrievalError(f"qdrant search failed: {exc}") from exc
        return [to_similar_ticket(dict(h.payload or {}), float(h.score)) for h in hits]

    async def count(self) -> int:
        """Exact number of points in the collection (0 if it does not exist)."""
        try:
            if not await self._client.collection_exists(self.collection):
                return 0
            result = await self._client.count(collection_name=self.collection, exact=True)
        except Exception as exc:
            raise RetrievalError(f"qdrant count failed: {exc}") from exc
        return int(result.count)


def build_qdrant_store(encoder: Encoder, url: str, collection: str) -> QdrantTicketStore:
    """Connect to Qdrant at ``url`` and return a store over ``collection``."""
    client = AsyncQdrantClient(url=url, timeout=10)
    return QdrantTicketStore(encoder=encoder, client=client, collection=collection)
