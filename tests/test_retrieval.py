"""Encoders, in-memory store and the Qdrant store (mocked client)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.errors import RetrievalError
from src.retrieval.encoders import CachedEncoder, DeterministicEncoder, HFEncoder, build_encoder
from src.retrieval.qdrant_store import QdrantTicketStore, point_id_for
from src.retrieval.similar_tickets import InMemoryTicketStore, cosine

SEEDS: list[dict[str, Any]] = [
    {
        "ticket_id": "t-1",
        "intent": "get_refund",
        "body": "refund broken item",
        "resolution": "Refunded.",
    },
    {
        "ticket_id": "t-2",
        "intent": "track_order",
        "body": "where is my order",
        "resolution": "Tracked.",
    },
    {
        "ticket_id": "t-3",
        "intent": "cancel_order",
        "body": "cancel my order now",
        "resolution": "Cancelled.",
    },
]


async def test_deterministic_encoder_is_normalised_and_stable() -> None:
    enc = DeterministicEncoder()
    [a, b, c] = await enc.encode(["refund please", "refund please", ""])
    assert a == b and len(a) == 256
    assert cosine(a, a) == pytest.approx(1.0)
    assert sum(x * x for x in c) == pytest.approx(1.0)


async def test_cached_encoder_batches_only_misses() -> None:
    inner = AsyncMock()
    inner.name = "inner"
    inner.encode.side_effect = lambda texts: [[float(len(t))] for t in texts]
    enc = CachedEncoder(inner, max_items=2)
    assert await enc.encode(["a", "bb", "a"]) == [[1.0], [2.0], [1.0]]
    inner.encode.assert_awaited_once_with(["a", "bb"])
    assert await enc.encode(["bb", "ccc"]) == [[2.0], [3.0]]
    assert inner.encode.await_args_list[-1].args == (["ccc"],)
    assert (enc.hits, enc.misses) == (1, 4)  # third "a" is a deduplicated miss
    await enc.encode(["a"])  # evicted (capacity 2) -> miss again
    assert enc.misses == 5


async def test_hf_encoder_lazy_factory() -> None:
    model = MagicMock()
    model.encode.return_value = [[0.6, 0.8]]
    factory = MagicMock(return_value=model)
    enc = HFEncoder(model_factory=factory)
    assert await enc.encode(["x"]) == [[0.6, 0.8]]
    factory.assert_called_once_with("sentence-transformers/all-mpnet-base-v2")


def test_hf_encoder_import_error_without_ml_extra() -> None:
    with pytest.raises(ImportError):
        HFEncoder()._load()


def test_build_encoder() -> None:
    assert isinstance(build_encoder("deterministic"), CachedEncoder)
    assert build_encoder("hf").name.startswith("cached(")
    with pytest.raises(ValueError):
        build_encoder("nope")


async def test_in_memory_store_ranks_by_similarity() -> None:
    store = InMemoryTicketStore(DeterministicEncoder())
    assert await store.index([]) == 0
    assert await store.index(SEEDS) == 3
    assert await store.count() == 3
    hits = await store.search("I want a refund for my broken item", k=2)
    assert [h.ticket_id for h in hits][0] == "t-1"
    assert len(hits) == 2 and hits[0].similarity >= hits[1].similarity
    assert await store.search("anything", k=0) == []
    assert await InMemoryTicketStore(DeterministicEncoder()).search("x") == []


def _qdrant_client(exists: bool = True) -> AsyncMock:
    client = AsyncMock()
    client.collection_exists.return_value = exists
    hit = MagicMock()
    hit.score = 0.87
    hit.payload = {"ticket_id": "t-2", "intent": "track_order", "body": "where", "resolution": "R"}
    client.search.return_value = [hit]
    client.count.return_value = MagicMock(count=3)
    return client


async def test_qdrant_store_creates_collection_and_upserts_in_batches() -> None:
    client = _qdrant_client(exists=False)
    store = QdrantTicketStore(DeterministicEncoder(), client, "tickets", batch_size=2)
    assert await store.index(SEEDS) == 3
    client.create_collection.assert_awaited_once()
    assert client.create_collection.await_args.kwargs["vectors_config"].size == 256
    assert client.upsert.await_count == 2  # 2 + 1
    ids = [p.id for call in client.upsert.await_args_list for p in call.kwargs["points"]]
    assert ids == [point_id_for("t-1"), point_id_for("t-2"), point_id_for("t-3")]


async def test_qdrant_store_search_maps_payload() -> None:
    store = QdrantTicketStore(DeterministicEncoder(), _qdrant_client(), "tickets")
    [hit] = await store.search("where is my order", k=1)
    assert hit.ticket_id == "t-2" and hit.similarity == pytest.approx(0.87)
    assert await store.search("x", k=0) == []
    assert await store.count() == 3


async def test_qdrant_store_count_missing_collection() -> None:
    store = QdrantTicketStore(DeterministicEncoder(), _qdrant_client(exists=False), "tickets")
    assert await store.count() == 0


async def test_qdrant_errors_are_wrapped() -> None:
    client = _qdrant_client()
    client.upsert.side_effect = RuntimeError("boom")
    client.search.side_effect = RuntimeError("boom")
    client.count.side_effect = RuntimeError("boom")
    store = QdrantTicketStore(DeterministicEncoder(), client, "tickets")
    with pytest.raises(RetrievalError):
        await store.index(SEEDS)
    with pytest.raises(RetrievalError):
        await store.search("x")
    with pytest.raises(RetrievalError):
        await store.count()
    bad = _qdrant_client(exists=False)
    bad.create_collection.side_effect = RuntimeError("no")
    with pytest.raises(RetrievalError):
        await QdrantTicketStore(DeterministicEncoder(), bad, "c").ensure_collection(4)


def test_point_id_is_stable_uuid() -> None:
    assert point_id_for("abc") == point_id_for("abc") != point_id_for("abd")
