"""Audit stores: in-memory ring buffer and Postgres with a fake pool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.schemas import AuditRecord
from src.errors import PersistenceError
from src.persistence.audit import InMemoryAuditStore, PostgresAuditStore


def _record(i: int) -> AuditRecord:
    return AuditRecord(
        trace_id=f"trace-{i}",
        ticket_id=f"t-{i}",
        channel="chat",
        body="hello",
        intent="track_order",
        intent_confidence=0.7,
        priority="P3",
        sentiment="neu",
        urgency_score=0.2,
        decision="suggest",
        confidence=0.66,
        rationale="r",
        draft_response="d",
        similar_ticket_ids=["a"],
        llm_used=False,
        latency_ms=5,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        created_at=datetime(2026, 9, 29, 12, 0, i, tzinfo=UTC),
    )


async def test_in_memory_store_keeps_newest_first_and_caps() -> None:
    store = InMemoryAuditStore(capacity=2)
    for i in range(3):
        await store.record(_record(i))
    rows = await store.recent()
    assert [r.trace_id for r in rows] == ["trace-2", "trace-1"]
    assert await store.recent(limit=1) == rows[:1]
    await store.close()


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.rows = rows or []
        self.fail = fail

    async def execute(self, sql: str, params: Any = None) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.executed.append((sql, params))

    @asynccontextmanager
    async def transaction(self):  # type: ignore[no-untyped-def]
        yield

    def cursor(self, row_factory: Any = None):  # type: ignore[no-untyped-def]
        conn = self
        cur = MagicMock()

        async def _execute(sql: str, params: Any) -> None:
            await conn.execute(sql, params)

        cur.execute = AsyncMock(side_effect=_execute)
        cur.fetchall = AsyncMock(return_value=conn.rows)

        @asynccontextmanager
        async def _cm():  # type: ignore[no-untyped-def]
            yield cur

        return _cm()


def _pool(conn: _FakeConn) -> MagicMock:
    pool = MagicMock()

    @asynccontextmanager
    async def _connection():  # type: ignore[no-untyped-def]
        yield conn

    pool.connection = _connection
    pool.close = AsyncMock()
    return pool


async def test_postgres_record_writes_ticket_and_audit_in_one_transaction() -> None:
    conn = _FakeConn()
    store = PostgresAuditStore(_pool(conn))
    await store.ensure_schema()
    await store.record(_record(1))
    sqls = [s for s, _ in conn.executed]
    assert "CREATE TABLE IF NOT EXISTS tickets" in sqls[0]
    assert "INSERT INTO tickets" in sqls[1] and "INSERT INTO triage_audit" in sqls[2]
    assert conn.executed[2][1]["trace_id"] == "trace-1"


async def test_postgres_recent_maps_rows() -> None:
    conn = _FakeConn(rows=[_record(7).model_dump()])
    store = PostgresAuditStore(_pool(conn))
    [row] = await store.recent(limit=5)
    assert row.trace_id == "trace-7"
    assert conn.executed[0][1] == {"limit": 5}
    await store.close()


async def test_postgres_errors_are_wrapped() -> None:
    store = PostgresAuditStore(_pool(_FakeConn(fail=True)))
    with pytest.raises(PersistenceError):
        await store.ensure_schema()
    with pytest.raises(PersistenceError):
        await store.record(_record(1))
    with pytest.raises(PersistenceError):
        await store.recent()
