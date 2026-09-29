"""Audit logger: every triage decision is persisted with its scores and cost.

:class:`InMemoryAuditStore` keeps the last N records in process memory (tests,
demos). :class:`PostgresAuditStore` writes a ``tickets`` row and a
``triage_audit`` row per request through a ``psycopg`` connection pool.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.api.schemas import AuditRecord
from src.errors import PersistenceError
from src.observability import trace_logger


class AuditStore(Protocol):
    """Append-only store of :class:`~src.api.schemas.AuditRecord`."""

    name: str

    async def record(self, entry: AuditRecord) -> None:
        """Persist one record."""
        ...

    async def recent(self, limit: int = 100) -> list[AuditRecord]:
        """Return the newest ``limit`` records, newest first."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


class InMemoryAuditStore:
    """Ring buffer of the most recent records.

    Args:
        capacity: Maximum records retained.
    """

    name = "memory"

    def __init__(self, capacity: int = 1000) -> None:
        self._rows: deque[AuditRecord] = deque(maxlen=capacity)

    async def record(self, entry: AuditRecord) -> None:
        """Append ``entry`` (oldest entry is dropped at capacity)."""
        self._rows.append(entry)

    async def recent(self, limit: int = 100) -> list[AuditRecord]:
        """Newest ``limit`` entries, newest first."""
        return list(reversed(self._rows))[: max(0, limit)]

    async def close(self) -> None:
        """No resources to release."""
        return None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      TEXT PRIMARY KEY,
    channel        TEXT NOT NULL,
    body           TEXT NOT NULL,
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS triage_audit (
    trace_id           TEXT PRIMARY KEY,
    ticket_id          TEXT NOT NULL REFERENCES tickets(ticket_id),
    intent             TEXT NOT NULL,
    intent_confidence  DOUBLE PRECISION NOT NULL,
    priority           TEXT NOT NULL,
    sentiment          TEXT NOT NULL,
    urgency_score      DOUBLE PRECISION NOT NULL,
    decision           TEXT NOT NULL,
    confidence         DOUBLE PRECISION NOT NULL,
    rationale          TEXT NOT NULL,
    draft_response     TEXT,
    similar_ticket_ids TEXT[] NOT NULL DEFAULT '{}',
    llm_used           BOOLEAN NOT NULL,
    latency_ms         INTEGER NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    cost_usd           DOUBLE PRECISION NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS triage_audit_created_at_idx ON triage_audit (created_at DESC);
"""

_UPSERT_TICKET = """
INSERT INTO tickets (ticket_id, channel, body)
VALUES (%(ticket_id)s, %(channel)s, %(body)s)
ON CONFLICT (ticket_id) DO UPDATE SET last_seen_at = now(), body = EXCLUDED.body
"""

_INSERT_AUDIT = """
INSERT INTO triage_audit (
    trace_id, ticket_id, intent, intent_confidence, priority, sentiment, urgency_score,
    decision, confidence, rationale, draft_response, similar_ticket_ids, llm_used,
    latency_ms, input_tokens, output_tokens, cost_usd, created_at
) VALUES (
    %(trace_id)s, %(ticket_id)s, %(intent)s, %(intent_confidence)s, %(priority)s, %(sentiment)s,
    %(urgency_score)s, %(decision)s, %(confidence)s, %(rationale)s, %(draft_response)s,
    %(similar_ticket_ids)s, %(llm_used)s, %(latency_ms)s, %(input_tokens)s, %(output_tokens)s,
    %(cost_usd)s, %(created_at)s
)
"""

_SELECT_RECENT = """
SELECT a.trace_id, a.ticket_id, t.channel, t.body, a.intent, a.intent_confidence, a.priority,
       a.sentiment, a.urgency_score, a.decision, a.confidence, a.rationale, a.draft_response,
       a.similar_ticket_ids, a.llm_used, a.latency_ms, a.input_tokens, a.output_tokens,
       a.cost_usd, a.created_at
FROM triage_audit a JOIN tickets t USING (ticket_id)
ORDER BY a.created_at DESC
LIMIT %(limit)s
"""


class PostgresAuditStore:
    """Audit store on PostgreSQL through an async ``psycopg`` pool.

    Args:
        pool: An opened :class:`psycopg_pool.AsyncConnectionPool`.
    """

    name = "postgres"

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        """Create the tables if they do not exist."""
        try:
            async with self._pool.connection() as conn:
                await conn.execute(SCHEMA_SQL)
        except Exception as exc:  # psycopg raises a wide family of errors
            raise PersistenceError(f"schema setup failed: {exc}") from exc

    async def record(self, entry: AuditRecord) -> None:
        """Insert the ticket (upsert) and the audit row in one transaction."""
        params = entry.model_dump()
        try:
            async with self._pool.connection() as conn, conn.transaction():
                await conn.execute(_UPSERT_TICKET, params)
                await conn.execute(_INSERT_AUDIT, params)
        except Exception as exc:
            raise PersistenceError(f"audit insert failed: {exc}") from exc
        trace_logger().debug("audit row persisted trace_id={}", entry.trace_id)

    async def recent(self, limit: int = 100) -> list[AuditRecord]:
        """Fetch the newest ``limit`` audit rows joined with their tickets."""
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(_SELECT_RECENT, {"limit": max(0, limit)})
                    rows = await cur.fetchall()
        except Exception as exc:
            raise PersistenceError(f"audit query failed: {exc}") from exc
        return [AuditRecord.model_validate(dict(row)) for row in rows]

    async def close(self) -> None:
        """Close the pool."""
        await self._pool.close()


async def open_postgres_store(dsn: str, *, min_size: int, max_size: int) -> PostgresAuditStore:
    """Open a pool against ``dsn``, create the schema and return the store.

    Raises:
        PersistenceError: When the pool cannot connect within 10 seconds.
    """
    pool: AsyncConnectionPool = AsyncConnectionPool(
        conninfo=dsn, min_size=min_size, max_size=max_size, open=False, timeout=10
    )
    try:
        await pool.open(wait=True, timeout=10)
    except Exception as exc:
        raise PersistenceError(f"cannot connect to postgres: {exc}") from exc
    store = PostgresAuditStore(pool)
    await store.ensure_schema()
    return store
