"""Structured logging, trace IDs and per-request token/cost accounting.

Every request gets a ``trace_id`` that is bound to the loguru context so all
log lines emitted while the request is processed carry it. Token usage from
each Claude call is accumulated into a :class:`TokenUsage` and converted to
USD with the prices configured in :mod:`src.config`.
"""

from __future__ import annotations

import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from loguru import logger

_TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="-")

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | "
    "trace={extra[trace_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
)


def configure_logging(level: str = "INFO") -> None:
    """Route loguru to stderr with the trace ID in every line.

    Args:
        level: Minimum level to emit (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    """
    logger.remove()
    logger.configure(extra={"trace_id": "-"})
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_LOG_FORMAT,
        backtrace=False,
        diagnose=False,
    )


def new_trace_id() -> str:
    """Generate a fresh 32-hex-character trace identifier."""
    return uuid.uuid4().hex


def bind_trace_id(trace_id: str) -> None:
    """Make ``trace_id`` the current trace for this async context."""
    _TRACE_ID.set(trace_id)


def current_trace_id() -> str:
    """Return the trace ID of the current async context (``"-"`` when unset)."""
    return _TRACE_ID.get()


def trace_logger() -> Any:
    """Return a loguru logger bound to the current trace ID."""
    return logger.bind(trace_id=current_trace_id())


@dataclass(frozen=True)
class TokenUsage:
    """Input/output token counts of one or more Claude calls.

    Attributes:
        input_tokens: Prompt tokens billed by the API.
        output_tokens: Completion tokens billed by the API.
        calls: Number of API calls aggregated into this record.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Aggregate two usage records."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


ZERO_USAGE = TokenUsage()


def estimate_cost_usd(
    usage: TokenUsage, *, price_input_per_mtok: float, price_output_per_mtok: float
) -> float:
    """Convert token usage into USD using per-million-token prices.

    Args:
        usage: Aggregated token counts.
        price_input_per_mtok: USD per 1M input tokens.
        price_output_per_mtok: USD per 1M output tokens.

    Returns:
        Cost in USD rounded to 6 decimals (zero when no tokens were consumed).
    """
    cost = (
        usage.input_tokens * price_input_per_mtok + usage.output_tokens * price_output_per_mtok
    ) / 1_000_000
    return round(cost, 6)


def log_node_io(node: str, state_in: dict[str, Any], state_out: dict[str, Any]) -> None:
    """Emit one DEBUG line with the keys a graph node read and the values it wrote.

    Values are truncated to keep log lines short; the full state lives in the
    audit store.

    Args:
        node: Graph node name.
        state_in: The state the node received.
        state_out: The partial state update the node returned.
    """
    summary = {k: _short(v) for k, v in state_out.items()}
    trace_logger().debug("node={} in_keys={} out={}", node, sorted(state_in.keys()), summary)


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
