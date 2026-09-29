"""Token/cost accounting, trace binding and Settings validation."""

from __future__ import annotations

import pytest

from src.config import Settings, get_settings
from src.observability import (
    TokenUsage,
    bind_trace_id,
    configure_logging,
    current_trace_id,
    estimate_cost_usd,
    new_trace_id,
)


def test_token_usage_addition_and_cost() -> None:
    u = TokenUsage(100, 50, 1) + TokenUsage(200, 100, 2)
    assert (u.input_tokens, u.output_tokens, u.calls, u.total_tokens) == (300, 150, 3, 450)
    cost = estimate_cost_usd(u, price_input_per_mtok=3.0, price_output_per_mtok=15.0)
    assert cost == pytest.approx((300 * 3 + 150 * 15) / 1_000_000)
    assert (
        estimate_cost_usd(TokenUsage(), price_input_per_mtok=3.0, price_output_per_mtok=15.0) == 0
    )


def test_trace_id_binding() -> None:
    assert current_trace_id() == "-"
    tid = new_trace_id()
    assert len(tid) == 32
    bind_trace_id(tid)
    assert current_trace_id() == tid
    configure_logging("DEBUG")


def test_settings_defaults_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings()
    assert s.anthropic_model == "claude-sonnet-4-5-20250929"
    assert s.llm_enabled is False and s.langsmith_enabled is False
    assert s.cors_origin_list == ["http://localhost:3000"]
    with pytest.raises(ValueError, match="ESCALATE_THRESHOLD"):
        Settings(auto_resolve_threshold=0.5, escalate_threshold=0.6)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example,*")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls")
    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_enabled and s.langsmith_enabled
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]
