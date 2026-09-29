"""Shared fixtures: fixed seeds, mocked Claude HTTP endpoint, offline pipeline."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import respx

from src.agents.orchestrator import TriagePipeline
from src.config import Settings, get_settings
from src.eval.runner import build_offline_pipeline
from src.llm.claude import ClaudeClient

SEED = 20260516
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5-20250929"


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def offline_settings() -> Settings:
    """Keyless configuration: every component runs its deterministic fallback."""
    return Settings(
        anthropic_api_key=None,
        use_local_classifier=False,
        encoder_backend="deterministic",
        retrieval_backend="memory",
        persistence_backend="memory",
        cors_origins="http://localhost:3000",
        rate_limit="1000/minute",
    )


@pytest.fixture
async def offline_pipeline(offline_settings: Settings) -> TriagePipeline:
    return await build_offline_pipeline(offline_settings)


def claude_response(
    text: str, *, input_tokens: int = 120, output_tokens: int = 40
) -> dict[str, Any]:
    """Build a Messages API JSON body with a single text block."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


@pytest.fixture
def anthropic_mock() -> Iterator[respx.MockRouter]:
    """A ``respx`` router intercepting the Messages endpoint."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def make_client() -> Callable[..., ClaudeClient]:
    """Factory for a :class:`ClaudeClient` whose HTTP goes through ``respx``."""

    def _make(max_attempts: int = 3, timeout_s: float = 5.0) -> ClaudeClient:
        return ClaudeClient(
            api_key="test-key",
            model=MODEL,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            http_client=httpx.AsyncClient(),
        )

    return _make


@pytest.fixture
def mock_json_reply(anthropic_mock: respx.MockRouter) -> Callable[[dict[str, Any]], respx.Route]:
    """Register a 200 reply whose text is ``json.dumps(payload)``."""

    def _register(payload: dict[str, Any]) -> respx.Route:
        return anthropic_mock.post(MESSAGES_URL).mock(
            return_value=httpx.Response(200, json=claude_response(json.dumps(payload)))
        )

    return _register
