"""ClaudeClient against a mocked Messages endpoint (respx)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
import respx
from pydantic import BaseModel

from src.errors import LLMError, LLMOutputError
from src.llm.claude import ClaudeClient, extract_json, parse_model
from tests.conftest import MESSAGES_URL, claude_response


async def test_complete_returns_text_and_usage(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    route = anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, json=claude_response('{"a": 1}', input_tokens=11, output_tokens=7)
        )
    )
    result = await make_client().complete(system="s", user="u", max_tokens=50)
    assert result.text == '{"a": 1}'
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.calls) == (11, 7, 1)
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "claude-sonnet-4-5-20250929"
    assert sent["max_tokens"] == 50
    assert sent["messages"] == [{"role": "user", "content": "u"}]


async def test_retries_on_overloaded_then_succeeds(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient], mocker
) -> None:
    mocker.patch("tenacity.nap.sleep")  # no real backoff in tests
    route = anthropic_mock.post(MESSAGES_URL).mock(
        side_effect=[
            httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
            httpx.Response(200, json=claude_response("ok")),
        ]
    )
    result = await make_client(max_attempts=3).complete(system="s", user="u")
    assert result.text == "ok"
    assert route.call_count == 2


async def test_gives_up_after_max_attempts(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient], mocker
) -> None:
    mocker.patch("tenacity.nap.sleep")
    route = anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(500, json={"type": "error", "error": {"type": "api_error"}})
    )
    with pytest.raises(LLMError, match="after 2 attempts"):
        await make_client(max_attempts=2).complete(system="s", user="u")
    assert route.call_count == 2


async def test_bad_request_is_not_retried(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    route = anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
        )
    )
    with pytest.raises(LLMError, match="API error"):
        await make_client(max_attempts=3).complete(system="s", user="u")
    assert route.call_count == 1


async def test_timeout_is_retried_then_raises(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient], mocker
) -> None:
    mocker.patch("tenacity.nap.sleep")
    route = anthropic_mock.post(MESSAGES_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError):
        await make_client(max_attempts=2).complete(system="s", user="u")
    assert route.call_count == 2


class _Schema(BaseModel):
    answer: int


@pytest.mark.parametrize(
    "text",
    ['{"answer": 3}', '```json\n{"answer": 3}\n```', 'Sure! Here it is: {"answer": 3} Done.'],
)
def test_extract_json_variants(text: str) -> None:
    assert extract_json(text) == {"answer": 3}
    assert parse_model(text, _Schema).answer == 3


@pytest.mark.parametrize("text", ["no json here", "{not: valid}", "[1, 2, 3]", ""])
def test_extract_json_rejects_malformed(text: str) -> None:
    with pytest.raises(LLMOutputError):
        extract_json(text)


def test_parse_model_rejects_schema_violation() -> None:
    with pytest.raises(LLMOutputError, match="_Schema"):
        parse_model('{"answer": "three"}', _Schema)
