"""Shared Claude client used by every LLM-backed component.

The client wraps :class:`anthropic.AsyncAnthropic` with an explicit timeout,
tenacity retries with exponential backoff on transient failures, token
accounting and a strict JSON extractor. Components never talk to the SDK
directly, which keeps mocking to a single seam (:meth:`ClaudeClient.complete`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.errors import LLMError, LLMOutputError
from src.observability import TokenUsage, trace_logger

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class LLMResult:
    """Text returned by Claude together with its billed token usage.

    Attributes:
        text: Concatenated text blocks of the response.
        usage: Tokens billed for this single call.
        model: Model ID that served the request.
        stop_reason: Provider stop reason (``end_turn``, ``max_tokens``...).
    """

    text: str
    usage: TokenUsage
    model: str
    stop_reason: str


def _is_transient(exc: BaseException) -> bool:
    """Decide whether an SDK exception is worth retrying."""
    if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 529
    return isinstance(exc, httpx.TimeoutException)


class ClaudeClient:
    """Async Claude client with timeout, retries and usage accounting.

    Args:
        api_key: Anthropic API key.
        model: Dated model ID (see :data:`src.config.DEFAULT_ANTHROPIC_MODEL`).
        timeout_s: Per-request HTTP timeout in seconds.
        max_attempts: Total attempts including the first one.
        http_client: Optional pre-built ``httpx.AsyncClient`` (used by tests to
            inject ``respx`` mocks).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 15.0,
        max_attempts: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        # SDK-level retries are disabled: tenacity owns the retry policy so the
        # backoff and attempt count are visible in one place.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Send one user turn to Claude and return the text plus token usage.

        Args:
            system: System prompt.
            user: Single user message.
            max_tokens: Output cap; defaults are small because every prompt in
                this service asks for compact JSON.
            temperature: Sampling temperature (0 for classification-like tasks).

        Returns:
            The provider response as an :class:`LLMResult`.

        Raises:
            LLMError: When every attempt failed (network, rate limit, 5xx) or
                the provider returned a non-retryable API error.
        """
        log = trace_logger()
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception(_is_transient),
            reraise=False,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        log.warning(
                            "claude retry attempt={} model={}",
                            attempt.retry_state.attempt_number,
                            self.model,
                        )
                    message = await self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return self._to_result(message)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            raise LLMError(
                f"Claude call failed after {self.max_attempts} attempts: {last}"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc
        raise LLMError("Claude call produced no result")  # pragma: no cover - defensive

    def _to_result(self, message: anthropic.types.Message) -> LLMResult:
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            calls=1,
        )
        trace_logger().info(
            "claude ok model={} in_tokens={} out_tokens={} stop={}",
            message.model,
            usage.input_tokens,
            usage.output_tokens,
            message.stop_reason,
        )
        return LLMResult(
            text=text,
            usage=usage,
            model=message.model,
            stop_reason=str(message.stop_reason or "end_turn"),
        )


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Handles bare JSON, JSON wrapped in Markdown code fences and JSON preceded
    or followed by prose (the first ``{`` ... last ``}`` span is used).

    Args:
        text: Raw model text.

    Returns:
        The decoded top-level object.

    Raises:
        LLMOutputError: When no JSON object can be decoded.
    """
    candidate = _FENCE.sub("", text.strip()).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMOutputError(f"no JSON object in LLM output: {text[:120]!r}")
    try:
        decoded = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"malformed JSON in LLM output: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise LLMOutputError("LLM output is JSON but not an object")
    return decoded


def parse_model(text: str, schema: type[ModelT]) -> ModelT:
    """Decode ``text`` as JSON and validate it against a Pydantic model.

    Args:
        text: Raw model text.
        schema: Pydantic model class to validate against.

    Returns:
        A validated instance of ``schema``.

    Raises:
        LLMOutputError: When the JSON is malformed or violates the schema.
    """
    payload = extract_json(text)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMOutputError(f"LLM output failed schema {schema.__name__}: {exc}") from exc
