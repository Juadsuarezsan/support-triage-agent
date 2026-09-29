"""Intent classifier backends."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from src.api.schemas import BITEXT_INTENTS, UNKNOWN_INTENT
from src.classifier.intent_classifier import (
    KEYWORD_RULES,
    ClaudeZeroShotClassifier,
    HFTransformerClassifier,
    KeywordHeuristicClassifier,
    build_classifier,
    get_classifier,
)
from src.config import Settings
from src.errors import LLMOutputError
from src.llm.claude import ClaudeClient


@pytest.fixture
def heuristic() -> KeywordHeuristicClassifier:
    return KeywordHeuristicClassifier()


async def test_refund_keyword(heuristic: KeywordHeuristicClassifier) -> None:
    result = await heuristic.classify("I want my money back, the product is broken")
    assert result.top.intent == "get_refund"
    assert result.backend == "keyword_heuristic"
    assert result.usage.calls == 0


async def test_human_keyword_scores_high(heuristic: KeywordHeuristicClassifier) -> None:
    result = await heuristic.classify("Please connect me with a human agent")
    assert result.top.intent == "contact_human_agent"
    assert result.top.score >= 0.8


async def test_default_unknown(heuristic: KeywordHeuristicClassifier) -> None:
    result = await heuristic.classify("Just wondering")
    assert result.top.intent == UNKNOWN_INTENT
    assert result.top.score == 0.0


def test_keyword_rules_cover_all_27_intents() -> None:
    covered = {intent for _, intent, _ in KEYWORD_RULES}
    assert covered == set(BITEXT_INTENTS)


@pytest.mark.parametrize("keywords,intent,_score", KEYWORD_RULES)
async def test_each_rule_first_keyword_routes_to_its_intent(
    heuristic: KeywordHeuristicClassifier, keywords: tuple[str, ...], intent: str, _score: float
) -> None:
    result = await heuristic.classify(f"Hello, {keywords[0]} please")
    assert result.top.intent == intent


async def test_zero_shot_parses_and_drops_unknown_labels(
    make_client: Callable[..., ClaudeClient], mock_json_reply: Callable[[dict[str, Any]], Any]
) -> None:
    mock_json_reply(
        {
            "top_intents": [
                {"intent": "made_up_label", "score": 0.9},
                {"intent": "track_order", "score": 0.6},
                {"intent": "delivery_period", "score": 0.3},
            ]
        }
    )
    clf = ClaudeZeroShotClassifier(make_client())
    result = await clf.classify("where is my parcel")
    assert [s.intent for s in result.intents] == ["track_order", "delivery_period"]
    assert result.backend == "claude_zero_shot"
    assert result.usage.calls == 1 and result.usage.input_tokens > 0


async def test_zero_shot_all_unknown_yields_unknown_intent(
    make_client: Callable[..., ClaudeClient], mock_json_reply: Callable[[dict[str, Any]], Any]
) -> None:
    mock_json_reply({"top_intents": [{"intent": "nope", "score": 0.5}]})
    result = await ClaudeZeroShotClassifier(make_client()).classify("x")
    assert result.top.intent == UNKNOWN_INTENT


async def test_zero_shot_malformed_output_raises(
    make_client: Callable[..., ClaudeClient], mock_json_reply: Callable[[dict[str, Any]], Any]
) -> None:
    mock_json_reply({"top_intents": []})  # violates min_length=1
    with pytest.raises(LLMOutputError):
        await ClaudeZeroShotClassifier(make_client()).classify("x")


def _fake_pipeline_factory(task: str, model: str, top_k: Any) -> Callable[[str], Any]:
    assert task == "text-classification" and top_k is None

    def _pipe(text: str) -> list[list[dict[str, Any]]]:
        return [[{"label": "cancel_order", "score": 0.2}, {"label": "get_refund", "score": 0.8}]]

    return _pipe


async def test_hf_classifier_uses_injected_pipeline_and_sorts() -> None:
    clf = HFTransformerClassifier("some/model", pipeline_factory=_fake_pipeline_factory)
    result = await clf.classify("refund please")
    assert [s.intent for s in result.intents] == ["get_refund", "cancel_order"]
    assert result.backend == "distilbert_lora_hf"


def test_hf_classifier_import_error_without_ml_extra() -> None:
    clf = HFTransformerClassifier("some/model")
    with pytest.raises(ImportError):
        clf._load()


def test_build_classifier_falls_back_to_heuristic_without_key_or_model() -> None:
    settings = Settings(anthropic_api_key=None, use_local_classifier=True)
    assert isinstance(build_classifier(settings), KeywordHeuristicClassifier)


def test_build_classifier_uses_zero_shot_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI exports ANTHROPIC_API_KEY="" ; an env value bound to the alias outranks
    # an init kwarg given by field name, so clear it and pass the alias.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(ANTHROPIC_API_KEY="k", USE_LOCAL_CLASSIFIER=False)
    assert isinstance(build_classifier(settings), ClaudeZeroShotClassifier)


def test_get_classifier_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_LOCAL_CLASSIFIER", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_classifier.cache_clear()
    assert get_classifier() is get_classifier()
    get_classifier.cache_clear()


def test_zero_shot_prompt_lists_every_intent() -> None:
    from src.classifier.intent_classifier import ZS_SYSTEM_PROMPT

    assert all(intent in ZS_SYSTEM_PROMPT for intent in BITEXT_INTENTS)
    json.dumps(ZS_SYSTEM_PROMPT)  # serialisable, no stray braces issue
