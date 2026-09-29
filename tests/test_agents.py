"""PrioritySentimentAnalyzer, Drafter and the decision rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from src.agents.drafter import Drafter
from src.agents.priority_sentiment import PrioritySentimentAnalyzer
from src.agents.triage_decision import confidence_score, decide
from src.api.schemas import IntentScore, SimilarTicket
from src.llm.claude import ClaudeClient
from tests.conftest import MESSAGES_URL, claude_response

# --- PrioritySentimentAnalyzer -------------------------------------------------


@pytest.fixture
def analyzer() -> PrioritySentimentAnalyzer:
    return PrioritySentimentAnalyzer(client=None)


async def test_urgent_and_negative_yield_p0(analyzer: PrioritySentimentAnalyzer) -> None:
    r = await analyzer.analyze("PRODUCTION IS DOWN, this is unacceptable")
    assert r.analysis.priority == "P0"
    assert r.analysis.urgency_score >= 0.9
    assert r.source == "heuristic"


async def test_urgent_only_yields_p1(analyzer: PrioritySentimentAnalyzer) -> None:
    r = await analyzer.analyze("I need this fixed today please")
    assert r.analysis.priority == "P1"
    assert r.analysis.sentiment == "neu"


async def test_negative_yields_p1_neg(analyzer: PrioritySentimentAnalyzer) -> None:
    r = await analyzer.analyze("This is the worst service ever")
    assert (r.analysis.priority, r.analysis.sentiment) == ("P1", "neg")


async def test_neutral_defaults_p3(analyzer: PrioritySentimentAnalyzer) -> None:
    r = await analyzer.analyze("How does shipping work?")
    assert (r.analysis.priority, r.analysis.sentiment) == ("P3", "neu")


async def test_positive_sentiment(analyzer: PrioritySentimentAnalyzer) -> None:
    r = await analyzer.analyze("Thanks, great service")
    assert r.analysis.sentiment == "pos"


async def test_llm_structured_output(
    make_client: Callable[..., ClaudeClient], mock_json_reply: Callable[[dict[str, Any]], Any]
) -> None:
    mock_json_reply(
        {"sentiment": "neg", "priority": "P1", "urgency_score": 0.7, "rationale": "angry"}
    )
    r = await PrioritySentimentAnalyzer(make_client()).analyze("bad")
    assert r.source == "llm"
    assert r.analysis.priority == "P1" and r.analysis.urgency_score == 0.7
    assert r.usage.calls == 1


async def test_llm_malformed_output_falls_back_to_heuristic(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=claude_response('{"priority": "P9"}'))
    )
    r = await PrioritySentimentAnalyzer(make_client()).analyze("worst experience")
    assert r.source == "heuristic_after_llm_error"
    assert r.analysis.sentiment == "neg"
    assert r.usage.calls == 1  # tokens were still billed


# --- Drafter ------------------------------------------------------------------


def _sim(score: float = 0.9, resolution: str = "Refund issued.") -> SimilarTicket:
    return SimilarTicket(
        ticket_id="t1",
        intent="get_refund",
        body_snippet="b",
        resolution_snippet=resolution,
        similarity=score,
    )


async def test_template_without_similar() -> None:
    r = await Drafter(client=None).draft("hi", [])
    assert r.source == "template" and "teammate" in r.text


async def test_template_reuses_top_resolution() -> None:
    r = await Drafter(client=None).draft("hi", [_sim()])
    assert "Refund issued." in r.text and "get_refund" in r.text


async def test_llm_draft(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    route = anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=claude_response("Sorry about that. Next step: ..."))
    )
    r = await Drafter(make_client()).draft("my order is late", [_sim()])
    assert r.source == "llm" and r.text.startswith("Sorry")
    assert b"<similar>" in route.calls[0].request.content


async def test_llm_empty_text_uses_template(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    anthropic_mock.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=claude_response("   "))
    )
    r = await Drafter(make_client()).draft("x", [])
    assert r.source == "llm" and "teammate" in r.text


# --- decision rules -----------------------------------------------------------


def _intent(name: str, score: float) -> list[IntentScore]:
    return [IntentScore(intent=name, score=score)]


THRESH = {"auto_threshold": 0.85, "escalate_threshold": 0.60}


def test_complaint_always_escalates() -> None:
    d = decide(
        intent=_intent("complaint", 0.99),
        similar=[_sim(0.99)],
        sentiment="neg",
        urgency_score=0.9,
        **THRESH,
    )
    assert d.decision == "escalate" and "hard rule" in d.rationale


def test_contact_human_always_escalates() -> None:
    d = decide(
        intent=_intent("contact_human_agent", 0.99),
        similar=[_sim(0.99)],
        sentiment="neu",
        urgency_score=0.1,
        **THRESH,
    )
    assert d.decision == "escalate"


def test_high_urgency_escalates_any_intent() -> None:
    d = decide(
        intent=_intent("track_order", 0.99),
        similar=[_sim(0.99)],
        sentiment="pos",
        urgency_score=0.95,
        **THRESH,
    )
    assert d.decision == "escalate" and "urgency" in d.rationale


def test_high_confidence_auto_resolves() -> None:
    d = decide(
        intent=_intent("track_order", 0.95),
        similar=[_sim(0.95)],
        sentiment="neu",
        urgency_score=0.2,
        auto_threshold=0.80,
        escalate_threshold=0.40,
    )
    assert d.decision == "auto_resolve"


def test_mid_confidence_suggests() -> None:
    d = decide(
        intent=_intent("track_order", 0.7),
        similar=[_sim(0.7)],
        sentiment="neu",
        urgency_score=0.2,
        **THRESH,
    )
    assert d.decision == "suggest"


def test_low_confidence_escalates() -> None:
    d = decide(
        intent=_intent("track_order", 0.3), similar=[], sentiment="neu", urgency_score=0.1, **THRESH
    )
    assert d.decision == "escalate"


def test_negative_sentiment_penalizes_confidence() -> None:
    s_neu = confidence_score(_intent("track_order", 0.8), [_sim(0.8)], "neu", 0.2)
    s_neg = confidence_score(_intent("track_order", 0.8), [_sim(0.8)], "neg", 0.2)
    assert s_neg == pytest.approx(s_neu - 0.15)


def test_ablation_switches_change_score() -> None:
    intent, sim = _intent("track_order", 0.6), [_sim(1.0)]
    full = confidence_score(intent, sim, "neg", 0.9)
    no_sim = confidence_score(intent, sim, "neg", 0.9, use_similarity=False)
    no_pen = confidence_score(intent, sim, "neg", 0.9, use_sentiment_penalty=False)
    assert no_sim < full < no_pen
    assert 0.0 <= no_sim <= 1.0


def test_score_is_clamped() -> None:
    assert confidence_score([], [], "neg", 0.99) == 0.0
    assert confidence_score(_intent("x", 1.0), [_sim(1.0)], "pos", 0.0) == 1.0
