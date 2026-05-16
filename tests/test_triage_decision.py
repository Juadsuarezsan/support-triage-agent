"""Pure decision logic — no external dependencies."""
from __future__ import annotations

from src.agents.triage_decision import confidence_score, decide
from src.api.schemas import IntentScore, SimilarTicket


def _intent(name: str, score: float) -> list[IntentScore]:
    return [IntentScore(intent=name, score=score)]


def _sim(score: float = 0.9) -> list[SimilarTicket]:
    return [SimilarTicket(ticket_id="t1", intent="x", body_snippet="", resolution_snippet="", similarity=score)]


def test_complaint_always_escalates() -> None:
    d = decide(
        intent=_intent("complaint", 0.99),
        similar=_sim(0.99),
        sentiment="neg",
        urgency_score=0.9,
        auto_threshold=0.85, escalate_threshold=0.60,
    )
    assert d.decision == "escalate"


def test_contact_human_always_escalates() -> None:
    d = decide(
        intent=_intent("contact_human_agent", 0.99),
        similar=_sim(0.99),
        sentiment="neu", urgency_score=0.1,
        auto_threshold=0.85, escalate_threshold=0.60,
    )
    assert d.decision == "escalate"


def test_high_confidence_auto_resolves() -> None:
    d = decide(
        intent=_intent("track_order", 0.95),
        similar=_sim(0.95),
        sentiment="neu", urgency_score=0.2,
        auto_threshold=0.80, escalate_threshold=0.40,
    )
    assert d.decision == "auto_resolve"


def test_negative_sentiment_penalizes_confidence() -> None:
    s_neu = confidence_score(_intent("track_order", 0.8), _sim(0.8), "neu", 0.2)
    s_neg = confidence_score(_intent("track_order", 0.8), _sim(0.8), "neg", 0.2)
    assert s_neg < s_neu


def test_low_confidence_escalates() -> None:
    d = decide(
        intent=_intent("track_order", 0.3),
        similar=[],
        sentiment="neu", urgency_score=0.1,
        auto_threshold=0.85, escalate_threshold=0.60,
    )
    assert d.decision == "escalate"
