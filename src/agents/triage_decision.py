"""Triage decision logic — pure deterministic rules over agent outputs."""
from __future__ import annotations

from src.api.schemas import IntentScore, SimilarTicket, TriageDecision


def confidence_score(
    intent: list[IntentScore],
    similar: list[SimilarTicket],
    sentiment: str,
    urgency_score: float,
) -> float:
    """Combine signals into a single confidence in [0, 1]."""
    intent_conf = intent[0].score if intent else 0.0
    sim_conf = similar[0].similarity if similar else 0.0
    # Negative sentiment + high urgency pull confidence down (caution).
    penalty = 0.0
    if sentiment == "neg":
        penalty += 0.15
    if urgency_score >= 0.85:
        penalty += 0.15
    return max(0.0, min(1.0, 0.5 * intent_conf + 0.4 * sim_conf + 0.1 - penalty))


def decide(
    *,
    intent: list[IntentScore],
    similar: list[SimilarTicket],
    sentiment: str,
    urgency_score: float,
    auto_threshold: float,
    escalate_threshold: float,
) -> TriageDecision:
    score = confidence_score(intent, similar, sentiment, urgency_score)
    top_intent = intent[0].intent if intent else "unknown"

    # Hard escalation rules first — never auto-resolve these.
    if top_intent in ("complaint", "contact_human_agent"):
        return TriageDecision(decision="escalate", confidence=score,
                                rationale=f"hard rule: intent={top_intent}")
    if urgency_score >= 0.9:
        return TriageDecision(decision="escalate", confidence=score,
                                rationale=f"hard rule: urgency_score={urgency_score:.2f}")

    if score >= auto_threshold:
        return TriageDecision(decision="auto_resolve", confidence=score,
                                rationale=f"confidence {score:.2f} >= {auto_threshold}")
    if score < escalate_threshold:
        return TriageDecision(decision="escalate", confidence=score,
                                rationale=f"confidence {score:.2f} < {escalate_threshold}")
    return TriageDecision(decision="suggest", confidence=score,
                            rationale=f"confidence {score:.2f} in suggest band")
