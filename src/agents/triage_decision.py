"""Triage decision: pure, deterministic rules over the agents' outputs.

``score = 0.5 * intent_conf + 0.4 * top_similarity + 0.1 - penalties`` where a
negative sentiment and a very high urgency each subtract 0.15. Two hard rules
run before the score: ``complaint`` and ``contact_human_agent`` always
escalate, and so does ``urgency_score >= 0.9``.

The keyword flags on :func:`confidence_score` exist for the ablation study in
``eval/``; production always uses the defaults.
"""

from __future__ import annotations

from src.api.schemas import IntentScore, SimilarTicket, TriageDecision

#: Intents that a machine must never close on its own.
HARD_ESCALATE_INTENTS: frozenset[str] = frozenset({"complaint", "contact_human_agent"})
HARD_ESCALATE_URGENCY = 0.9

W_INTENT = 0.5
W_SIMILARITY = 0.4
BIAS = 0.1
PENALTY_NEGATIVE = 0.15
PENALTY_URGENT = 0.15
URGENT_PENALTY_FROM = 0.85


def confidence_score(
    intent: list[IntentScore],
    similar: list[SimilarTicket],
    sentiment: str,
    urgency_score: float,
    *,
    use_similarity: bool = True,
    use_sentiment_penalty: bool = True,
) -> float:
    """Combine the agents' signals into one confidence in ``[0, 1]``.

    Args:
        intent: Ranked intents; only the top score is used.
        similar: Retrieved tickets; only the top similarity is used.
        sentiment: ``neg`` / ``neu`` / ``pos``.
        urgency_score: Urgency in ``[0, 1]``.
        use_similarity: Ablation switch; when false the similarity weight is
            redistributed to the intent term so the scale is unchanged.
        use_sentiment_penalty: Ablation switch for the sentiment/urgency penalty.

    Returns:
        Clamped confidence.
    """
    intent_conf = intent[0].score if intent else 0.0
    sim_conf = similar[0].similarity if similar else 0.0
    if use_similarity:
        base = W_INTENT * intent_conf + W_SIMILARITY * sim_conf + BIAS
    else:
        base = (W_INTENT + W_SIMILARITY) * intent_conf + BIAS
    penalty = 0.0
    if use_sentiment_penalty:
        if sentiment == "neg":
            penalty += PENALTY_NEGATIVE
        if urgency_score >= URGENT_PENALTY_FROM:
            penalty += PENALTY_URGENT
    return max(0.0, min(1.0, base - penalty))


def decide(
    *,
    intent: list[IntentScore],
    similar: list[SimilarTicket],
    sentiment: str,
    urgency_score: float,
    auto_threshold: float,
    escalate_threshold: float,
    use_similarity: bool = True,
    use_sentiment_penalty: bool = True,
) -> TriageDecision:
    """Route a ticket to ``auto_resolve`` / ``suggest`` / ``escalate``.

    Args:
        intent: Ranked intents from the classifier.
        similar: Similar resolved tickets from the retriever.
        sentiment: Sentiment label.
        urgency_score: Urgency in ``[0, 1]``.
        auto_threshold: Score at or above which the ticket auto-resolves.
        escalate_threshold: Score below which the ticket escalates.
        use_similarity: Forwarded to :func:`confidence_score` (ablation).
        use_sentiment_penalty: Forwarded to :func:`confidence_score` (ablation).

    Returns:
        The decision, its confidence and a one-line rationale.
    """
    score = confidence_score(
        intent,
        similar,
        sentiment,
        urgency_score,
        use_similarity=use_similarity,
        use_sentiment_penalty=use_sentiment_penalty,
    )
    top_intent = intent[0].intent if intent else "unknown"

    if top_intent in HARD_ESCALATE_INTENTS:
        return TriageDecision(
            decision="escalate", confidence=score, rationale=f"hard rule: intent={top_intent}"
        )
    if urgency_score >= HARD_ESCALATE_URGENCY:
        return TriageDecision(
            decision="escalate",
            confidence=score,
            rationale=f"hard rule: urgency_score={urgency_score:.2f}",
        )
    if score >= auto_threshold:
        return TriageDecision(
            decision="auto_resolve",
            confidence=score,
            rationale=f"confidence {score:.2f} >= {auto_threshold:.2f}",
        )
    if score < escalate_threshold:
        return TriageDecision(
            decision="escalate",
            confidence=score,
            rationale=f"confidence {score:.2f} < {escalate_threshold:.2f}",
        )
    return TriageDecision(
        decision="suggest", confidence=score, rationale=f"confidence {score:.2f} in suggest band"
    )
