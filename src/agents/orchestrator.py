"""LangGraph orchestration of the triage flow."""
from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agents.drafter import Drafter
from src.agents.priority_sentiment import PrioritySentimentAnalyzer
from src.agents.triage_decision import decide
from src.api.schemas import IntentScore, SimilarTicket, TriageDecision, TriageOut
from src.classifier.intent_classifier import get_classifier
from src.config import get_settings
from src.retrieval.similar_tickets import InMemoryTicketStore


class TriageState(TypedDict, total=False):
    ticket_id: str
    body: str
    intents: list[IntentScore]
    priority: str
    sentiment: str
    urgency_score: float
    rationale: str
    similar: list[SimilarTicket]
    draft: str | None
    decision: TriageDecision
    started_at: float


def build_graph(store: InMemoryTicketStore):
    s = get_settings()
    classifier = get_classifier()
    ps = PrioritySentimentAnalyzer(model=s.anthropic_model, api_key=s.anthropic_api_key)
    drafter = Drafter(model=s.anthropic_model, api_key=s.anthropic_api_key)

    async def classify_node(state: TriageState) -> dict[str, Any]:
        intents = await classifier.classify(state["body"])
        logger.info(f"classified intent={intents[0].intent if intents else 'none'}")
        return {"intents": intents}

    async def sentiment_node(state: TriageState) -> dict[str, Any]:
        priority, sentiment, urgency, rationale = await ps.analyze(state["body"])
        return {"priority": priority, "sentiment": sentiment,
                "urgency_score": urgency, "rationale": rationale}

    async def similar_node(state: TriageState) -> dict[str, Any]:
        similar = await store.search(state["body"], k=5)
        return {"similar": similar}

    async def draft_node(state: TriageState) -> dict[str, Any]:
        draft = await drafter.draft(state["body"], state.get("similar", []))
        return {"draft": draft}

    def decide_node(state: TriageState) -> dict[str, Any]:
        decision = decide(
            intent=state.get("intents", []),
            similar=state.get("similar", []),
            sentiment=state.get("sentiment", "neu"),
            urgency_score=state.get("urgency_score", 0.0),
            auto_threshold=s.auto_resolve_threshold,
            escalate_threshold=s.escalate_threshold,
        )
        return {"decision": decision}

    g = StateGraph(TriageState)
    g.add_node("classify", classify_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("similar", similar_node)
    g.add_node("draft", draft_node)
    g.add_node("decide", decide_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "sentiment")
    g.add_edge("sentiment", "similar")
    g.add_edge("similar", "draft")
    g.add_edge("draft", "decide")
    g.add_edge("decide", END)
    return g.compile()


async def triage_ticket(graph: Any, *, ticket_id: str, body: str) -> TriageOut:
    started = time.perf_counter()
    initial: TriageState = {"ticket_id": ticket_id, "body": body, "started_at": started}
    state = await graph.ainvoke(initial)
    latency_ms = int((time.perf_counter() - started) * 1000)
    intents: list[IntentScore] = state.get("intents", [])
    return TriageOut(
        ticket_id=ticket_id,
        intent=intents[0].intent if intents else "general_question",
        intent_confidence=intents[0].score if intents else 0.0,
        top_intents=intents,
        priority=state.get("priority", "P3"),
        sentiment=state.get("sentiment", "neu"),
        urgency_score=state.get("urgency_score", 0.0),
        similar_resolved=state.get("similar", []),
        draft_response=state.get("draft"),
        decision=state.get("decision", TriageDecision(decision="escalate", confidence=0.0, rationale="missing")),
        latency_ms=latency_ms,
    )
