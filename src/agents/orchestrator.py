"""LangGraph orchestration of the triage flow.

``classify_intent -> analyze_priority -> retrieve_similar -> draft_solution ->
decide_route -> record_audit``. Every node logs the keys it read and the
values it wrote, and the LLM-backed nodes add their token usage to the state
so the final :class:`~src.api.schemas.TriageOut` carries tokens and cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from src.agents.drafter import Drafter
from src.agents.priority_sentiment import PrioritySentimentAnalyzer
from src.agents.triage_decision import decide
from src.api.schemas import (
    UNKNOWN_INTENT,
    AuditRecord,
    IntentScore,
    Priority,
    Sentiment,
    SimilarTicket,
    TicketIn,
    TriageDecision,
    TriageOut,
)
from src.classifier.intent_classifier import IntentClassifier, build_classifier
from src.config import Settings
from src.llm.claude import ClaudeClient
from src.observability import (
    TokenUsage,
    bind_trace_id,
    estimate_cost_usd,
    log_node_io,
    new_trace_id,
    trace_logger,
)
from src.persistence.audit import AuditStore
from src.retrieval.similar_tickets import TicketStore


class TriageState(TypedDict, total=False):
    """Mutable state threaded through the graph.

    Node names never reuse these keys (LangGraph rejects that at build time).
    """

    trace_id: str
    ticket_id: str
    channel: str
    body: str
    intents: list[IntentScore]
    classifier_backend: str
    priority: Priority
    sentiment: Sentiment
    urgency_score: float
    rationale: str
    similar: list[SimilarTicket]
    draft: str | None
    decision: TriageDecision
    input_tokens: int
    output_tokens: int
    llm_calls: int
    started_at: float
    latency_ms: int
    cost_usd: float


@dataclass
class TriagePipeline:
    """All components of the triage flow plus the compiled graph.

    Attributes:
        settings: Effective configuration.
        classifier: Intent classifier backend.
        analyzer: Sentiment/priority analyzer.
        drafter: Reply drafter.
        store: Similar-ticket store.
        audit: Audit log store.
        graph: Compiled LangGraph (built by :func:`build_pipeline`).
    """

    settings: Settings
    classifier: IntentClassifier
    analyzer: PrioritySentimentAnalyzer
    drafter: Drafter
    store: TicketStore
    audit: AuditStore
    graph: CompiledStateGraph

    @property
    def llm_enabled(self) -> bool:
        """Whether any component will call Claude."""
        return self.settings.llm_enabled

    async def triage(self, ticket: TicketIn) -> TriageOut:
        """Run the full graph for ``ticket`` and return the API response.

        A fresh ``trace_id`` is generated and bound for the duration of the
        call so every log line and the audit row share it.
        """
        trace_id = new_trace_id()
        bind_trace_id(trace_id)
        started = time.perf_counter()
        initial: TriageState = {
            "trace_id": trace_id,
            "ticket_id": ticket.ticket_id,
            "channel": ticket.channel,
            "body": ticket.body,
            "started_at": started,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
        }
        trace_logger().info(
            "triage start ticket_id={} chars={}", ticket.ticket_id, len(ticket.body)
        )
        state = cast(TriageState, await self.graph.ainvoke(initial))
        out = _to_output(state, self.settings)
        trace_logger().info(
            "triage done ticket_id={} intent={} decision={} latency_ms={} tokens={}/{} cost_usd={}",
            out.ticket_id,
            out.intent,
            out.decision.decision,
            out.latency_ms,
            out.input_tokens,
            out.output_tokens,
            out.cost_usd,
        )
        return out


def _to_output(state: TriageState, settings: Settings) -> TriageOut:
    intents = state.get("intents", [])
    usage = TokenUsage(
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        calls=state.get("llm_calls", 0),
    )
    return TriageOut(
        ticket_id=state["ticket_id"],
        trace_id=state["trace_id"],
        intent=intents[0].intent if intents else UNKNOWN_INTENT,
        intent_confidence=intents[0].score if intents else 0.0,
        top_intents=intents[:3],
        classifier_backend=state.get("classifier_backend", "unknown"),
        priority=state.get("priority", "P3"),
        sentiment=state.get("sentiment", "neu"),
        urgency_score=state.get("urgency_score", 0.0),
        similar_resolved=state.get("similar", []),
        draft_response=state.get("draft"),
        decision=state.get(
            "decision", TriageDecision(decision="escalate", confidence=0.0, rationale="missing")
        ),
        llm_used=usage.calls > 0,
        latency_ms=state.get("latency_ms", 0),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=estimate_cost_usd(
            usage,
            price_input_per_mtok=settings.price_input_per_mtok,
            price_output_per_mtok=settings.price_output_per_mtok,
        ),
    )


def audit_record_from(state: TriageState, out: TriageOut) -> AuditRecord:
    """Build the audit row for a finished triage."""
    channel = state.get("channel", "email")
    return AuditRecord(
        trace_id=out.trace_id,
        ticket_id=out.ticket_id,
        channel=channel if channel in ("email", "chat", "slack", "twitter") else "email",
        body=state.get("body", ""),
        intent=out.intent,
        intent_confidence=out.intent_confidence,
        priority=out.priority,
        sentiment=out.sentiment,
        urgency_score=out.urgency_score,
        decision=out.decision.decision,
        confidence=out.decision.confidence,
        rationale=out.decision.rationale,
        draft_response=out.draft_response,
        similar_ticket_ids=[t.ticket_id for t in out.similar_resolved],
        llm_used=out.llm_used,
        latency_ms=out.latency_ms,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
        cost_usd=out.cost_usd,
        created_at=datetime.now(UTC),
    )


def _usage_update(state: TriageState, usage: TokenUsage) -> dict[str, int]:
    return {
        "input_tokens": state.get("input_tokens", 0) + usage.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + usage.output_tokens,
        "llm_calls": state.get("llm_calls", 0) + usage.calls,
    }


def build_graph(
    *,
    settings: Settings,
    classifier: IntentClassifier,
    analyzer: PrioritySentimentAnalyzer,
    drafter: Drafter,
    store: TicketStore,
    audit: AuditStore,
) -> CompiledStateGraph:
    """Wire the six nodes into a compiled LangGraph."""

    async def classify_intent(state: TriageState) -> dict[str, Any]:
        result = await classifier.classify(state["body"])
        update: dict[str, Any] = {
            "intents": result.intents,
            "classifier_backend": result.backend,
            **_usage_update(state, result.usage),
        }
        log_node_io("classify_intent", dict(state), update)
        return update

    async def analyze_priority(state: TriageState) -> dict[str, Any]:
        result = await analyzer.analyze(state["body"])
        update: dict[str, Any] = {
            "priority": result.analysis.priority,
            "sentiment": result.analysis.sentiment,
            "urgency_score": result.analysis.urgency_score,
            "rationale": result.analysis.rationale,
            **_usage_update(state, result.usage),
        }
        log_node_io("analyze_priority", dict(state), update)
        return update

    async def retrieve_similar(state: TriageState) -> dict[str, Any]:
        similar = await store.search(state["body"], k=5)
        update: dict[str, Any] = {"similar": similar}
        log_node_io("retrieve_similar", dict(state), update)
        return update

    async def draft_solution(state: TriageState) -> dict[str, Any]:
        result = await drafter.draft(state["body"], state.get("similar", []))
        update: dict[str, Any] = {"draft": result.text, **_usage_update(state, result.usage)}
        log_node_io("draft_solution", dict(state), update)
        return update

    def decide_route(state: TriageState) -> dict[str, Any]:
        decision = decide(
            intent=state.get("intents", []),
            similar=state.get("similar", []),
            sentiment=state.get("sentiment", "neu"),
            urgency_score=state.get("urgency_score", 0.0),
            auto_threshold=settings.auto_resolve_threshold,
            escalate_threshold=settings.escalate_threshold,
        )
        latency_ms = int(
            (time.perf_counter() - state.get("started_at", time.perf_counter())) * 1000
        )
        update: dict[str, Any] = {"decision": decision, "latency_ms": latency_ms}
        log_node_io("decide_route", dict(state), update)
        return update

    async def record_audit(state: TriageState) -> dict[str, Any]:
        out = _to_output(state, settings)
        await audit.record(audit_record_from(state, out))
        update: dict[str, Any] = {"cost_usd": out.cost_usd}
        log_node_io("record_audit", dict(state), update)
        return update

    graph: StateGraph = StateGraph(TriageState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("analyze_priority", analyze_priority)
    graph.add_node("retrieve_similar", retrieve_similar)
    graph.add_node("draft_solution", draft_solution)
    graph.add_node("decide_route", decide_route)
    graph.add_node("record_audit", record_audit)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "analyze_priority")
    graph.add_edge("analyze_priority", "retrieve_similar")
    graph.add_edge("retrieve_similar", "draft_solution")
    graph.add_edge("draft_solution", "decide_route")
    graph.add_edge("decide_route", "record_audit")
    graph.add_edge("record_audit", END)
    return graph.compile()


def build_pipeline(
    settings: Settings,
    *,
    store: TicketStore,
    audit: AuditStore,
    classifier: IntentClassifier | None = None,
    client: ClaudeClient | None = None,
) -> TriagePipeline:
    """Assemble a :class:`TriagePipeline` from ``settings`` and the given stores.

    Args:
        settings: Effective configuration.
        store: Similar-ticket store (already indexed or to be indexed by the caller).
        audit: Audit store.
        classifier: Override for the classifier (default: :func:`build_classifier`).
        client: Override for the Claude client. When ``None`` and an API key is
            configured a client is created; without a key the analyzer and the
            drafter run their deterministic fallbacks.
    """
    if client is None and settings.anthropic_api_key:
        client = ClaudeClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_s=settings.llm_timeout_s,
            max_attempts=settings.llm_max_attempts,
        )
    clf = classifier or build_classifier(settings)
    analyzer = PrioritySentimentAnalyzer(client)
    drafter = Drafter(client)
    graph = build_graph(
        settings=settings,
        classifier=clf,
        analyzer=analyzer,
        drafter=drafter,
        store=store,
        audit=audit,
    )
    if settings.langsmith_enabled:
        trace_logger().info("LangSmith tracing enabled project={}", settings.langsmith_project)
    return TriagePipeline(
        settings=settings,
        classifier=clf,
        analyzer=analyzer,
        drafter=drafter,
        store=store,
        audit=audit,
        graph=graph,
    )
