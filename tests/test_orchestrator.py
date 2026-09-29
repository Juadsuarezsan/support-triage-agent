"""End-to-end graph runs: offline and with a mocked Claude client."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import respx

from src.agents.orchestrator import TriagePipeline, build_pipeline
from src.api.schemas import TicketIn
from src.config import Settings
from src.eval.runner import build_offline_pipeline
from src.llm.claude import ClaudeClient
from src.observability import TokenUsage, estimate_cost_usd
from tests.conftest import MESSAGES_URL, claude_response


async def test_offline_complaint_escalates_and_is_audited(offline_pipeline: TriagePipeline) -> None:
    out = await offline_pipeline.triage(
        TicketIn(
            ticket_id="c-1",
            body="This is unacceptable, I want to file a complaint",
            channel="slack",
        )
    )
    assert out.intent == "complaint"
    assert out.decision.decision == "escalate"
    assert out.priority in ("P0", "P1") and out.sentiment == "neg"
    assert out.llm_used is False and out.cost_usd == 0.0
    assert out.classifier_backend == "keyword_heuristic"
    [row] = await offline_pipeline.audit.recent(1)
    assert row.trace_id == out.trace_id and row.channel == "slack"
    assert row.decision == "escalate" and row.similar_ticket_ids


async def test_offline_routine_ticket_gets_draft_from_similar(
    offline_pipeline: TriagePipeline,
) -> None:
    out = await offline_pipeline.triage(TicketIn(ticket_id="r-1", body="Where is my order #66721?"))
    assert out.intent == "track_order"
    assert out.similar_resolved[0].intent == "track_order"
    assert out.draft_response and "track_order" in out.draft_response
    assert out.latency_ms >= 0 and len(out.top_intents) == 1


async def test_llm_pipeline_accounts_tokens_and_cost(
    anthropic_mock: respx.MockRouter, make_client: Callable[..., ClaudeClient]
) -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["system"]
        if "27 intents" in system:
            text = json.dumps({"top_intents": [{"intent": "get_refund", "score": 0.93}]})
        elif "urgency_score" in system:
            text = json.dumps(
                {"sentiment": "neu", "priority": "P2", "urgency_score": 0.3, "rationale": "ok"}
            )
        else:
            text = "Sorry about the broken item; a refund is on its way. Next step: keep the box."
        return httpx.Response(200, json=claude_response(text, input_tokens=100, output_tokens=20))

    anthropic_mock.post(MESSAGES_URL).mock(side_effect=reply)
    settings = Settings(anthropic_api_key="k", use_local_classifier=False)
    pipeline = await build_offline_pipeline(settings, client=make_client())
    out = await pipeline.triage(
        TicketIn(ticket_id="l-1", body="I want a refund for my broken item")
    )
    assert out.classifier_backend == "claude_zero_shot"
    assert out.intent == "get_refund" and out.intent_confidence == 0.93
    assert out.llm_used is True
    assert (out.input_tokens, out.output_tokens) == (300, 60)  # 3 calls
    expected = estimate_cost_usd(
        TokenUsage(300, 60, 3), price_input_per_mtok=3.0, price_output_per_mtok=15.0
    )
    assert out.cost_usd == expected > 0
    assert out.draft_response and out.draft_response.startswith("Sorry")
    assert out.decision.decision == "auto_resolve"


def test_build_pipeline_creates_client_when_key_present(mocker) -> None:
    from src.persistence.audit import InMemoryAuditStore
    from src.retrieval.encoders import DeterministicEncoder
    from src.retrieval.similar_tickets import InMemoryTicketStore

    settings = Settings(
        anthropic_api_key="k",
        use_local_classifier=False,
        langsmith_tracing=True,
        langsmith_api_key="ls",
    )
    p = build_pipeline(
        settings, store=InMemoryTicketStore(DeterministicEncoder()), audit=InMemoryAuditStore()
    )
    assert p.llm_enabled and p.classifier.name == "claude_zero_shot"
    assert p.settings.langsmith_enabled
