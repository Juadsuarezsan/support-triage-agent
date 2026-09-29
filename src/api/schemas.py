"""Public Pydantic schemas for the triage API and the internal domain records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal["email", "chat", "slack", "twitter"]
Priority = Literal["P0", "P1", "P2", "P3"]
Sentiment = Literal["neg", "neu", "pos"]
Decision = Literal["auto_resolve", "suggest", "escalate"]

#: The 27 intents of the Bitext customer-support dataset, in the exact order
#: used by ``models/intent-classifier-lora/label_mapping.json``.
BITEXT_INTENTS: tuple[str, ...] = (
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
)

#: Label returned when no backend can name an intent with any confidence.
UNKNOWN_INTENT = "unknown"

#: Maximum accepted ticket length; longer bodies are rejected with HTTP 422.
MAX_TICKET_CHARS = 5000


class TicketIn(BaseModel):
    """Inbound support ticket.

    Attributes:
        ticket_id: Caller-provided identifier (non-empty).
        channel: Origin channel of the ticket.
        body: Free-text ticket body, 1 to 5000 characters.
        customer_email: Optional customer email, echoed to the audit log only.
    """

    ticket_id: str = Field(..., min_length=1, max_length=64)
    channel: Channel = "email"
    body: str = Field(..., min_length=1, max_length=MAX_TICKET_CHARS)
    customer_email: str | None = Field(default=None, max_length=254)


class IntentScore(BaseModel):
    """One intent label with its probability-like score."""

    intent: str
    score: float = Field(..., ge=0.0, le=1.0)


class SimilarTicket(BaseModel):
    """A previously resolved ticket returned by the retriever."""

    ticket_id: str
    intent: str
    body_snippet: str
    resolution_snippet: str
    similarity: float


class TriageDecision(BaseModel):
    """Routing decision with the confidence that produced it."""

    decision: Decision
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class TriageOut(BaseModel):
    """Full triage result returned by ``POST /api/triage``.

    Attributes:
        trace_id: Unique ID shared by every log line and audit row of the request.
        classifier_backend: Which classifier produced ``intent``.
        llm_used: Whether any step called Claude (false means deterministic fallback).
        input_tokens: Claude input tokens billed for the whole request.
        output_tokens: Claude output tokens billed for the whole request.
        cost_usd: Estimated USD cost of the request (zero without an API key).
    """

    ticket_id: str
    trace_id: str
    intent: str
    intent_confidence: float = Field(..., ge=0.0, le=1.0)
    top_intents: list[IntentScore] = Field(default_factory=list)
    classifier_backend: str
    priority: Priority
    sentiment: Sentiment
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    similar_resolved: list[SimilarTicket] = Field(default_factory=list)
    draft_response: str | None = None
    decision: TriageDecision
    llm_used: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class AuditRecord(BaseModel):
    """Row persisted by the audit logger for every triaged ticket."""

    trace_id: str
    ticket_id: str
    channel: Channel = "email"
    body: str
    intent: str
    intent_confidence: float
    priority: Priority
    sentiment: Sentiment
    urgency_score: float
    decision: Decision
    confidence: float
    rationale: str
    draft_response: str | None
    similar_ticket_ids: list[str] = Field(default_factory=list)
    llm_used: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: datetime


class HealthOut(BaseModel):
    """Payload of ``GET /health``."""

    status: Literal["ok"]
    version: str
    model: str
    classifier_backend: str
    retrieval_backend: str
    persistence_backend: str
    llm_enabled: bool
    langsmith_enabled: bool
