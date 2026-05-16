"""Public schemas for the triage API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal["email", "chat", "slack", "twitter"]
Priority = Literal["P0", "P1", "P2", "P3"]
Sentiment = Literal["neg", "neu", "pos"]
Decision = Literal["auto_resolve", "suggest", "escalate"]

# 27 intents from the Bitext dataset
BITEXT_INTENTS: tuple[str, ...] = (
    "cancel_order", "change_order", "check_invoice", "check_payment_methods",
    "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account", "delivery_options",
    "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address",
    "switch_account", "track_order", "track_refund",
    "report_payment_issue", "general_question",
)


class TicketIn(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    channel: Channel = "email"
    body: str = Field(..., min_length=1, max_length=5000)
    customer_email: str | None = None


class IntentScore(BaseModel):
    intent: str
    score: float = Field(..., ge=0.0, le=1.0)


class SimilarTicket(BaseModel):
    ticket_id: str
    intent: str
    body_snippet: str
    resolution_snippet: str
    similarity: float


class TriageDecision(BaseModel):
    decision: Decision
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class TriageOut(BaseModel):
    ticket_id: str
    intent: str
    intent_confidence: float
    top_intents: list[IntentScore] = Field(default_factory=list)
    priority: Priority
    sentiment: Sentiment
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    similar_resolved: list[SimilarTicket] = Field(default_factory=list)
    draft_response: str | None = None
    decision: TriageDecision
    latency_ms: int = 0
    cost_usd: float = 0.0
