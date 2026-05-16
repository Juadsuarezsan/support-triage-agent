"""Support Triage API — placeholder until v0.1.0 build out."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.config import get_settings

TriageDecision = Literal["auto_resolve", "suggest", "escalate"]


class TicketIn(BaseModel):
    ticket_id: str
    channel: Literal["email", "chat", "slack", "twitter"] = "email"
    body: str = Field(..., min_length=1, max_length=5000)


class TriageOut(BaseModel):
    ticket_id: str
    intent: str
    intent_confidence: float
    priority: Literal["P0", "P1", "P2", "P3"]
    sentiment: Literal["neg", "neu", "pos"]
    decision: TriageDecision
    draft_response: str | None = None
    similar_resolved_tickets: list[dict] = []
    latency_ms: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Customer Support Triage Agent",
    version="0.1.0",
    description="DistilBERT+LoRA intent classifier + Claude reasoning + similar-ticket retrieval.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "stage": "scaffolding",
        "classifier_model": s.classifier_model,
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/triage", response_model=TriageOut)
async def triage(ticket: TicketIn) -> TriageOut:
    # TODO: classifier → priority/sentiment → similar tickets → drafter → decision
    return TriageOut(
        ticket_id=ticket.ticket_id,
        intent="not_yet_implemented",
        intent_confidence=0.0,
        priority="P3",
        sentiment="neu",
        decision="escalate",
        draft_response=None,
        similar_resolved_tickets=[],
        latency_ms=0,
    )
