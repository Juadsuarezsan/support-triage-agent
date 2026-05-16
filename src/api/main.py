"""FastAPI: triage endpoint + eval endpoint."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from src.agents.orchestrator import build_graph, triage_ticket
from src.api.schemas import TicketIn, TriageOut
from src.config import get_settings
from src.retrieval.similar_tickets import DeterministicEncoder, InMemoryTicketStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use the deterministic encoder by default so the app starts without
    # downloading the ~420 MB sentence-transformer model. Swap encoder=None in
    # production for the real semantic similarity.
    store = InMemoryTicketStore(encoder=DeterministicEncoder())
    seed_path = Path(__file__).parent.parent.parent / "data" / "eval" / "seed_tickets.jsonl"
    if seed_path.exists():
        seeds = [json.loads(l) for l in seed_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        await store.index(seeds)
        logger.info(f"Indexed {store.count()} seed tickets at startup")
    app.state.store = store
    app.state.graph = build_graph(store)
    yield


app = FastAPI(
    title="Customer Support Triage Agent",
    version="0.5.0",
    description="Intent classifier + sentiment/priority + similar-ticket retrieval + decision routing.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.5.0",
        "stage": "substantive",
        "classifier_model": s.classifier_model,
        "use_local_classifier": str(s.use_local_classifier),
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/triage", response_model=TriageOut)
async def triage(ticket: TicketIn) -> TriageOut:
    try:
        return await triage_ticket(app.state.graph, ticket_id=ticket.ticket_id, body=ticket.body)
    except Exception as exc:  # noqa: BLE001
        logger.exception("triage failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eval/run")
async def run_eval_endpoint() -> dict:
    from src.eval.runner import run_eval
    return await run_eval(use_fast_encoder=True)
