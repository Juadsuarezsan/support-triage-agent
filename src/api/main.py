"""FastAPI transport layer for the triage agent.

The module-level ``app`` is built from the process environment. Tests build
their own instance with :func:`create_app` and explicit :class:`Settings`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agents.orchestrator import TriagePipeline, build_pipeline
from src.api.schemas import AuditRecord, HealthOut, TicketIn, TriageOut
from src.config import Settings, get_settings
from src.errors import TriageError
from src.observability import configure_logging, trace_logger
from src.persistence.audit import AuditStore, InMemoryAuditStore, open_postgres_store
from src.retrieval.encoders import build_encoder
from src.retrieval.qdrant_store import build_qdrant_store
from src.retrieval.similar_tickets import InMemoryTicketStore, TicketStore, load_jsonl

API_VERSION = "0.6.0"
ROOT = Path(__file__).resolve().parent.parent.parent
SEED_TICKETS = ROOT / "data" / "eval" / "seed_tickets.jsonl"


async def build_ticket_store(settings: Settings) -> TicketStore:
    """Instantiate the retrieval backend named by ``settings`` and seed it."""
    encoder = build_encoder(settings.encoder_backend)
    store: TicketStore
    if settings.retrieval_backend == "qdrant":
        store = build_qdrant_store(encoder, settings.qdrant_url, settings.qdrant_collection)
    else:
        store = InMemoryTicketStore(encoder)
    if SEED_TICKETS.exists():
        indexed = await store.index(load_jsonl(SEED_TICKETS))
        trace_logger().info("indexed {} seed tickets into {}", indexed, store.name)
    return store


async def build_audit_store(settings: Settings) -> AuditStore:
    """Instantiate the persistence backend named by ``settings``."""
    if settings.persistence_backend == "postgres":
        return await open_postgres_store(
            settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )
    return InMemoryAuditStore()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Explicit configuration; defaults to the environment.
    """
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)
    limiter = Limiter(key_func=get_remote_address)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = await build_ticket_store(cfg)
        audit = await build_audit_store(cfg)
        app.state.pipeline = build_pipeline(cfg, store=store, audit=audit)
        trace_logger().info(
            "startup model={} llm_enabled={} classifier={} retrieval={} persistence={}",
            cfg.anthropic_model,
            cfg.llm_enabled,
            app.state.pipeline.classifier.name,
            store.name,
            audit.name,
        )
        try:
            yield
        finally:
            await audit.close()

    app = FastAPI(
        title="Customer Support Triage Agent",
        version=API_VERSION,
        description=(
            "Intent classification, sentiment/priority analysis, similar-ticket retrieval, "
            "reply drafting and confidence-gated routing (auto-resolve / suggest / escalate)."
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.limiter = limiter

    def rate_limited(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RateLimitExceeded)
        return _rate_limit_exceeded_handler(request, exc)

    app.add_exception_handler(RateLimitExceeded, rate_limited)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    def pipeline() -> TriagePipeline:
        p: TriagePipeline = app.state.pipeline
        return p

    @app.get("/health", response_model=HealthOut)
    async def health() -> HealthOut:
        """Liveness/readiness probe with the effective backends."""
        p = pipeline()
        return HealthOut(
            status="ok",
            version=API_VERSION,
            model=cfg.anthropic_model,
            classifier_backend=p.classifier.name,
            retrieval_backend=p.store.name,
            persistence_backend=p.audit.name,
            llm_enabled=cfg.llm_enabled,
            langsmith_enabled=cfg.langsmith_enabled,
        )

    @app.post("/api/triage", response_model=TriageOut)
    @limiter.limit(cfg.rate_limit)
    async def triage(request: Request, ticket: TicketIn) -> TriageOut:
        """Triage one ticket. Invalid bodies are rejected with 422 by Pydantic."""
        try:
            return await pipeline().triage(ticket)
        except TriageError as exc:
            trace_logger().error("triage failed: {}", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/audit/recent", response_model=list[AuditRecord])
    async def audit_recent(limit: int = 50) -> list[AuditRecord]:
        """Newest audit rows (feeds the Kanban demo)."""
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        try:
            return await pipeline().audit.recent(limit)
        except TriageError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/eval/run")
    async def run_eval_endpoint() -> dict[str, Any]:
        """Run the offline evaluation against the in-process pipeline."""
        from src.eval.runner import run_pipeline_eval

        report = await run_pipeline_eval(pipeline())
        return report.model_dump(exclude={"cases"})

    return app


load_dotenv()
app = create_app()
