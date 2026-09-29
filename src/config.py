"""Environment-driven configuration for the triage service.

Every knob the service reads from the environment lives here so that the
domain layer never touches ``os.environ`` directly. Values are loaded from a
local ``.env`` file when present; see ``.env.example`` for the documented
list.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Product model, pinned to a dated snapshot so evaluation runs are reproducible.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

RetrievalBackend = Literal["memory", "qdrant"]
PersistenceBackend = Literal["memory", "postgres"]
EncoderBackend = Literal["deterministic", "hf"]


class Settings(BaseSettings):
    """Typed view over the process environment.

    Attributes:
        anthropic_api_key: Anthropic API key. When ``None`` every LLM-backed
            component falls back to its deterministic heuristic and the run is
            labelled as such in logs and evaluation reports.
        anthropic_model: Dated Claude model ID used for every LLM call.
        llm_timeout_s: Per-request HTTP timeout for Claude calls, in seconds.
        llm_max_attempts: Total attempts (first try + retries) for a Claude call.
        price_input_per_mtok: USD per million input tokens for ``anthropic_model``.
        price_output_per_mtok: USD per million output tokens for ``anthropic_model``.
        classifier_model: HF Hub ID (or local path) of the fine-tuned classifier.
        use_local_classifier: Try to load the HF classifier before falling back.
        encoder_backend: Embedding backend for similar-ticket retrieval.
        retrieval_backend: Where ticket embeddings are stored.
        qdrant_url: Qdrant HTTP endpoint (used when ``retrieval_backend="qdrant"``).
        qdrant_collection: Qdrant collection holding resolved tickets.
        persistence_backend: Where the audit log is persisted.
        database_url: PostgreSQL DSN (used when ``persistence_backend="postgres"``).
        db_pool_min_size: Minimum connections kept open by the psycopg pool.
        db_pool_max_size: Maximum connections opened by the psycopg pool.
        auto_resolve_threshold: Confidence at or above which a ticket auto-resolves.
        escalate_threshold: Confidence below which a ticket escalates.
        cors_origins: Comma-separated list of allowed browser origins.
        rate_limit: ``slowapi`` rate-limit string applied to ``POST /api/triage``.
        log_level: Minimum loguru level emitted to stderr.
        langsmith_tracing: Mirrors ``LANGCHAIN_TRACING_V2``; LangGraph sends traces
            to LangSmith when this is true and ``langsmith_api_key`` is set.
        langsmith_api_key: LangSmith API key (never logged).
        langsmith_project: LangSmith project name for the traces.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, alias="ANTHROPIC_MODEL")
    llm_timeout_s: float = Field(default=15.0, gt=0, alias="LLM_TIMEOUT_S")
    llm_max_attempts: int = Field(default=3, ge=1, le=6, alias="LLM_MAX_ATTEMPTS")
    price_input_per_mtok: float = Field(default=3.0, ge=0, alias="PRICE_INPUT_PER_MTOK")
    price_output_per_mtok: float = Field(default=15.0, ge=0, alias="PRICE_OUTPUT_PER_MTOK")

    classifier_model: str = Field(
        default="Juadsuarezsan/distilbert-support-intent-lora", alias="CLASSIFIER_MODEL"
    )
    use_local_classifier: bool = Field(default=True, alias="USE_LOCAL_CLASSIFIER")

    encoder_backend: EncoderBackend = Field(default="deterministic", alias="ENCODER_BACKEND")
    retrieval_backend: RetrievalBackend = Field(default="memory", alias="RETRIEVAL_BACKEND")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="resolved_tickets", alias="QDRANT_COLLECTION")

    persistence_backend: PersistenceBackend = Field(default="memory", alias="PERSISTENCE_BACKEND")
    database_url: str = Field(
        default="postgresql://support:support_dev@localhost:5432/support", alias="DATABASE_URL"
    )
    db_pool_min_size: int = Field(default=1, ge=0, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=8, ge=1, alias="DB_POOL_MAX_SIZE")

    auto_resolve_threshold: float = Field(default=0.85, ge=0, le=1, alias="AUTO_RESOLVE_THRESHOLD")
    escalate_threshold: float = Field(default=0.60, ge=0, le=1, alias="ESCALATE_THRESHOLD")

    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    rate_limit: str = Field(default="60/minute", alias="RATE_LIMIT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    langsmith_tracing: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langsmith_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="support-triage-agent", alias="LANGCHAIN_PROJECT")

    @field_validator("escalate_threshold")
    @classmethod
    def _escalate_below_auto(cls, v: float, info: ValidationInfo) -> float:
        """Reject configurations where the suggest band is empty or inverted."""
        auto = info.data.get("auto_resolve_threshold")
        if auto is not None and v >= auto:
            raise ValueError("ESCALATE_THRESHOLD must be lower than AUTO_RESOLVE_THRESHOLD")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        """Return the CORS origins as a cleaned list (never ``["*"]``)."""
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return [o for o in origins if o != "*"]

    @property
    def llm_enabled(self) -> bool:
        """Whether Claude-backed components can run (an API key is configured)."""
        return bool(self.anthropic_api_key)

    @property
    def langsmith_enabled(self) -> bool:
        """Whether LangGraph will ship traces to LangSmith for this process."""
        return self.langsmith_tracing and bool(self.langsmith_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Tests that mutate environment variables call ``get_settings.cache_clear()``
    afterwards so the next caller re-reads the environment.
    """
    return Settings()
