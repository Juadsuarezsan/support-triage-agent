from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    classifier_model: str = Field(
        default="Juadsuarezsan/distilbert-support-intent-lora",
        alias="CLASSIFIER_MODEL",
    )
    use_local_classifier: bool = Field(default=True, alias="USE_LOCAL_CLASSIFIER")

    database_url: str = Field(
        default="postgresql://support:support_dev@localhost:5432/support",
        alias="DATABASE_URL",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")

    auto_resolve_threshold: float = Field(default=0.85, alias="AUTO_RESOLVE_THRESHOLD")
    escalate_threshold: float = Field(default=0.60, alias="ESCALATE_THRESHOLD")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
