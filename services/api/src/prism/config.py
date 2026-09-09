"""Runtime configuration, loaded from the environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # .env.example ships every key with an empty value. Without this, `KEY=`
        # parses as "" — a hard error on every numeric field, and a silent empty
        # string on every text one.
        env_ignore_empty=True,
    )

    prism_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://prism:prism@localhost:5433/prism"
    redis_url: str = "redis://localhost:6380/0"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    dep_check_timeout_s: float = 2.0
    # Generous next to the query budget: a cold model load is a one-off, and a
    # slow embed must surface as an error, never as a missing vector.
    embed_timeout_s: float = 30.0

    planner_model: str = "qwen2.5:14b"  # plan_query, rewrite_query
    grader_model: str = "llama3.1:8b"  # grade_docs, verify_grounding
    generator_model: str = "qwen2.5:32b"  # generate, before escalation
    reranker_model: str = "bge-reranker-v2-m3"  # in-process

    # Correction-loop policy. max_attempts is total attempts per loop, and the
    # two loops count independently.
    abstention_threshold: float = 0.58
    max_attempts: int = 3
    rerank_score_floor: float = 0.44

    # Per-query budgets for the cost-aware router. No eligible provider under
    # both is a refusal, not an overspend.
    cost_budget_usd: float = 0.0050
    latency_budget_s: float = 6.0

    # Unset in dev, always unset in CI.
    ollama_cloud_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # Ollama Cloud's cap of 1 is an external constraint, not a tuning knob.
    concurrency_ollama_local: int = 8
    concurrency_ollama_cloud: int = 1
    concurrency_gemini: int = 4
    concurrency_openai: int = 2

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
