"""Runtime configuration, loaded from the environment."""

from functools import lru_cache
from pathlib import Path
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

    # Relative paths resolve against the working directory; /app in the container.
    storage_dir: Path = Path("var/uploads")
    max_upload_bytes: int = 25 * 1024 * 1024

    # The HNSW knobs bind only if the planner picks that index over the
    # collection_id btree; see prism.retrieval.search.
    retrieval_top_k: int = 10
    # Each half retrieves this deep before fusion, so a chunk one half ranks
    # well survives the other half missing it entirely.
    retrieval_candidate_k: int = 30
    # RRF consumes ranks, never scores; see ADR 0010.
    rrf_k: int = 60
    hnsw_ef_search: int = 64
    hnsw_iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "strict_order"

    # Characters, not tokens — chunking is deterministic and tokenizer-free.
    # Pages are chunked independently, so these bound a single page's windows.
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    embed_batch_size: int = 64

    vision_enabled: bool = False
    # Local by default; gemini is the paid escalation and the only lane that
    # spends money.
    vision_lane: Literal["ollama", "gemini"] = "ollama"
    vision_ollama_model: str = "qwen2.5vl:7b"
    vision_model: str = "gemini-3.6-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Ingestion is not subject to the per-query latency budget.
    vision_timeout_s: float = 90.0
    # 150dpi keeps 8pt table text legible without tripling the payload.
    vision_render_dpi: int = 150
    # Path objects that make a page worth a paid call. The cost/recall knob:
    # a booktabs table is three or four rules.
    vision_min_path_objects: int = 6
    vision_max_pages: int = 20  # per-document ceiling on paid calls

    planner_model: str = "qwen2.5:14b"  # plan_query, rewrite_query
    grader_model: str = "llama3.1:8b"  # grade_docs, verify_grounding
    generator_model: str = "qwen2.5:32b"  # generate, before escalation
    reranker_model: str = "bge-reranker-v2-m3"  # in-process

    # rerank_score_floor is calibrated against this revision and quantization;
    # changing either re-opens it (ADR 0011).
    reranker_repo: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    reranker_revision: str = "6f5ff65298512715a1e669753bc754d2bc8f367b"
    reranker_quantization: Literal["int8"] = "int8"
    reranker_dir: Path = Path("var/models")
    # Fused candidates scored per query, at roughly 0.2s each on CPU: 30 would
    # spend the whole latency budget on rerank alone (eval/README.md).
    rerank_candidate_k: int = 10
    rerank_batch_size: int = 16
    rerank_max_tokens: int = 512
    # One forward pass already uses every core; a second only queues on them.
    rerank_concurrency: int = 1
    rerank_timeout_s: float = 3.0

    # Correction-loop policy. max_attempts is total attempts per loop, and the
    # two loops count independently.
    abstention_threshold: float = 0.58
    max_attempts: int = 3
    # Applies to the reranker's sigmoid, never its raw logit.
    rerank_score_floor: float = 0.44

    # Per-query budgets for the cost-aware router. No eligible provider under
    # both is a refusal, not an overspend.
    cost_budget_usd: float = 0.0050
    latency_budget_s: float = 6.0

    # Bootstraps the first tenant, which no tenant key can do. Unset by default:
    # the tenant routes 503 rather than fall back to a weaker check.
    admin_token: str | None = None

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
