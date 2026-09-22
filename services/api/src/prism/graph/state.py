"""The graph's state: what a fork must re-enter with.

Everything here is serialized into a checkpoint. Anything that can be re-fetched
by id is, so candidates are references rather than hydrated chunks (ADR 0017).
"""

from typing import Annotated, Literal, TypedDict
from uuid import UUID

from prism.config import Settings

__all__ = [
    "CandidateRef",
    "GraphState",
    "RefusalReason",
    "SearchParams",
    "TerminalStatus",
    "monotonic",
    "search_params_from",
]

TerminalStatus = Literal["cached", "answered", "refused"]
RefusalReason = Literal["no_relevant_evidence", "insufficient_evidence"]


def monotonic(current: int, incoming: int) -> int:
    """Reducer for counters that only go up. Without one, concurrent writes error."""
    return max(current, incoming)


class SearchParams(TypedDict):
    """What `plan_query` resolved this run to retrieve and correct with.

    In state rather than read from `Settings` per node, so a fork runs the
    parameters the original run used (ADR 0017).

    Model names are not here by choice: which model answers is the cost-aware
    router's decision, taken per call (Phase 3).
    """

    k: int
    candidate_k: int
    rrf_k: int
    # The fused pool `rerank` scores, before it cuts to `k` (ADR 0020).
    rerank_candidate_k: int
    rerank_score_floor: float
    # Read by `grade_docs` and by `after_grade_docs`. Not widths, same argument.
    doc_relevance_threshold: float
    max_attempts: int


def search_params_from(settings: Settings) -> SearchParams:
    """The configured defaults, resolved once per run by `plan_query`."""
    return SearchParams(
        k=settings.retrieval_top_k,
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
        rerank_candidate_k=settings.rerank_candidate_k,
        rerank_score_floor=settings.rerank_score_floor,
        doc_relevance_threshold=settings.doc_relevance_threshold,
        max_attempts=settings.max_attempts,
    )


class CandidateRef(TypedDict):
    """One candidate, by reference.

    Fusion positions carry no magnitude (ADR 0010). `rerank_score` is the first
    real one in the pipeline: `retrieve` writes None, `rerank` fills it in, and
    generation and `query_citations.rerank_score` read it from here.

    Chunk text is hydrated by id where it is needed.
    """

    chunk_id: UUID
    document_id: UUID
    rank: int
    vector_rank: int | None
    lexical_rank: int | None
    rerank_score: float | None


class GraphState(TypedDict):
    query_id: UUID
    tenant_id: UUID
    collection_id: UUID
    thread_id: str

    # What the user asked. Written once at mint and never again: grading and
    # generation judge against this, so a rewrite cannot move the goalposts it
    # is about to be judged against (ADR 0019).
    question: str
    # What retrieval currently searches for. Seeded to `question`, replaced by
    # `rewrite_query`. Read by plan_query, embed_query and retrieve, and by
    # nothing that judges.
    retrieval_query: str

    # The two correction loops count independently.
    retrieval_attempts: Annotated[int, monotonic]
    grounding_attempts: Annotated[int, monotonic]

    # The last sequence written; the trace writer takes the next from here.
    sequence: Annotated[int, monotonic]

    # Replaced, not accumulated, on each pass of the rewrite loop. No reducer:
    # LastValue replaces and raises on two writes in one step (ADR 0017).
    search_terms: list[str]
    search_params: SearchParams
    query_embedding: list[float] | None
    candidates: list[CandidateRef]

    status: TerminalStatus
    refusal_reason: RefusalReason | None
