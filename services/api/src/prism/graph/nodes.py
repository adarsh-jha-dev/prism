"""The graph's nodes.

`plan_query`, `embed_query` and `retrieve` are implemented; the rest are stubs
writing one trace row each. No conditional edges yet, so every run ends at
`abstain`.

Nodes report through the TraceContext (ADR 0016), as references rather than
copies (ADR 0012), and call models through the registry (ADR 0018).
"""

import math
from typing import Any

import structlog
from pydantic import BaseModel, Field

from prism.chat import Message
from prism.config import get_settings
from prism.graph.state import CandidateRef, GraphState, search_params_from
from prism.graph.trace import TraceContext, traced
from prism.providers import get_registry
from prism.retrieval.hybrid import hybrid_search

__all__ = [
    "QueryPlan",
    "abstain",
    "embed_query",
    "generate",
    "grade_docs",
    "plan_query",
    "rerank",
    "retrieve",
    "rewrite_query",
    "verify_grounding",
]

log = structlog.get_logger(__name__)

# `generate`'s lane is the Phase 3 router's decision.
_LOCAL_LANE = "ollama"

_PLANNER_SYSTEM = (
    "You extract search terms for a full-text index. The index requires EVERY "
    "term to appear in the same passage, so fewer and rarer terms retrieve more. "
    "Return only terms that appear in the question or are its obvious singular "
    "form. Prefer proper nouns, technical terms and numbers. Never add synonyms, "
    "never add words the question does not contain, and never include common "
    "words such as what, how, does, the, is, using or between."
)

_PLANNER_MAX_TOKENS = 200


class QueryPlan(BaseModel):
    terms: list[str] = Field(default_factory=list)


@traced("plan_query")
async def plan_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Extract the lexical half's search terms and pin the retrieval parameters.

    `websearch_to_tsquery` ANDs its terms, so fewer and rarer terms retrieve more
    (ADR 0010). Parameters are pinned here so a fork retrieves at the width the
    original run used (ADR 0017).
    """
    settings = get_settings()
    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_PLANNER_SYSTEM),
            Message(role="user", content=state["question"]),
        ],
        QueryPlan,
        model=settings.planner_model,
        max_tokens=_PLANNER_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    terms = _clean_terms(result.value.terms, max_terms=settings.planner_max_terms)
    params = search_params_from(settings)

    trace.record_input({"question": state["question"]})
    trace.record_output({"terms": terms, "params": dict(params)})
    return {"search_terms": terms, "search_params": params}


def _clean_terms(terms: list[str], *, max_terms: int) -> list[str]:
    """Strip, de-duplicate case-insensitively, and keep the first `max_terms`.

    An empty result is normal: `retrieve` then parses the raw question.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        cleaned = " ".join(term.split())
        # Longer than this is an echoed sentence, which ANDs to nothing.
        if not cleaned or len(cleaned) > 64:
            continue
        if cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        kept.append(cleaned)
    return kept[:max_terms]


@traced("embed_query")
async def embed_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Embed the question once, for retrieval and for every replay of it.

    The vector goes into state rather than being recomputed in `retrieve`, which
    would misattribute the meter and break replay (ADR 0017).
    """
    result = await get_registry().embed(_LOCAL_LANE, [state["question"]])
    trace.record_usage(result.usage)
    vector = result.value[0]

    # The vector itself is 768 floats of tenant-derived data and belongs in no
    # payload. Its width and norm are what a reader of the row wants anyway:
    # a wrong width means the wrong model, and a zero norm means a dead vector.
    norm = math.sqrt(sum(x * x for x in vector))
    trace.record_input({"question": state["question"]})
    trace.record_output({"dim": len(vector), "norm": round(norm, 6)})
    return {"query_embedding": vector}


@traced("retrieve")
async def retrieve(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Hybrid FTS + HNSW, fused by RRF. Fusion only — `rerank` is its own node.

    An empty candidate set is a normal outcome, not an error: the run proceeds
    and refuses.
    """
    params = state["search_params"]
    terms = state["search_terms"]

    hits = await hybrid_search(
        state["question"],
        tenant_id=state["tenant_id"],
        collection_id=state["collection_id"],
        terms=terms or None,
        k=params["k"],
        candidate_k=params["candidate_k"],
        rrf_k=params["rrf_k"],
        query_vector=state["query_embedding"],
    )

    candidates: list[CandidateRef] = [
        CandidateRef(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            rank=hit.rank,
            vector_rank=hit.vector_rank,
            lexical_rank=hit.lexical_rank,
        )
        for hit in hits
    ]

    trace.record_input({"terms": terms, **params})
    # Ids and positions: fusion yields no magnitude (ADR 0010), and chunk text is
    # already a row in `chunks` (ADR 0012).
    trace.record_output([dict(candidate) for candidate in candidates])
    if not candidates:
        log.info(
            "retrieve.empty",
            query_id=str(state["query_id"]),
            collection_id=str(state["collection_id"]),
            terms=terms,
        )
    return {"candidates": candidates}


@traced("grade_docs")
async def grade_docs(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("rewrite_query")
async def rewrite_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("rerank")
async def rerank(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("generate")
async def generate(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("verify_grounding")
async def verify_grounding(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("abstain")
async def abstain(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {"status": "refused", "refusal_reason": "no_relevant_evidence"}
