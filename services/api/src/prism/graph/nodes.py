"""The graph's nodes.

The retrieval loop is implemented — `plan_query`, `embed_query`, `retrieve`,
`grade_docs`, `rewrite_query` and `abstain`. `rerank`, `generate` and
`verify_grounding` are still stubs writing one trace row each, so the pass path
runs through them and refuses at `abstain` for want of a citation.

Retrieval reads `retrieval_query`, which the loop rewrites. Grading and
generation read `question`, which nothing rewrites (ADR 0019).

Nodes report through the TraceContext (ADR 0016), as references rather than
copies (ADR 0012), and call models through the registry (ADR 0018).
"""

import math
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from prism.chat import Message
from prism.config import get_settings
from prism.graph.state import (
    CandidateRef,
    GraphState,
    RefusalReason,
    search_params_from,
)
from prism.graph.trace import TraceContext, traced
from prism.providers import get_registry
from prism.retrieval.hybrid import hybrid_search
from prism.retrieval.hydrate import HydratedChunk, hydrate_chunks

__all__ = [
    "ChunkRelevance",
    "QueryPlan",
    "QueryRewrite",
    "RelevanceVerdicts",
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

_GRADER_SYSTEM = (
    "You judge whether each numbered passage contains evidence that helps answer "
    "the question. Score every passage from 0 to 1 by how directly it bears on "
    "the question, and return its number unchanged. A passage on the same broad "
    "topic that does not address the question scores low. Judge only what the "
    "passage says: never use outside knowledge, and never reward a passage for "
    "being well written. Return one entry per passage."
)

# Bounded by the candidate set: ten entries of a number and a score.
_GRADER_MAX_TOKENS = 400

_REWRITER_SYSTEM = (
    "The last search found nothing relevant. Rewrite it as a different search "
    "for the same question. Keep the user's intent exactly: never broaden it to "
    "a different question, never narrow it to one part of it, and never add "
    "facts or entities the question does not contain. Prefer the words a "
    "document answering this would use. Return the query only."
)

_REWRITER_MAX_TOKENS = 200


class QueryPlan(BaseModel):
    terms: list[str] = Field(default_factory=list)


class ChunkRelevance(BaseModel):
    """One passage's score, keyed by the label the prompt numbered it with."""

    label: int
    score: float


class RelevanceVerdicts(BaseModel):
    # min_length is load-bearing, not decoration: it becomes `minItems` in the
    # schema the provider constrains generation with, and without it
    # `llama3.1:8b` satisfies the schema with `{"verdicts": []}` every time —
    # the cheapest completion that validates. Every candidate would then fail
    # for want of a verdict and no query could ever be answered.
    #
    # It is also the honest contract: a grader scores every passage, and a
    # passage it finds irrelevant scores low rather than going unmentioned.
    verdicts: list[ChunkRelevance] = Field(min_length=1)


class QueryRewrite(BaseModel):
    query: str = ""


@traced("plan_query", attempts="retrieval_attempts")
async def plan_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Extract the lexical half's search terms, and pin the parameters once.

    `websearch_to_tsquery` ANDs its terms, so fewer and rarer terms retrieve more
    (ADR 0010). The rewrite loop re-enters here so that rule lives in one prompt
    rather than two that can diverge (ADR 0019).

    Parameters are pinned so a fork retrieves at the width the original run used
    (ADR 0017), which means the loop's later passes must not re-pin them: a
    second execution would re-read `Settings` and a fork taken after a change
    would retrieve at a width the original run never used.
    """
    settings = get_settings()
    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_PLANNER_SYSTEM),
            Message(role="user", content=state["retrieval_query"]),
        ],
        QueryPlan,
        model=settings.planner_model,
        max_tokens=_PLANNER_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    terms = _clean_terms(result.value.terms, max_terms=settings.planner_max_terms)
    update: dict[str, Any] = {"search_terms": terms}

    pinned = state["retrieval_attempts"] == 0
    params = search_params_from(settings) if pinned else state["search_params"]
    if pinned:
        update["search_params"] = params

    trace.record_input({"retrieval_query": state["retrieval_query"]})
    trace.record_output({"terms": terms, "params": dict(params), "pinned": pinned})
    return update


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


@traced("embed_query", attempts="retrieval_attempts")
async def embed_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Embed the retrieval query once, for retrieval and for every replay of it.

    The vector goes into state rather than being recomputed in `retrieve`, which
    would misattribute the meter and break replay (ADR 0017). A rewrite re-embeds
    because it asks a different question, not because it checkpointed again.
    """
    result = await get_registry().embed(_LOCAL_LANE, [state["retrieval_query"]])
    trace.record_usage(result.usage)
    vector = result.value[0]

    # The vector itself is 768 floats of tenant-derived data and belongs in no
    # payload. Its width and norm are what a reader of the row wants anyway:
    # a wrong width means the wrong model, and a zero norm means a dead vector.
    norm = math.sqrt(sum(x * x for x in vector))
    trace.record_input({"retrieval_query": state["retrieval_query"]})
    trace.record_output({"dim": len(vector), "norm": round(norm, 6)})
    return {"query_embedding": vector}


@traced("retrieve", attempts="retrieval_attempts")
async def retrieve(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Hybrid FTS + HNSW, fused by RRF. Fusion only — `rerank` is its own node.

    An empty candidate set is a normal outcome, not an error: the run proceeds
    and refuses.
    """
    params = state["search_params"]
    terms = state["search_terms"]

    hits = await hybrid_search(
        state["retrieval_query"],
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

    trace.record_input({"retrieval_query": state["retrieval_query"], "terms": terms, **params})
    # Ids and positions: fusion yields no magnitude (ADR 0010), and chunk text is
    # already a row in `chunks` (ADR 0012).
    trace.record_output([dict(candidate) for candidate in candidates])
    if not candidates:
        log.info(
            "retrieve.empty",
            query_id=str(state["query_id"]),
            collection_id=str(state["collection_id"]),
            terms=terms,
            attempt=state["retrieval_attempts"] + 1,
        )
    return {"candidates": candidates}


@traced("grade_docs", attempts="retrieval_attempts")
async def grade_docs(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Judge every candidate in one call, and keep the ones that pass.

    One call, not one per chunk: a trace row has one provider and one meter
    (ADR 0016), and thirty serialized grader calls do not fit the latency budget
    (ADR 0019). The set is judged against `question` — never `retrieval_query`,
    which the rewriter produced and which grading would otherwise let it define
    its own success by.

    Closes the attempt by incrementing `retrieval_attempts`. The surviving
    candidates are the graph's pass/fail signal: nothing the grader rejected
    reaches `rerank`, `generate` or a citation.
    """
    settings = get_settings()
    candidates = state["candidates"]
    attempt = state["retrieval_attempts"] + 1

    if not candidates:
        # Nothing to grade, and a grader asked to judge nothing invents a
        # verdict. No call, no meter, and a fail that costs nothing.
        trace.record_input({"candidates": []})
        trace.record_output({"verdicts": [], "kept": 0, "reason": "no_candidates"})
        return {"retrieval_attempts": attempt}

    chunks = await hydrate_chunks(
        [candidate["chunk_id"] for candidate in candidates],
        tenant_id=state["tenant_id"],
        collection_id=state["collection_id"],
    )

    if not chunks:
        # Retrieved ids that hydrate to nothing: deleted or re-ingested since
        # (ADR 0017). Still a fail, and still no call.
        trace.record_input({"candidates": [str(c["chunk_id"]) for c in candidates]})
        trace.record_output({"verdicts": [], "kept": 0, "reason": "no_hydrated_chunks"})
        return {"retrieval_attempts": attempt}

    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_GRADER_SYSTEM),
            Message(role="user", content=_grading_prompt(state["question"], chunks)),
        ],
        RelevanceVerdicts,
        model=settings.grader_model,
        max_tokens=_GRADER_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    scores = _scores_by_chunk(result.value, chunks)
    verdicts = [
        {
            "chunk_id": chunk.chunk_id,
            "score": scores.get(chunk.chunk_id),
            # Derived from one threshold, so the grader cannot both score 0.9
            # and call it irrelevant.
            "verdict": "pass"
            if (scores.get(chunk.chunk_id) or 0.0) >= settings.doc_relevance_threshold
            else "fail",
        }
        for chunk in chunks
    ]
    kept_ids = {v["chunk_id"] for v in verdicts if v["verdict"] == "pass"}
    kept = [candidate for candidate in candidates if candidate["chunk_id"] in kept_ids]

    trace.record_input({"candidates": [str(c["chunk_id"]) for c in candidates]})
    trace.record_output(
        {
            "verdicts": [{**v, "chunk_id": str(v["chunk_id"])} for v in verdicts],
            "kept": len(kept),
            "threshold": settings.doc_relevance_threshold,
        }
    )
    if not kept:
        log.info(
            "grade_docs.no_relevant_candidates",
            query_id=str(state["query_id"]),
            attempt=attempt,
            graded=len(chunks),
        )
    return {"retrieval_attempts": attempt, "candidates": kept}


def _grading_prompt(question: str, chunks: list[HydratedChunk]) -> str:
    """The candidates, numbered. The label is what a verdict comes back keyed by."""
    passages = "\n\n".join(
        f"[{label}] {chunk.content}" for label, chunk in enumerate(chunks, start=1)
    )
    return f"Question:\n{question}\n\nPassages:\n{passages}"


def _scores_by_chunk(verdicts: RelevanceVerdicts, chunks: list[HydratedChunk]) -> dict[UUID, float]:
    """Map labels back to chunks. Out-of-range labels are dropped.

    Keyed by an explicit label, never by position: a grader that returns nine
    verdicts for ten passages would otherwise shift every assignment by one and
    say so nowhere. A chunk with no verdict gets none, and an absent judgement
    is not evidence of relevance.
    """
    scores: dict[UUID, float] = {}
    for verdict in verdicts.verdicts:
        if 1 <= verdict.label <= len(chunks):
            # Out of 0-1 is thresholded as given rather than clamped: a clamp
            # would make a nonsense score indistinguishable from a confident one.
            scores[chunks[verdict.label - 1].chunk_id] = verdict.score
    return scores


@traced("rewrite_query", attempts="retrieval_attempts")
async def rewrite_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Produce the next retrieval query. Writes `retrieval_query`, never `question`.

    The rewriter sees what the user asked and what the last attempt searched for.
    It does not see the chunks that failed: they are another tenant read and
    another payload, and the loop is cheap enough without them for now.
    """
    settings = get_settings()
    before = state["retrieval_query"]

    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_REWRITER_SYSTEM),
            Message(
                role="user",
                content=f"Original question:\n{state['question']}\n\nLast search:\n{before}",
            ),
        ],
        QueryRewrite,
        model=settings.planner_model,
        max_tokens=_REWRITER_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    after = " ".join(result.value.query.split())
    if not after:
        # An empty rewrite is a wasted attempt either way; going back to what
        # the user asked is the one that does not compound the drift.
        after = state["question"]

    trace.record_input({"question": state["question"], "before": before})
    trace.record_output({"after": after, "changed": after != before})
    return {"retrieval_query": after}


@traced("rerank")
async def rerank(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("generate", attempts="grounding_attempts")
async def generate(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("verify_grounding", attempts="grounding_attempts")
async def verify_grounding(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("abstain")
async def abstain(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Refuse, with the reason the run earned rather than a constant.

    A refusal must be as inspectable as an answer, so the reason is read from
    what the run holds. No candidate survived grading means retrieval found
    nothing to work with. Candidates that did survive but grounded no answer
    means generation is what failed, and the evidence is still there to inspect.
    """
    candidates = state["candidates"]
    reason: RefusalReason = "no_relevant_evidence" if not candidates else "insufficient_evidence"

    trace.record_input(
        {
            "candidates": len(candidates),
            "retrieval_attempts": state["retrieval_attempts"],
            "grounding_attempts": state["grounding_attempts"],
        }
    )
    trace.record_output({"status": "refused", "refusal_reason": reason})
    log.info(
        "abstain",
        query_id=str(state["query_id"]),
        refusal_reason=reason,
        retrieval_attempts=state["retrieval_attempts"],
    )
    return {"status": "refused", "refusal_reason": reason}
