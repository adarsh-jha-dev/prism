"""The graph's nodes.

Both correction loops are implemented. The retrieval loop is `plan_query`,
`embed_query`, `retrieve`, `rerank`, `grade_docs` and `rewrite_query`; the
grounding loop is `generate` and `verify_grounding`, the gate that opens the
only path to `answered` (ADR 0022). `abstain` ends the rest.

`rerank` sits between `retrieve` and `grade_docs` (ADR 0020): it scores the
fused pool and cuts it to `k`, so the grader judges only what the floor kept.

Retrieval reads `retrieval_query`, which the loop rewrites. Grading and
generation read `question`, which nothing rewrites (ADR 0019).

Nodes report through the TraceContext (ADR 0016), as references rather than
copies (ADR 0012), and call models through the registry (ADR 0018).
"""

import math
import re
import time
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, create_model

from prism.chat import Message
from prism.chat.base import Usage
from prism.collections import abstention_threshold_for
from prism.config import get_settings
from prism.graph.state import (
    CandidateRef,
    CitationRef,
    GraphState,
    RefusalReason,
    search_params_from,
)
from prism.graph.trace import TraceContext, traced
from prism.providers import get_registry
from prism.rerank import RerankError, get_reranker
from prism.retrieval.hybrid import HybridHit, hybrid_search
from prism.retrieval.hydrate import HydratedChunk, hydrate_chunks
from prism.retrieval.rerank import rerank as rerank_hits

__all__ = [
    "ChunkRelevance",
    "Citation",
    "GroundedAnswer",
    "GroundingVerdicts",
    "QueryPlan",
    "QueryRewrite",
    "RelevanceVerdicts",
    "SpanGroundedness",
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

# `generate` calls the local lane and only the local lane. Which provider answers
# is the Phase 3 router's decision, taken on cost and latency; nothing here
# chooses one.
_LOCAL_LANE = "ollama"
# Not a registry lane: the reranker runs in this process and reports no Usage of
# its own. The name is what `model_pricing` prices it under (ADR 0013).
_IN_PROCESS_LANE = "in-process"

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

_GENERATOR_SYSTEM = (
    "You answer strictly from the numbered passages. Mark every claim with the "
    "label of the passage that supports it, written as [1], and list those same "
    "labels in citations. Never cite a label that is not among the passages, "
    "never use outside knowledge to fill a gap, and never offer a plausible "
    "guess in place of evidence. If the passages do not support an answer, say "
    "so plainly and cite nothing: that is a correct answer, not a failure."
)

# Prose, so wider than the graders. Still a ceiling: an answer running past this
# is not summarizing the passages it was given.
_GENERATOR_MAX_TOKENS = 800

_REGENERATION_PREFACE = (
    "Your previous answer was rejected: the claims below were not supported by "
    "the passages. Answer again from the passages alone. Drop every claim they "
    "do not state, and if what remains does not answer the question, say so "
    "plainly and cite nothing."
)

_VERIFIER_SYSTEM = (
    "You judge whether each numbered claim is supported by the numbered "
    "passages. Claims are numbered (1), (2); passages are numbered [1], [2]. "
    "Score every claim from 0 to 1 by how fully the passages state it, and "
    "return its number unchanged. A claim the passages do not state scores low, "
    "even when it is plausible and even when it is true. Judge only what the "
    "passages say: never use outside knowledge, and never reward a claim for "
    "being well written. Return one entry per claim."
)

# Wider than the grader's: an answer has more claims than a pool has passages.
_VERIFIER_MAX_TOKENS = 600

# A sentence ends at .!? followed by whitespace and the start of the next one —
# a capital, a quote, a bracket. Never before a digit: "0.58" and "approx. 20"
# are one claim, and a system whose answers are mostly numbers cannot afford a
# splitter that reads a decimal point as a claim boundary (ADR 0022).
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\[(])')

# Below this, a span is a fragment rather than a claim. Merged into its
# neighbour instead of judged alone, so a bad split cannot refuse a sound answer.
_MIN_SPAN_CHARS = 24

# What counts as an inline citation marker. Only a bracketed number **shown in
# this prompt** is one: with five passages, an answer that legitimately contains
# "[20]" keeps it as prose rather than having it read as a citation (ADR 0021).
_MARKER = re.compile(r"\[(\d+)\]")


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


def _relevance_schema(passages: int) -> type[RelevanceVerdicts]:
    """`RelevanceVerdicts` fitted to one call: exactly one verdict per passage sent.

    `min_length=1` alone let `llama3.1:8b` score passage [1] and stop, failing
    every other passage for want of a verdict, and set no upper bound, so a
    grader could emit entries until `max_tokens` cut it mid-JSON (ADR 0025).
    Same name, so logs and stubs still key on `RelevanceVerdicts`.
    """
    verdict = create_model(
        "ChunkRelevance",
        __base__=ChunkRelevance,
        label=(int, Field(ge=1, le=passages)),
    )
    return create_model(
        "RelevanceVerdicts",
        __base__=RelevanceVerdicts,
        verdicts=(list[verdict], Field(min_length=passages, max_length=passages)),  # type: ignore[valid-type]
    )


class SpanGroundedness(BaseModel):
    """One claim's score, keyed by the label the prompt numbered it with."""

    label: int
    score: float


class GroundingVerdicts(BaseModel):
    # min_length for `RelevanceVerdicts`' reason: without it the cheapest
    # completion that validates is `{"verdicts": []}`, and a verifier always has
    # claims to score. Every span would then fail for want of a verdict, which
    # refuses in the right direction for entirely the wrong reason.
    verdicts: list[SpanGroundedness] = Field(min_length=1)


class QueryRewrite(BaseModel):
    query: str = ""


class Citation(BaseModel):
    """One source the answer draws on, keyed by the label the prompt showed it under.

    An object rather than a bare int so that citation character offsets — a known
    gap in `REVIEW.md` — arrive as fields here rather than as a second list that
    can disagree with this one.
    """

    label: int


class GroundedAnswer(BaseModel):
    # No `min_length` on `citations`, unlike `RelevanceVerdicts`. A grader always
    # has passages to score, so an empty list there is the model taking the
    # cheapest completion that validates. Generation is different: a model
    # correctly reporting that the passages do not cover the question has nothing
    # to cite, and a schema that forced it would manufacture the binding this node
    # exists to establish (ADR 0021). An answer that cites nothing is an outcome,
    # handled below.
    answer: str
    citations: list[Citation] = Field(default_factory=list)


@traced("plan_query", attempts="retrieval_attempts")
async def plan_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Extract the lexical half's search terms, and pin the parameters once.

    `websearch_to_tsquery` ANDs its terms, so fewer and rarer terms retrieve more
    (ADR 0010). The rewrite loop re-enters here so that rule lives in one prompt
    rather than two that can diverge (ADR 0019).

    Parameters are pinned so a fork retrieves and corrects as the original run
    did (ADR 0017), which means later passes must not re-pin: a second execution
    would re-read `Settings`. tau is pinned with them, from the collection — the
    one policy constant a tenant sets per collection (ADR 0022).
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
    params = state["search_params"]
    if pinned:
        # One keyed read under the tenant predicate, in a node that has just
        # made a 14b call. Unreadable falls back to `Settings` rather than
        # failing the run: `queries` holds a composite FK to `collections`, so
        # this is a defensive default, not a supported configuration.
        tau = await abstention_threshold_for(state["collection_id"], tenant_id=state["tenant_id"])
        if tau is None:
            log.warning(
                "plan_query.collection_tau_unreadable",
                query_id=str(state["query_id"]),
                collection_id=str(state["collection_id"]),
            )
        params = search_params_from(settings, abstention_threshold=tau)
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
    # The pool `rerank` scores, not the final k — it cuts to k (ADR 0020). Same
    # width the eval harness fuses at, so a pinned baseline measures this node.
    pool = max(params["k"], params["rerank_candidate_k"])

    hits = await hybrid_search(
        state["retrieval_query"],
        tenant_id=state["tenant_id"],
        collection_id=state["collection_id"],
        terms=terms or None,
        k=pool,
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
            # Fusion yields no magnitude (ADR 0010); `rerank` fills this in.
            rerank_score=None,
        )
        for hit in hits
    ]

    # The widths this search ran at, not the whole pinned set.
    trace.record_input(
        {
            "retrieval_query": state["retrieval_query"],
            "terms": terms,
            "pool": pool,
            "candidate_k": params["candidate_k"],
            "rrf_k": params["rrf_k"],
        }
    )
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

    The threshold is the run's pinned value, not the live one (ADR 0017).

    Closes the attempt by incrementing `retrieval_attempts`. The surviving
    candidates are the graph's pass/fail signal: nothing the grader rejected
    reaches `rerank`, `generate` or a citation.

    Writes `verdict` — on its no-call paths too. An empty candidate set and a set
    that hydrates to nothing are both judgements, and a NULL there would read as
    a node with none to make (ADR 0022).
    """
    settings = get_settings()
    threshold = state["search_params"]["doc_relevance_threshold"]
    candidates = state["candidates"]
    attempt = state["retrieval_attempts"] + 1

    if not candidates:
        # Nothing to grade, and a grader asked to judge nothing invents a
        # verdict. No call, no meter, and a fail that costs nothing.
        trace.record_verdict("fail")
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
        # (ADR 0017). A fail, and still no call — so the set has to be cleared
        # here, or an unreadable candidate would read downstream as a pass.
        trace.record_verdict("fail")
        trace.record_input({"candidates": [str(c["chunk_id"]) for c in candidates]})
        trace.record_output({"verdicts": [], "kept": 0, "reason": "no_hydrated_chunks"})
        return {"retrieval_attempts": attempt, "candidates": []}

    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_GRADER_SYSTEM),
            Message(role="user", content=_grading_prompt(state["question"], chunks)),
        ],
        _relevance_schema(len(chunks)),
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
            "verdict": "pass" if (scores.get(chunk.chunk_id) or 0.0) >= threshold else "fail",
        }
        for chunk in chunks
    ]
    kept_ids = {v["chunk_id"] for v in verdicts if v["verdict"] == "pass"}
    kept = [candidate for candidate in candidates if candidate["chunk_id"] in kept_ids]

    trace.record_verdict("pass" if kept else "fail")
    trace.record_input({"candidates": [str(c["chunk_id"]) for c in candidates]})
    trace.record_output(
        {
            "verdicts": [{**v, "chunk_id": str(v["chunk_id"])} for v in verdicts],
            "kept": len(kept),
            "threshold": threshold,
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


def _numbered_passages(chunks: list[HydratedChunk]) -> str:
    """The passages, numbered from 1. The label is the handle everything downstream uses.

    One function for the grader and the generator, so the shape a verdict comes
    back keyed by is the shape a citation comes back keyed by (ADR 0021). Two
    prompts numbering passages their own way would drift, and the drift would
    show up as citations pointing at the wrong chunk.
    """
    return "\n\n".join(f"[{label}] {chunk.content}" for label, chunk in enumerate(chunks, start=1))


def _grading_prompt(question: str, chunks: list[HydratedChunk]) -> str:
    return f"Question:\n{question}\n\nPassages:\n{_numbered_passages(chunks)}"


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


@traced("rerank", attempts="retrieval_attempts")
async def rerank(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Score the fused pool with the cross-encoder, cut to k, apply the floor.

    Runs before `grade_docs` (ADR 0020), so the grader judges only what the
    floor kept. Calls the same `retrieval.rerank.rerank` the eval harness calls,
    with the run's pinned floor and pool, so a pinned baseline keeps measuring
    this node rather than something shaped like it.

    Scored against `question`, never `retrieval_query`: reranking is a relevance
    judgement, and a judging node reads what the user asked (ADR 0019).

    A set the floor empties re-enters the retrieval loop (ADR 0011) — through
    `grade_docs`, which finds nothing to grade and closes the attempt. This node
    never increments the counter: one pass of the loop is one attempt, whichever
    gate fails it.
    """
    settings = get_settings()
    params = state["search_params"]
    floor = params["rerank_score_floor"]
    candidates = state["candidates"]
    ids = [candidate["chunk_id"] for candidate in candidates]
    trace.record_input({"candidates": [str(chunk_id) for chunk_id in ids], "floor": floor})

    if not candidates:
        trace.record_output({"ranked": [], "kept": 0, "reason": "no_candidates"})
        return {}

    chunks = await hydrate_chunks(
        ids, tenant_id=state["tenant_id"], collection_id=state["collection_id"]
    )
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    hits = [
        _hybrid_hit(candidate, by_id[candidate["chunk_id"]])
        for candidate in candidates
        if candidate["chunk_id"] in by_id
    ]
    if not hits:
        # Retrieved ids that hydrate to nothing (ADR 0017). Nothing to score,
        # and an empty set is what sends the pass back round the loop.
        trace.record_output({"ranked": [], "kept": 0, "reason": "no_hydrated_chunks"})
        return {"candidates": []}

    reranker = get_reranker()
    clock = time.perf_counter()
    try:
        # Idempotent and lock-guarded, and a no-op once the process has loaded.
        # The node cannot assume someone else loaded the weights: `score` raises
        # rather than loading lazily, so a node that skipped this would fall back
        # to fusion order on every query and say so only on its own trace row.
        await reranker.load()
        reranked = await rerank_hits(
            state["question"],
            hits,
            k=params["k"],
            reranker=reranker,
            settings=settings.model_copy(
                update={
                    "rerank_candidate_k": params["rerank_candidate_k"],
                    "rerank_score_floor": floor,
                }
            ),
        )
    except RerankError as exc:
        return _rerank_fallback(state, trace, hits, params["k"], exc)

    # No provider and no meter, but a model ran and ADR 0013 prices it — at zero,
    # from a real `model_pricing` row, so the node reads as free rather than
    # unpriced. `metered` because there is nothing to estimate: the unit is none.
    trace.record_usage(
        Usage(
            model=reranker.model,
            provider=_IN_PROCESS_LANE,
            billing_unit="none",
            input_tokens=None,
            output_tokens=None,
            gpu_ms=None,
            duration_ms=int((time.perf_counter() - clock) * 1000),
            cost_basis="metered",
        )
    )

    # Reranked order, with the fusion positions the candidate came in with: the
    # trace row and the dashboard show both, and neither is derivable from the other.
    fused = {candidate["chunk_id"]: candidate for candidate in candidates}
    kept = [
        CandidateRef(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            rank=position,
            vector_rank=fused[hit.chunk_id]["vector_rank"],
            lexical_rank=fused[hit.chunk_id]["lexical_rank"],
            rerank_score=hit.score,
        )
        for position, hit in enumerate(reranked.hits, start=1)
    ]

    # The whole ranking before the floor, marked: the viewer shows what the floor
    # discarded, not only its effect.
    trace.record_output(
        {
            "floor": floor,
            "fallback": False,
            "kept": len(kept),
            "ranked": [
                {
                    "chunk_id": str(hit.chunk_id),
                    "rank": hit.rank,
                    "fused_rank": hit.fused_rank,
                    "score": hit.score,
                    "kept": hit.score >= floor,
                }
                for hit in reranked.ranked
            ],
        }
    )
    if not kept:
        log.info(
            "rerank.floor_emptied",
            query_id=str(state["query_id"]),
            attempt=state["retrieval_attempts"] + 1,
            scored=len(reranked.ranked),
            floor=floor,
        )
    return {"candidates": kept}


def _hybrid_hit(candidate: CandidateRef, chunk: HydratedChunk) -> HybridHit:
    """What the reranker takes: a candidate reference plus its hydrated text."""
    return HybridHit(
        chunk_id=candidate["chunk_id"],
        document_id=candidate["document_id"],
        filename=chunk.filename,
        content=chunk.content,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        rank=candidate["rank"],
        vector_rank=candidate["vector_rank"],
        lexical_rank=candidate["lexical_rank"],
    )


def _rerank_fallback(
    state: GraphState,
    trace: TraceContext,
    hits: list[HybridHit],
    k: int,
    exc: RerankError,
) -> dict[str, Any]:
    """Keep fusion order with the floor unapplied, and mark the row degraded.

    The node's decision, not an error (ADR 0011): the query proceeds, because
    rerank buys precision and cost rather than groundedness, and tau still
    decides at `verify_grounding`. What it costs is the measurement — a run that
    skipped rerank is not measuring the optimized path, so `fallback` on this row
    is what excludes the whole query from every eval and benchmark aggregate.

    No usage: no model ran.
    """
    kept = [
        CandidateRef(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            rank=position,
            vector_rank=hit.vector_rank,
            lexical_rank=hit.lexical_rank,
            # Unscored, not zero: nothing scored it.
            rerank_score=None,
        )
        for position, hit in enumerate(hits[:k], start=1)
    ]
    trace.record_output(
        {
            "fallback": True,
            "reason": "rerank_error",
            "error": f"{type(exc).__name__}: {exc}",
            "kept": len(kept),
            "ranked": [
                {
                    "chunk_id": str(candidate["chunk_id"]),
                    "rank": candidate["rank"],
                    "fused_rank": candidate["rank"],
                    "score": None,
                    "kept": True,
                }
                for candidate in kept
            ],
        }
    )
    log.warning(
        "rerank.fallback",
        query_id=str(state["query_id"]),
        attempt=state["retrieval_attempts"] + 1,
        candidates=len(kept),
        error=str(exc),
    )
    return {"candidates": kept}


@traced("generate", attempts="grounding_attempts")
async def generate(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Answer from the surviving evidence, and bind every claim to the chunk behind it.

    Answers `question`, never `retrieval_query`: generation is judged against
    what the user submitted, and a rewriter that drifted must not get to define
    the question it is answered on (ADR 0019).

    One structured call, on the local lane. The citation list has to validate, and
    a list parsed out of prose afterwards is post-hoc matching — which would put
    cosine similarity back in the evidence path, where `CLAUDE.md` forbids it.

    On a retry it is told which claims the gate found unsupported, so a
    regeneration is constrained by its own failure rather than being the same
    call with a different seed (ADR 0022). Which provider answers is still not
    decided here: escalation is the Phase 3 router's.

    Nothing here finalizes anything. `verdict` stays NULL because this node has an
    outcome and no judgement (migration 0009), `grounding_attempts` stays where it
    was because the gate increments it, and the answer stays in state until
    `verify_grounding` passes it (ADR 0021).
    """
    settings = get_settings()
    # Rerank order, explicitly, rather than inherited from whatever order the
    # upstream nodes happened to leave: `rank` is the position `rerank` assigned
    # by score, and the labels the model sees are that order.
    candidates = sorted(state["candidates"], key=lambda candidate: candidate["rank"])
    unsupported = state["unsupported_spans"]
    trace.record_input(
        {
            "candidates": [
                {"chunk_id": str(c["chunk_id"]), "rerank_score": c["rerank_score"]}
                for c in candidates
            ],
            # The count, not the text: the claims themselves are on the previous
            # attempt's `verify_grounding` row, which eval already addresses by
            # (query_id, node_name, attempt).
            "unsupported_spans": len(unsupported),
        }
    )

    if not candidates:
        # Only reachable on a fork resumed here: `after_grade_docs` routes an
        # empty set back round the loop. No call — a model asked to answer from
        # nothing answers from itself.
        return _ungrounded(trace, reason="no_candidates")

    chunks = await hydrate_chunks(
        [candidate["chunk_id"] for candidate in candidates],
        tenant_id=state["tenant_id"],
        collection_id=state["collection_id"],
    )
    if not chunks:
        # Deleted or re-ingested since retrieval (ADR 0017). Nothing to show the
        # model, and nothing that could be cited if it answered anyway.
        return _ungrounded(trace, reason="no_hydrated_chunks")

    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_GENERATOR_SYSTEM),
            Message(role="user", content=_generation_prompt(state, chunks, unsupported)),
        ],
        GroundedAnswer,
        model=settings.generator_model,
        max_tokens=_GENERATOR_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    answer = result.value.answer.strip()
    citations, dropped = _bind_citations(result.value, chunks, candidates)

    if not answer:
        # The call succeeded and the schema validated, so this is an outcome and
        # the row stays `ok` (ADR 0021). `error` would claim a failure that did
        # not happen, and migration 0009 would want error text we do not have.
        return _ungrounded(trace, reason="empty_answer", dropped=dropped)
    if not citations:
        # Either the model cited nothing, or everything it cited was fabricated.
        # Either way there is no binding, so there is no answer to carry: the run
        # refuses rather than passing prose to a verifier as if it had evidence.
        return _ungrounded(trace, reason="no_valid_citations", dropped=dropped)

    # Lengths and labels, never the answer text: that belongs on
    # `queries.final_answer`, written once at finalization (ADR 0012).
    trace.record_output(
        {
            "answer_length": len(answer),
            "passages": len(chunks),
            "cited_labels": [citation["label"] for citation in citations],
            "cited_chunk_ids": [str(citation["chunk_id"]) for citation in citations],
            "dropped_labels": dropped,
        }
    )
    if dropped:
        log.warning(
            "generate.fabricated_citation",
            query_id=str(state["query_id"]),
            attempt=state["grounding_attempts"] + 1,
            labels=dropped,
            passages=len(chunks),
        )
    return {"answer": answer, "citations": citations}


def _generation_prompt(
    state: GraphState, chunks: list[HydratedChunk], unsupported: list[str]
) -> str:
    """The question and the passages, plus what the last attempt got wrong.

    The rejected claims go last, where a local model attends to them, and they
    are the gate's own finding rather than a generic "be stricter" (ADR 0022).
    """
    prompt = f"Question:\n{state['question']}\n\nPassages:\n{_numbered_passages(chunks)}"
    if not unsupported:
        return prompt
    rejected = "\n".join(f"- {span}" for span in unsupported)
    return f"{prompt}\n\n{_REGENERATION_PREFACE}\n{rejected}"


def _ungrounded(
    trace: TraceContext, *, reason: str, dropped: list[int] | None = None
) -> dict[str, Any]:
    """A generation that bound nothing: no answer leaves this node.

    Clears both channels rather than returning nothing. Once the grounding loop
    exists, a later attempt that grounds nothing must not leave the previous
    attempt's answer standing for finalization to find (ADR 0021).
    """
    trace.record_output(
        {
            "reason": reason,
            "answer_length": 0,
            "cited_labels": [],
            "dropped_labels": dropped or [],
        }
    )
    return {"answer": None, "citations": []}


def _bind_citations(
    produced: GroundedAnswer,
    chunks: list[HydratedChunk],
    candidates: list[CandidateRef],
) -> tuple[list[CitationRef], list[int]]:
    """Reconcile the answer's markers and its citation list against what was shown.

    A label the passage set never held is a fabricated chunk id — one of the
    injection categories this project tests for — so it is dropped here and never
    reaches a row. The prose is left exactly as the model wrote it: a bracket that
    resolves to nothing renders as prose and the dropped label is named on the
    trace row, which is a better record than an audit trail that edits its subject.

    The two directions are deliberately asymmetric (ADR 0021). A marker on a real
    passage that the list omits is admitted, because the passage was shown and
    admitting it fabricates nothing. A listed label with no marker is kept, which
    is what saves an answer whose prose came back clean but unmarked.

    Returns the citations in label order — which is rerank order, since that is
    how the passages were numbered — and the labels that were dropped.
    """
    shown = dict(enumerate(chunks, start=1))
    by_id = {candidate["chunk_id"]: candidate for candidate in candidates}

    marked = {int(label) for label in _MARKER.findall(produced.answer)} & shown.keys()
    listed = {citation.label for citation in produced.citations}

    dropped = sorted(listed - shown.keys())
    cited = sorted(marked | (listed & shown.keys()))

    return [
        CitationRef(
            label=label,
            chunk_id=shown[label].chunk_id,
            document_id=shown[label].document_id,
            page_number=shown[label].page_number,
            chunk_index=shown[label].chunk_index,
            # Dense from 1 in citation order, not the candidate's rank: it becomes
            # `query_citations.rank`, which is unique per query and checked >= 1.
            rank=rank,
            rerank_score=by_id[shown[label].chunk_id]["rerank_score"],
            # The text as the model was shown it. Snapshotted here rather than
            # re-read at finalization, which would record whatever the chunk says
            # by then (ADR 0021).
            content=shown[label].content,
        )
        for rank, label in enumerate(cited, start=1)
    ], dropped


@traced("verify_grounding", attempts="grounding_attempts")
async def verify_grounding(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    """Judge the answer against the evidence it cited. The only node that reads tau.

    tau is compared against one quantity and one only: the groundedness score
    this call produces. Never a cosine similarity, never a rerank score — they
    share a 0-1 scale and nothing else (CLAUDE.md, ADR 0022). The value is the
    run's pinned one, so a fork verifies at the bar the original run used.

    Judged against the citations, not every candidate that survived grading. An
    answer is grounded in what it cited, and the cited text is already in state
    as the model was shown it (ADR 0021) — so there is no read here to drift
    under the judgement.

    Closes the attempt by incrementing `grounding_attempts`, as `grade_docs`
    does for the retrieval loop. The two counters are independent and neither
    borrows the other's budget.

    On a pass it writes `status`, which is what the edge routes on: the node that
    judged is the node that recorded it. On a fail it writes the claims that
    failed, for the next `generate` to answer under.
    """
    settings = get_settings()
    tau = state["search_params"]["abstention_threshold"]
    attempt = state["grounding_attempts"] + 1
    answer, citations = state["answer"], state["citations"]

    trace.record_input(
        {
            "tau": tau,
            "cited_labels": [citation["label"] for citation in citations],
            "cited_chunk_ids": [str(citation["chunk_id"]) for citation in citations],
        }
    )

    if not answer or not citations:
        # `generate` clears both when it binds nothing, and an answer that cited
        # nothing has no evidence to be judged against. A fail, and no call: a
        # verifier handed no passages invents a verdict.
        reason = "no_answer" if not answer else "no_citations"
        return _unverified(trace, attempt, tau, reason=reason)

    spans = _spans(answer)
    if not spans:
        return _unverified(trace, attempt, tau, reason="no_spans")

    result = await get_registry().structured(
        _LOCAL_LANE,
        [
            Message(role="system", content=_VERIFIER_SYSTEM),
            Message(
                role="user",
                content=_verification_prompt(state["question"], citations, spans),
            ),
        ],
        GroundingVerdicts,
        model=settings.grader_model,
        max_tokens=_VERIFIER_MAX_TOKENS,
    )
    trace.record_usage(result.usage)

    scores = _scores_by_span(result.value, spans)
    labels = range(1, len(spans) + 1)
    # The minimum, not the mean: `answered` claims every span is grounded, and a
    # mean lets one fabricated sentence hide behind four sound ones — which is
    # the exact signature of the injections this project counts (ADR 0022).
    groundedness = min(scores.get(label) or 0.0 for label in labels)
    unsupported = [
        span for label, span in enumerate(spans, start=1) if (scores.get(label) or 0.0) < tau
    ]
    grounded = not unsupported

    trace.record_verdict("pass" if grounded else "fail")
    trace.record_output(
        {
            # The span text, which is the one place answer text reaches a trace
            # payload. A rejected attempt never reaches `queries.final_answer`,
            # so this row is the only record of what the run refused to say.
            "spans": [
                {
                    "label": label,
                    "span": span,
                    "score": scores.get(label),
                    # Derived from the one threshold, so the verifier cannot
                    # both score 0.9 and call a claim unsupported.
                    "verdict": "pass" if (scores.get(label) or 0.0) >= tau else "fail",
                }
                for label, span in enumerate(spans, start=1)
            ],
            "groundedness": groundedness,
            "tau": tau,
            "unsupported": len(unsupported),
        }
    )

    if not grounded:
        log.info(
            "verify_grounding.ungrounded",
            query_id=str(state["query_id"]),
            attempt=attempt,
            groundedness=groundedness,
            tau=tau,
            unsupported=len(unsupported),
        )
        return {"grounding_attempts": attempt, "unsupported_spans": unsupported}

    return {
        "grounding_attempts": attempt,
        "unsupported_spans": [],
        "status": "answered",
        "refusal_reason": None,
    }


def _unverified(trace: TraceContext, attempt: int, tau: float, *, reason: str) -> dict[str, Any]:
    """A verification with nothing to verify: a fail, and no model call.

    Clears the feedback rather than carrying the last attempt's. There is no
    answer this attempt, so there are no claims of it to have failed, and the
    only correction available is to generate again.
    """
    trace.record_verdict("fail")
    trace.record_output(
        {"spans": [], "groundedness": 0.0, "tau": tau, "unsupported": 0, "reason": reason}
    )
    return {"grounding_attempts": attempt, "unsupported_spans": []}


def _spans(answer: str) -> list[str]:
    """Split the answer into claims. Deterministic, and it never drops text.

    Ours rather than the model's, for ADR 0021's reason one node later: a
    model-chosen claim list is a paraphrase that cannot be aligned back to the
    text we hold, so a claim quietly omitted from it is a claim never judged.
    """
    parts = [part.strip() for part in _SENTENCE_END.split(answer.strip()) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < _MIN_SPAN_CHARS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    # A leading fragment has no previous span to join, so it takes the next one.
    if len(merged) > 1 and len(merged[0]) < _MIN_SPAN_CHARS:
        head = merged.pop(0)
        merged[0] = f"{head} {merged[0]}"
    return merged


def _verification_prompt(question: str, citations: list[CitationRef], spans: list[str]) -> str:
    """Claims as (n), evidence as [n] — the labels `generate` showed.

    Two numbering schemes in one prompt, deliberately different: the spans carry
    the answer's own `[n]` markers, so a claim still points at the passage it
    claims, and an 8b model is never asked which bracket means which.
    """
    passages = "\n\n".join(f"[{c['label']}] {c['content']}" for c in citations)
    claims = "\n".join(f"({label}) {span}" for label, span in enumerate(spans, start=1))
    return f"Question:\n{question}\n\nPassages:\n{passages}\n\nClaims:\n{claims}"


def _scores_by_span(verdicts: GroundingVerdicts, spans: list[str]) -> dict[int, float]:
    """Map labels back to spans. Out-of-range labels are dropped.

    Keyed by label, never by position, for `_scores_by_chunk`'s reason. A span
    with no verdict gets none and fails: an absent judgement is not evidence of
    groundedness, and this is the node that must refuse when unsure.
    """
    return {
        verdict.label: verdict.score
        for verdict in verdicts.verdicts
        if 1 <= verdict.label <= len(spans)
    }


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
