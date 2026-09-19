"""Node stubs: one trace row each, state unchanged apart from the sequence.

No retrieval, no model calls, no conditional edges. Each takes the TraceContext
its real version will report through, and reports nothing. abstain refuses,
because a graph with no evidence has nothing else to return.
"""

from typing import Any

from prism.graph.state import GraphState
from prism.graph.trace import TraceContext, traced

__all__ = [
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


@traced("plan_query")
async def plan_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("embed_query")
async def embed_query(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


@traced("retrieve")
async def retrieve(state: GraphState, trace: TraceContext) -> dict[str, Any]:
    return {}


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
