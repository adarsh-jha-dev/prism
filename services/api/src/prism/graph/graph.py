"""StateGraph assembly.

Two conditional edges, one per correction loop. `grade_docs` either passes to
`generate` or sends the run back around retrieval; `verify_grounding` either
finalizes, regenerates, or refuses. The second is the only path to `answered`.

`rerank` sits between `retrieve` and `grade_docs` (ADR 0020), inside the loop.
It needs no conditional edge of its own: a set its floor empties reaches
`grade_docs`, which finds nothing to grade, closes the attempt and fails the
pass — so the floor and the grader share one counter and one refusal reason.

The loop re-enters at `plan_query` so term extraction stays in one prompt
(ADR 0019). `plan_query` does not re-pin `search_params` on a later pass.

Edges read their policy from state, never from `Settings`: an edge decides
without writing a trace row.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from prism.graph import nodes
from prism.graph.state import GraphState

__all__ = [
    "NODES",
    "after_grade_docs",
    "after_verify_grounding",
    "build_graph",
    "compile_graph",
]

# Canonical names: they are trace rows and the dashboard's waterfall.
NODES = (
    "plan_query",
    "embed_query",
    "retrieve",
    "rerank",
    "grade_docs",
    "rewrite_query",
    "generate",
    "verify_grounding",
    "abstain",
)


def after_grade_docs(state: GraphState) -> str:
    """Pass, retry, or refuse.

    `grade_docs` keeps the candidates that passed and drops the rest, so a set
    that survives grading is the pass signal and an empty one is the fail —
    whether it was the grader or `rerank`'s floor that emptied it.

    `max_attempts` is the run's pinned value, so a fork has as many passes left
    as the original run did (ADR 0017).
    """
    if state["candidates"]:
        return "generate"
    if state["retrieval_attempts"] < state["search_params"]["max_attempts"]:
        return "rewrite_query"
    # Exhaustion refuses directly. Nothing goes between here and `abstain`: the
    # design's web-search fallback is out, because generation nodes make no
    # network egress and a web result has no `query_citations.chunk_id`
    # (CLAUDE.md).
    return "abstain"


def after_verify_grounding(state: GraphState) -> str:
    """Finalize, regenerate, or refuse.

    `verify_grounding` writes `status` and this reads it, as `after_grade_docs`
    reads the candidates the grader kept: the node renders the verdict, the edge
    only routes it.

    `grounding_attempts` is this loop's own counter. A run that spent its whole
    retrieval budget still arrives here with every generation unspent, and
    neither counter is readable from the other (ADR 0022).

    The pass goes to END rather than to a `finalize` node. `CLAUDE.md` has three
    terminals and no such node, and the transactional write needs the latency
    measured around the run — so `run.py` owns it, as it already did for refusals.
    """
    if state["status"] == "answered":
        return END
    if state["grounding_attempts"] < state["search_params"]["max_attempts"]:
        return "generate"
    return "abstain"


def build_graph() -> StateGraph[GraphState, None, GraphState, GraphState]:
    builder: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    for name in NODES:
        builder.add_node(name, getattr(nodes, name))

    builder.add_edge(START, "plan_query")
    builder.add_edge("plan_query", "embed_query")
    builder.add_edge("embed_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade_docs")

    builder.add_conditional_edges(
        "grade_docs",
        after_grade_docs,
        ["generate", "rewrite_query", "abstain"],
    )
    builder.add_edge("rewrite_query", "plan_query")

    builder.add_edge("generate", "verify_grounding")
    builder.add_conditional_edges(
        "verify_grounding",
        after_verify_grounding,
        ["generate", "abstain", END],
    )

    builder.add_edge("abstain", END)
    return builder


def compile_graph(checkpointer: BaseCheckpointSaver[str]) -> CompiledStateGraph[GraphState]:
    """Compiled per run: the checkpointer is bound at compile time."""
    return build_graph().compile(checkpointer=checkpointer)
