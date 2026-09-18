"""StateGraph assembly.

A straight line: no conditional routing, no loops, no error edges. The canonical
order from CLAUDE.md flattened, so every node runs exactly once.
"""

from itertools import pairwise

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from prism.graph import nodes
from prism.graph.state import GraphState

__all__ = ["NODE_SEQUENCE", "build_graph", "compile_graph"]

NODE_SEQUENCE = (
    "plan_query",
    "embed_query",
    "retrieve",
    "grade_docs",
    "rewrite_query",
    "rerank",
    "generate",
    "verify_grounding",
    "abstain",
)


def build_graph() -> StateGraph[GraphState, None, GraphState, GraphState]:
    builder: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    for name in NODE_SEQUENCE:
        builder.add_node(name, getattr(nodes, name))

    builder.add_edge(START, NODE_SEQUENCE[0])
    for source, target in pairwise(NODE_SEQUENCE):
        builder.add_edge(source, target)
    builder.add_edge(NODE_SEQUENCE[-1], END)
    return builder


def compile_graph(checkpointer: BaseCheckpointSaver[str]) -> CompiledStateGraph[GraphState]:
    """Compiled per run: the checkpointer is bound at compile time."""
    return build_graph().compile(checkpointer=checkpointer)
