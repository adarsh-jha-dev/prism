"""The graph's state. Every value here is serialized into every checkpoint."""

from typing import Annotated, Literal, TypedDict
from uuid import UUID

__all__ = ["GraphState", "RefusalReason", "TerminalStatus", "monotonic"]

TerminalStatus = Literal["cached", "answered", "refused"]
RefusalReason = Literal["no_relevant_evidence", "insufficient_evidence"]


def monotonic(current: int, incoming: int) -> int:
    """Reducer for counters that only go up. Without one, concurrent writes error."""
    return max(current, incoming)


class GraphState(TypedDict):
    query_id: UUID
    tenant_id: UUID
    collection_id: UUID
    thread_id: str
    question: str

    # The two correction loops count independently.
    retrieval_attempts: Annotated[int, monotonic]
    grounding_attempts: Annotated[int, monotonic]

    # The last sequence written; the trace writer takes the next from here.
    sequence: Annotated[int, monotonic]

    status: TerminalStatus
    refusal_reason: RefusalReason | None
