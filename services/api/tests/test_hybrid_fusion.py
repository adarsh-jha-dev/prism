"""Unit: reciprocal rank fusion. No database, no network."""

from uuid import UUID

from prism.retrieval.hybrid import HybridHit, _fuse

A = UUID("00000000-0000-0000-0000-0000000000aa")
B = UUID("00000000-0000-0000-0000-0000000000bb")
C = UUID("00000000-0000-0000-0000-0000000000cc")
D = UUID("00000000-0000-0000-0000-0000000000dd")
FILL = [UUID(f"00000000-0000-0000-0000-{i:012d}") for i in range(1, 7)]


def test_agreement_between_halves_beats_a_single_first_place() -> None:
    """The point of fusion: B is second in both, A only leads one list."""
    assert _fuse([[A, B], [C, B]], k=60) == [B, A, C]


def test_a_hit_in_one_half_only_still_places() -> None:
    assert _fuse([[A], []], k=60) == [A]
    assert _fuse([[], [A]], k=60) == [A]


def test_empty_halves_fuse_to_nothing() -> None:
    assert _fuse([[], []], k=60) == []


def test_order_is_preserved_when_halves_agree() -> None:
    assert _fuse([[A, B, C], [A, B, C]], k=60) == [A, B, C]


def test_ties_break_deterministically() -> None:
    """Distinct chunks at the same position in different halves score equally."""
    first = _fuse([[A], [B]], k=60)
    assert first == _fuse([[A], [B]], k=60)
    assert sorted(first, key=str) == sorted([A, B], key=str)


def test_second_in_both_beats_first_in_one_at_every_k() -> None:
    """2/(k+2) > 1/(k+1) for all k > 0 — agreement is not a tunable preference."""
    for k in (1, 10, 60, 1000):
        assert _fuse([[A, B], [C, B]], k=k)[0] == B, f"k={k}"


def test_k_sets_how_deep_agreement_still_outweighs_a_lone_first_place() -> None:
    """A leads one half; B is only fifth in both. k decides which wins."""
    half_one = [A, *FILL[0:3], B]
    half_two = [C, *FILL[3:6], B]
    assert _fuse([half_one, half_two], k=1)[0] == A, "small k rewards position"
    assert _fuse([half_one, half_two], k=60)[0] == B, "the configured k rewards agreement"


def test_hit_carries_no_fused_score_to_threshold() -> None:
    """ADR 0010: fusion yields an ordering, never a magnitude. Enforced by shape."""
    fields = set(HybridHit.__dataclass_fields__)
    assert not {f for f in fields if "score" in f}, fields
    assert {"rank", "vector_rank", "lexical_rank"} <= fields
