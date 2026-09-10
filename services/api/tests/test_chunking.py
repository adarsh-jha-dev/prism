from itertools import pairwise

import pytest

from prism.ingestion.chunking import chunk_text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", 10, 2) == []


def test_text_shorter_than_size_is_one_chunk() -> None:
    assert chunk_text("abc", 10, 2) == ["abc"]


def test_text_exactly_size_is_one_chunk() -> None:
    assert chunk_text("abcdefghij", 10, 2) == ["abcdefghij"]


def test_one_character_past_size_starts_a_second_chunk() -> None:
    assert chunk_text("abcdefghijk", 10, 2) == ["abcdefghij", "ijk"]


def test_exact_multiple_of_stride_has_no_empty_tail() -> None:
    # Two windows of 4 stepping by 3 end exactly on the last character.
    assert chunk_text("abcdefg", 4, 1) == ["abcd", "defg"]


def test_windows_overlap_by_the_requested_amount() -> None:
    chunks = chunk_text("abcdefghijklmno", 6, 2)
    assert chunks == ["abcdef", "efghij", "ijklmn", "mno"]
    for previous, current in pairwise(chunks):
        assert previous[-2:] == current[:2]


def test_zero_overlap_partitions_the_text() -> None:
    chunks = chunk_text("abcdefgh", 3, 0)
    assert chunks == ["abc", "def", "gh"]
    assert "".join(chunks) == "abcdefgh"


def test_size_of_one() -> None:
    assert chunk_text("abc", 1, 0) == ["a", "b", "c"]


def test_chunks_are_never_empty_and_cover_the_whole_text() -> None:
    text = "x" * 101
    for size in range(1, 12):
        for overlap in range(size):
            chunks = chunk_text(text, size, overlap)
            assert all(chunks)
            assert all(len(chunk) <= size for chunk in chunks)
            step = size - overlap
            rebuilt = chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:])
            assert rebuilt == text, (size, overlap, step)


def test_no_chunk_is_contained_in_its_predecessor() -> None:
    # A short tail must extend the previous window, not repeat part of it.
    chunks = chunk_text("abcdefghi", 5, 4)
    assert chunks == ["abcde", "bcdef", "cdefg", "defgh", "efghi"]


def test_whitespace_is_preserved_verbatim() -> None:
    assert chunk_text("a b\n c", 3, 0) == ["a b", "\n c"]


@pytest.mark.parametrize("size", [0, -1])
def test_non_positive_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        chunk_text("abc", size, 0)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_text("abc", 10, -1)


@pytest.mark.parametrize("overlap", [10, 11])
def test_overlap_at_or_above_size_is_rejected(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than size"):
        chunk_text("abc", 10, overlap)


def test_arguments_are_validated_before_the_text_is_examined() -> None:
    with pytest.raises(ValueError):
        chunk_text("", 10, 10)
