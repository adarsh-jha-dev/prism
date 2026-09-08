import time
from uuid import UUID

from prism.core.ids import uuid7


def test_version_and_variant_bits() -> None:
    value = uuid7()
    assert value.version == 7
    assert (value.int >> 62) & 0b11 == 0b10


def test_timestamp_matches_wall_clock() -> None:
    before = time.time_ns() // 1_000_000
    value = uuid7()
    after = time.time_ns() // 1_000_000
    embedded = value.int >> 80
    assert before <= embedded <= after


def test_ids_sort_in_creation_order() -> None:
    ids = [uuid7() for _ in range(1000)]
    assert ids == sorted(ids, key=lambda u: u.int >> 80)


def test_ids_are_unique() -> None:
    ids = {uuid7() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_roundtrips_through_string_form() -> None:
    value = uuid7()
    assert UUID(str(value)) == value
