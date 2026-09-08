"""UUIDv7 generation (RFC 9562).

Generated application-side: Postgres 16 has no native uuidv7(). Moving to
Postgres 18 can make this a column default with no data migration.
"""

import os
import time
from uuid import UUID

__all__ = ["uuid7"]


def uuid7() -> UUID:
    """unix_ts_ms (48) | ver (4) | rand_a (12) | var (2) | rand_b (62)"""
    timestamp_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")

    rand_a = (rand >> 62) & 0x0FFF
    rand_b = rand & 0x3FFF_FFFF_FFFF_FFFF

    value = (timestamp_ms & 0xFFFF_FFFF_FFFF) << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return UUID(int=value)
