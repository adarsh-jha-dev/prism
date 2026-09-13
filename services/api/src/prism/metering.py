"""Per-key rate limiting and usage metering.

Rationale: docs/decisions/0008-rate-limiting-and-usage-metering.md

The two halves take opposite stances on failure, deliberately. The limiter is a
control: if it cannot count, the request is refused rather than passed through
unmetered. Metering is a record of work already authorized: if the row cannot be
written, the request still succeeds and the loss is logged.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.core.ids import uuid7
from prism.db import get_engine, get_redis

__all__ = [
    "RateLimitExceeded",
    "RateLimitState",
    "RateLimiterUnavailable",
    "consume",
    "record_usage",
    "window_start",
]

log = structlog.get_logger(__name__)

WINDOW_SECONDS = 60
_USAGE_GRAIN_SECONDS = 3600


class RateLimitExceeded(Exception):
    def __init__(self, state: "RateLimitState") -> None:
        super().__init__(f"rate limit of {state.limit}/min exceeded")
        self.state = state


class RateLimiterUnavailable(Exception):
    """Redis could not be reached. The request is refused, not passed through."""


@dataclass(frozen=True)
class RateLimitState:
    limit: int
    remaining: int
    reset_seconds: int

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_seconds),
        }


def window_start(now: datetime | None = None, *, seconds: int = _USAGE_GRAIN_SECONDS) -> datetime:
    """Floor `now` to the start of its metering window."""
    moment = now or datetime.now(UTC)
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)


async def consume(api_key_id: UUID, *, limit: int, now: datetime | None = None) -> RateLimitState:
    """Count one request against `api_key_id`'s minute window.

    Raises RateLimitExceeded past the limit, and RateLimiterUnavailable if Redis
    cannot be reached — an uncounted request is a refusal, never a free one.
    """
    moment = now or datetime.now(UTC)
    epoch = int(moment.timestamp())
    bucket = epoch // WINDOW_SECONDS
    reset_seconds = WINDOW_SECONDS - (epoch % WINDOW_SECONDS)
    redis = get_redis()
    key = f"ratelimit:{api_key_id}:{bucket}"

    try:
        # INCR then EXPIRE in one round trip; the key is new only on the first
        # request of a window, so the TTL is set once and never extended.
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, WINDOW_SECONDS)
            used: int = (await pipe.execute())[0]
    except RedisError as exc:
        raise RateLimiterUnavailable(str(exc)) from exc

    state = RateLimitState(limit=limit, remaining=max(0, limit - used), reset_seconds=reset_seconds)
    if used > limit:
        raise RateLimitExceeded(state)
    return state


_UPSERT_USAGE = text(
    """
    INSERT INTO usage_records (id, api_key_id, tenant_id, window_start, requests, throttled)
    VALUES (:id, :api_key_id, :tenant_id, :window_start, :requests, :throttled)
    ON CONFLICT (api_key_id, window_start) DO UPDATE
    SET requests   = usage_records.requests  + EXCLUDED.requests,
        throttled  = usage_records.throttled + EXCLUDED.throttled,
        updated_at = now()
    """
)
_TOUCH_KEY = text("UPDATE api_keys SET last_used_at = now() WHERE id = :id")


async def record_usage(
    api_key_id: UUID,
    tenant_id: UUID,
    *,
    throttled: bool = False,
    now: datetime | None = None,
    engine: AsyncEngine | None = None,
) -> None:
    """Add one request to this key's hourly row. Never raises.

    A throttled request still counts as usage: the caller consumed capacity
    deciding to refuse it, and a quota report that hid refusals would misstate
    how hard a key is being driven.
    """
    try:
        async with (engine or get_engine()).begin() as conn:
            await conn.execute(
                _UPSERT_USAGE,
                {
                    "id": uuid7(),
                    "api_key_id": api_key_id,
                    "tenant_id": tenant_id,
                    "window_start": window_start(now),
                    "requests": 1,
                    "throttled": 1 if throttled else 0,
                },
            )
            await conn.execute(_TOUCH_KEY, {"id": api_key_id})
    except SQLAlchemyError as exc:
        # Metering is a record, not a control: losing a row must not lose a request.
        log.warning(
            "usage_not_recorded",
            api_key_id=str(api_key_id),
            tenant_id=str(tenant_id),
            error=f"{type(exc).__name__}: {exc}",
        )
