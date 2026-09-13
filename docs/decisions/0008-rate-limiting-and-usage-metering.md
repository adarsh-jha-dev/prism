# 0008 — Rate limiting in Redis, usage metering in Postgres

- **Status:** accepted
- **Date:** 2026-09-13
- **Closes:** the `REVIEW.md` tenancy gap "no usage-metering table exists at all"

## Context

`api_keys.rate_limit_rpm` has existed since migration 0001 and was enforced by
nothing. `REVIEW.md` also notes that RPM and the quota the dashboard shows
(`341k / 500k queries`) are different things: one bounds a burst, the other
accumulates over a billing period. A single mechanism cannot serve both.

## Decision

**Two stores, because the two questions have different durability needs.**

*Rate limiting* is a fixed one-minute window in Redis, counted with `INCR` and
expired by TTL. The count is allowed to be lost — losing it grants a caller one
extra window, which is the cheapest possible failure. A fixed window admits a
burst of up to `2 * rpm` across a boundary; a sliding window would cost a sorted
set per key per request to avoid a burst that `rpm` does not promise anything
about anyway.

*Usage metering* is `usage_records` in Postgres: one row per key per hour,
upserted with an atomic `ON CONFLICT DO UPDATE`. It must survive a restart —
this is what a quota, an invoice and the benchmark's per-tenant numbers are all
read from.

**The hour is the grain.** Per-request rows would be a request log, and the
benchmark's data model — including what a trace row looks like — is still an
open gap in `REVIEW.md`; writing one now would prejudge it. An hour aggregates
to a day or a month without loss and is narrow enough to show a burst on a
dashboard. Going finer later means a migration; going coarser is a `GROUP BY`.

**The limiter fails closed.** If Redis cannot be reached the request is refused
with 503, not allowed through unmetered. This follows the house rule that an
unverifiable condition is a refusal rather than an optimistic pass — the same
reason a query with no provider inside budget refuses instead of overspending.

**Metering never fails the request.** A usage row that cannot be written is
logged and the request proceeds. Metering is an observation of work already
authorized; dropping a row loses an accounting entry, while refusing would turn
a reporting outage into a service outage. This is deliberately the opposite
stance from the limiter, because the limiter is a control and metering is a
record.

**Headers on every response, not only on 429.** `X-RateLimit-Limit`,
`-Remaining` and `-Reset` are set on success too, so a client can back off
before being refused rather than discovering the limit by hitting it. A 429 adds
`Retry-After` in seconds.

## Consequences

- Redis moves from a dependency of the eventual semantic cache to a dependency
  of every authenticated request. `make up` already starts it, and `/health/deps`
  already reports it, but an API that could previously serve reads without Redis
  no longer can.
- The limit is per key, not per tenant. A tenant with several keys can exceed
  any single key's rpm; a tenant-wide ceiling is a separate column and a separate
  decision.
- A window is a wall-clock minute, so limits reset on the minute rather than
  rolling. Clients that retry on the reset second will synchronize; `Retry-After`
  is therefore the remaining seconds, not a fixed value.
- `usage_records` is the sixth table. It closes a gap `REVIEW.md` names rather
  than opening a new one, and it deliberately does not carry cost: `cost_usd`
  has no price basis yet, which is still an open gap.
