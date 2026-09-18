# 0015 — The checkpointer's tables are the library's; Alembic decides when they appear

- **Status:** accepted
- **Date:** 2026-09-18
- **Relates to:** [0006](0006-postgres-rls-for-tenant-isolation.md),
  [0012](0012-query-trace-and-citation-model.md),
  [0014](0014-circuit-breaker-state-is-per-process.md)

## Context

Stage D needs a checkpointer. `CLAUDE.md` requires that every node write a trace
row *and* a checkpoint, and that any query be replayable and forkable from any
node. ADR 0012 already reserved the handle — `queries.thread_id text NOT NULL
UNIQUE` — and already said the checkpointer owns its own tables and gets no
foreign key into them.

What it did not decide is who runs the DDL, and that collides with a rule this
repo has held since migration 0001: **every table is Alembic-owned, and
`alembic upgrade head` is the whole schema.** That rule has three callers today,
not one:

- `make migrate`
- the compose `migrate` one-shot, which `api` waits on
- the CI `Migrate` step, which runs before a test suite that includes the
  integration tier

Anything the checkpointer needs that Alembic does not produce has to be taught
to all three, forever.

### What the library actually does

`langgraph-checkpoint-postgres` (3.1.2, against `langgraph` 1.2.11) carries a
`MIGRATIONS` list of ten statements whose *list index is the version number*,
and tracks the applied version in a table of its own:

```
checkpoint_migrations (v INTEGER PRIMARY KEY)
checkpoints        (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
                    parent_checkpoint_id TEXT, type TEXT, checkpoint JSONB,
                    metadata JSONB, PK (thread_id, checkpoint_ns, checkpoint_id))
checkpoint_blobs   (thread_id, checkpoint_ns, channel, version, type, blob BYTEA)
checkpoint_writes  (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                    channel, type, blob BYTEA, task_path TEXT NOT NULL DEFAULT '')
```

`setup()` creates `checkpoint_migrations`, reads `max(v)`, applies the tail of
the list and inserts each version as it goes. Four properties of that routine
decide this ADR:

1. **Three of the ten statements are `CREATE INDEX CONCURRENTLY`**, which
   Postgres refuses inside a transaction block. A tenth adds `task_path` to
   `checkpoint_writes`, and a fifth drops a NOT NULL. So the list is a real
   migration history, not a one-shot `CREATE TABLE`, and it has already changed
   shape at least twice.
2. **It is idempotent and version-aware.** Running it on a database already at
   version 9 applies nothing.
3. **It takes no lock.** Two workers calling `setup()` at the same time both
   read the same `max(v)` and both try to insert it; one gets a primary-key
   violation. `setup()` is a deploy step, not a startup step, under every option
   below.
4. **The connection must be `autocommit=True`.** The row factory does not
   matter — every cursor the saver opens sets `row_factory=dict_row` itself —
   but without autocommit the DDL is not committed and the concurrent indexes
   cannot run at all.

The tables carry `thread_id TEXT`, no tenant column, no foreign key, and blobs
serialized with the library's own codec. That is the schema we are deciding how
to own.

## Option 1 — the checkpointer owns its tables outside Alembic

A `prism.graph.checkpointer setup` entry point, called once per deploy.

**For.** The schema is authored by the code that reads it, so it cannot be
wrong. Library upgrades apply themselves: bump the pin, run setup, the delta
lands — including the concurrent indexes, which Alembic's transactional
migrations cannot issue without extra ceremony. This is also the path every
LangGraph deployment takes, so it is the one with the fewest surprises.

**Against.** `alembic upgrade head` stops being the whole schema, and the cost
is not one line in the Makefile — it is a second step in the compose one-shot
and in CI as well, plus every future runner. The failure mode when someone
forgets is not a migration error; it is `relation "checkpoints" does not exist`
raised from inside a node, at query time. `make migrate` can wrap both commands
and read as one path, but the invariant that actually protects us is the one
about `alembic upgrade head`, and this option gives exactly that one up.

## Option 2 — vendor the DDL into migration 0010, never call `setup()`

**For.** One path, no argument. The four tables appear in a migration like every
other table, `downgrade()` drops them, and the schema is reviewable in the diff.
`CREATE INDEX CONCURRENTLY` is not needed on a table being created empty, so the
indexes vendor as ordinary ones.

**Against.** It is a hand-maintained copy of a schema we do not control, and the
copy is only correct against one pinned version. To leave the door open we would
also have to seed `checkpoint_migrations` with `v = 0..9`, which is our code
asserting on the library's behalf that its first ten migrations have been
applied. If the transcription is wrong, the library believes it is up to date
and never repairs it. When the pin moves — and `langgraph-checkpoint-postgres`
is on its third major — a human has to notice, read the new statements, and
transcribe them again. Nothing about that is enforced by the tool that is
supposed to be enforcing schema discipline, and the failure is silent until a
query references a column that was added upstream. `task_path` is exactly that
column, added in version 9 of a list that started at 3.

## Option 3 — our own saver on the SQLAlchemy session

**For.** It is the only option that can give checkpoints a `tenant_id`, which is
the only option that can put them under RLS when ADR 0006 step 3 lands. It is
the only option where a node's trace row and its checkpoint commit in one
transaction — ADR 0012 says a trace row without its checkpoint is a picture of a
run you cannot re-enter, and this is the option that makes that pairing
atomic rather than merely intended. One pool, one driver, one retention path.

**Against.** It does not save the dependency: `BaseCheckpointSaver` and the
serializer live in `langgraph-checkpoint`, which `langgraph` pulls in anyway, so
the saving is one package (`langgraph-checkpoint-postgres`, whose own additions
are `orjson` and `psycopg-pool` — and `psycopg[binary,pool]` is already a
dependency). What we would be taking on instead is a protocol that moves:
`get_tuple`/`list`/`put`/`put_writes`/`delete_thread` and their async twins, the
channel-versions-to-blobs split, the typed serde format, and whatever the next
minor adds. The payoff is Phase 3 value — RLS on checkpoints, atomic writes —
bought at Phase 2 cost, in the commit whose entire point is that the plumbing is
real and nothing else is.

## Decision

**Recommended: the library authors the DDL, and Alembic decides when it runs.**

Migration `0010` calls the library's `setup()` rather than transcribing it:

```python
with op.get_context().autocommit_block():
    PostgresSaver(op.get_bind().connection.driver_connection).setup()
```

The autocommit block is what makes the three `CREATE INDEX CONCURRENTLY`
statements legal; if borrowing Alembic's connection turns out not to yield an
autocommit psycopg connection, the migration opens its own short-lived one from
`Settings.database_url` instead, which is equivalent and no less contained.
`downgrade()` drops the four tables.

This is option 1's answer to *who authors the DDL* with option 2's answer to
*what triggers it*. It is deliberately not one of the three as posed, because
the question is two questions and options 1 and 2 each pay full price for
conflating them. **If the hybrid is rejected, take option 1**: a copy of a
library's schema that only a human can keep current is worse than an extra
deploy step, because the extra step fails loudly and the stale copy does not.

Three things keep it honest:

- **The pin is exact, and a bump fails in CI.** A unit test asserts
  `len(MIGRATIONS) == 10` against a constant recorded here. Raising the pin
  breaks that test, and the fix is migration `0011`, three lines, calling
  `setup()` again — idempotent by construction, applying only the delta.
- **`setup()` is never called at startup, from a node, or from a test fixture.**
  It has no lock; N workers racing it is a primary-key violation on
  `checkpoint_migrations`. Migration time is the only time the database is being
  changed by one process on purpose.
- **We do not name a table `checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes` or `checkpoint_migrations`.** Those four names are the
  library's, unqualified, in `public`.

### The connection

A dedicated `psycopg_pool.AsyncConnectionPool`, opened in `lifespan_resources`
beside the SQLAlchemy engine and exposed as `get_checkpointer()` returning a
`BaseCheckpointSaver`, with `autocommit=True`, `prepare_threshold=0` and
`row_factory=dict_row`.

Not SQLAlchemy's pool. The saver wants a psycopg connection in autocommit with
prepared statements off; reaching through `engine.raw_connection()` to get one
means handing back a connection whose settings we mutated to a pool whose other
users assume we did not. Not `from_conn_string` per query either — that is a
connect per query, under the one workload the benchmark exists to measure.

Nodes never see the pool. They see `get_checkpointer()`, which is the seam that
makes option 3 a one-file change later rather than a rewrite.

## Consequences

- **Trace rows and checkpoints are not written in one transaction.** The trace
  row goes through SQLAlchemy inside the node; the checkpoint goes through the
  psycopg pool after the node returns. A crash in between leaves one without the
  other, and ADR 0012's retention — which must drop both together — becomes a
  two-store operation. It has a supported API on both sides:
  `adelete_thread(thread_id)` exists, so retention never writes raw SQL into
  library tables. Commit 10's tests assert both artifacts exist after a run;
  they cannot assert atomicity, because there is none.
- **Checkpoints will never be under RLS.** The tables have no tenant column and
  we are not adding one to someone else's schema. The state they hold includes
  the question text and retrieved chunk ids, so the rule is: **`thread_id` is
  never accepted from a client.** Resume and fork take a `query_id`, resolve it
  to `thread_id` through `queries` under the tenant predicate, and the
  unguessable-by-construction thread id is the second layer, not the first.
  If RLS step 3 lands and checkpoints are the last unscoped store, that is the
  criterion for revisiting option 3 — not a mood.
- **`AsyncPostgresSaver` holds one `asyncio.Lock` around every cursor it opens.**
  All checkpoint I/O in a process serializes on it regardless of pool size, and
  a query checkpoints at every node. At ~612 queries/24h it should be invisible;
  under the concurrent-load benchmark it is a candidate p95 contributor and
  should be measured rather than assumed away. It is also why the pool is small.
- **Postgres connections per process go up**: SQLAlchemy's 10 + 5 plus the
  checkpointer pool, against a container default of 100.
- **Migration 0010's output depends on the installed library version.** That is
  the price of not transcribing, and `uv.lock` with `uv sync --frozen` is what
  makes it deterministic for CI and dev. `alembic upgrade --sql` cannot render
  0010; this repo does not use offline mode.
- **`target_metadata` is still `None`, so autogenerate is not a concern today.**
  If ORM models ever arrive, the four library tables must be excluded from
  autogenerate, or Alembic will offer to drop them.
- **The shallow saver is not used.** `AsyncShallowPostgresSaver` keeps only the
  latest checkpoint per thread, which cannot satisfy "forkable from any node".
- No new network egress: the checkpointer talks to Postgres and nothing else,
  and CI's no-paid-call rule is untouched.

## Alternatives considered

- **Options 1, 2 and 3 as posed**, above. Option 1 is the fallback and is not a
  bad answer; option 2 trades a silent failure for a tidy diff; option 3 is
  deferred behind a named criterion, not discarded.
- **`setup()` at application startup**, guarded by a flag. It is the common
  shortcut and it races with itself across workers, has no lock to fix that, and
  puts DDL on the path of a process whose job is to serve queries.
- **A separate Postgres schema for the library's tables**, to keep `public`
  clean. Whether the pinned version supports one is not something to assume, and
  four fixed names in `public` is a smaller problem than a schema search-path
  rule that every future connection has to honour.
- **An in-memory saver for tests.** Tempting, and it would make the commit-10
  tests unit-tier. Rejected: the point of this commit is that the checkpoints
  are real, and a test against a different saver proves nothing about the one
  that runs. The tests are integration-marked, as the brief says.
