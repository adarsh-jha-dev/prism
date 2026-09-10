# 0002 — Chunks stop at page boundaries, and carry their own reading order

- **Status:** accepted
- **Date:** 2026-09-10

## Context

The ingestion pipeline composes the three tested pieces — extraction, chunking,
embedding — into rows in `documents` and `chunks`. Two choices it had to make
are cheap now and expensive later, because changing either means re-ingesting
every document in the corpus.

**Where a chunk may end.** `extract_pages` returns `(page_number, text)` and
`chunks.page_number` is a single nullable integer. Concatenating the pages
before chunking would produce better windows at page joins, but a window
spanning pages 3 and 4 has no honest page number, and a citation is only as
trustworthy as that number.

**How reading order is recovered.** Nothing in `chunks` records position within
a document. The obvious answer was to lean on UUIDv7's time ordering, since ids
are already v7 and `ORDER BY id` costs nothing.

That answer is wrong, and the integration test caught it: every chunk of one
document is generated inside the same millisecond, and `prism.core.ids.uuid7`
fills the sub-millisecond bits with randomness (RFC 9562 method 3, no monotonic
counter). Below millisecond resolution, `ORDER BY id` is `ORDER BY random()`.

## Decision

Chunk each page independently, so `chunks.page_number` is exact rather than
approximate.

Record a document-wide 0-based `chunk_index` in `chunks.metadata`, assigned in
reading order across pages.

## Consequences

- Text that straddles a page break is split there, and a sentence crossing the
  boundary is retrievable only in halves. The chunk overlap does not bridge
  pages. Acceptable for now: a mis-attributed citation is a correctness failure,
  a slightly worse window is a recall cost.
- A short final chunk per page is normal — a 40-character page is a
  40-character chunk. Chunk size is a ceiling, not a target.
- Reading order lives in `metadata` rather than a column, so it needs no
  migration and no backfill. If ordering becomes a query predicate rather than a
  display concern, promote it to an indexed column then.
- `metadata` now has one defined key. The bounding-box shape for figures and
  tables is still unspecified (`docs/design/REVIEW.md`), and `chunk_index` does
  not constrain it.
- This does not touch citation character offsets, which remain an open design
  gap. `chunk_index` orders chunks; it does not locate a span inside one.
- Blank pages still count as pages. `IngestionResult.pages` and
  `IngestionResult.chunks` differ whenever a page carries no text layer, and
  that difference is the signal that a page was scanned rather than typed.
