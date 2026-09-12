# 0004 — Figures and tables are parsed by a vision model, and marked as such

- **Status:** accepted
- **Date:** 2026-09-12

## Context

The text path ingests a PDF's text layer and nothing else. A table drawn with
ruled lines and a chart drawn as vector paths carry no extractable text, so a
document can ingest cleanly to `ready` while the numbers a reader actually wants
are absent from `chunks` entirely. Retrieval cannot tell that from a document
that never mentioned them.

Closing that gap means the first paid provider in the codebase, which drags in
four decisions at once: which pages are worth paying for, what the provider is
shown, what a parsed figure becomes in the schema, and what happens when the
call fails. `chunks.chunk_type` has allowed `figure | table | equation` since
migration `0001` and nothing has ever written them.

## Decision

**Pages are selected by their page-object inventory, not their text.** pdfium
enumerates the objects on a page; a page qualifies if it carries at least one
raster image, or at least `vision_min_path_objects` (6) vector path objects.
Counting raster images alone — all pypdf can do — would miss ruled tables and
vector charts, which is most of the content this feature exists for.

**The whole page is rendered and sent, at `vision_render_dpi` (150).** Not the
cropped figure: a chart without its caption and axis labels is much harder to
describe, and cropping needs bounding boxes, which are an open design gap.

**PNG is encoded in-process from pdfium's bitmap** (`prism.ingestion.render`),
so the image stack is one dependency — `pypdfium2`, a pure wheel — rather than
two.

**A parsed figure is its own chunk, marked as model-written.** `chunk_type` is
the model's `figure|table|equation`, `content` is the model's markdown or
description, and `metadata` carries `source: "vision"`, the `vision_model`, and
the printed `caption` if there was one. The caption stays in metadata rather
than being prepended to the content: the caption is the document's words and
the content is the model's, and `verify_grounding` will need that line to be
sharp. Reading order is one unbroken `chunk_index` across both sources, with a
page's text before its figures.

**Vision is off by default (`vision_enabled`), and when it fails, the document
fails.** No chunks are written, the row goes to `failed`, the blob is kept, and
a retry is a re-ingest. A per-document ceiling (`vision_max_pages`, 20) is also
a hard failure rather than a truncation.

**The model is `gemini-3.6-flash`, not the `gemini-2.5-flash` the provider
table originally pinned.** 2.5-flash returns 404 for API keys issued after its
deprecation ("no longer available to new users"), so the original pin was not a
usable default. CLAUDE.md's model table is updated to match.

**The provider is replayed in tests, never called.** `tests/cassette.py` matches
on method and URL; `scripts/record_fixtures.py` (`make record-fixtures`) records
against the live API with a real key, records no request headers, and refuses to
write a file the key appears in.

## Consequences

- A keyless environment — CI, and dev by default — runs exactly the pipeline it
  ran before. Turning vision on is the only way to spend money, and it is a
  settings change, not a code path.
- `vision_min_path_objects` is the cost knob, and it is a heuristic with two
  honest failure modes: a page whose only path objects are header rules buys a
  call that returns nothing, and a borderline table below the threshold is
  silently left to the text path. The first is visible as a cost; the second is
  not visible at all. Raising recall here means paying for more pages.
- Re-ingesting a document re-parses its figures. `temperature: 0` and a pinned
  response schema make that as reproducible as the provider allows, but it is
  not a guarantee, and a re-ingest can change chunk content under an eval run.
- Committed cassettes are `"provenance": "synthetic"` — hand-authored from the
  documented response shape. Recording against the live API was attempted and
  failed: the Gemini project returns 429 `RESOURCE_EXHAUSTED` on every model.
  They exercise our parsing and are *not* evidence about Gemini's behaviour.
  Re-record with `make record-fixtures` once the project has credits.
- Cost is logged as tokens and not converted to dollars. There is still no
  `model_pricing` table (`REVIEW.md` B4), and a rate hardcoded at this call site
  is precisely the unrecorded price basis that gap is about.
- No circuit breaker yet. The lane's concurrency cap is honoured with a
  semaphore, and a failing page cancels its siblings, but repeated failures
  across documents are not tracked. That is Phase 3's provider interface.
- Bounding boxes are still not stored. `chunks.metadata` is where they will go
  when the citation-offset gap closes; nothing here forecloses that.

## Alternatives considered

- **Extract embedded images with pypdf and send those.** No new dependency, but
  blind to vector-drawn tables and charts, and a cropped image loses the caption
  that explains it.
- **Render and send every page.** Complete and simple, and directly contrary to
  the cost discipline the benchmark is meant to demonstrate.
- **Keep the text chunks and mark the document `partial` on a vision failure.**
  Needs a new `documents.status` value and an error column, and introduces a
  half-ingested document — a second thing retrieval would have to reason about,
  for a failure that a retry already fixes.
- **Fold figure descriptions into the page's text chunks.** Keeps one chunk
  type, and makes generated text indistinguishable from extracted text, which
  would leave grounding verification unfalsifiable.
