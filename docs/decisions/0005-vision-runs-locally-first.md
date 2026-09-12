# 0005 — Vision parsing runs on Ollama first, Gemini by escalation

- **Status:** accepted
- **Date:** 2026-09-12
- **Amends:** [0004](0004-vision-parsing-for-figures-and-tables.md)

## Context

ADR 0004 built the vision pass with a single provider, Gemini, because the
provider table assigns multimodal ingestion to the `gemini` lane. That made the
only way to parse a figure a paid call, which sits badly against two other
commitments: Ollama is meant to serve all dev and CI inference, and the cascade
is meant to be cheap-first with escalation, not paid-first.

It also had a practical cost. Ingestion could not be exercised end to end
without credits on a Google project, and the recorded cassettes — the thing
standing in for the provider in CI — could not be recorded at all until billing
was sorted out.

Ollama serves vision models (`qwen2.5vl`, `granite3.2-vision`, `minicpm-v`) and
supports constrained decoding through `format`, which is the same guarantee
`responseSchema` gives on the Gemini side. The `VisionProvider` protocol 0004
introduced already admits a second implementation.

## Decision

**`vision_lane` selects the provider: `ollama` (default) or `gemini`.**
`OllamaVisionProvider` posts to `/api/chat` with the page as a base64 image and
the figure schema in `format`; `GeminiVisionProvider` is unchanged and remains
the paid escalation.

**`qwen2.5vl:7b` is the local default** (`vision_ollama_model`), chosen over
`granite3.2-vision:2b` and `minicpm-v:8b` for table fidelity.

**The prompt and the figure-list validation are shared.** `PROMPT` and
`figures_from_json` moved to `prism.vision.base`; each provider owns only its
transport and its response envelope. Two providers that disagreed about what a
valid figure is would be a silent corpus inconsistency.

**Each provider keeps its own schema dialect.** Gemini takes uppercase
`OBJECT`/`ARRAY`; Ollama takes standard JSON Schema. Normalising them behind one
shape would be a translation layer for two call sites.

## Consequences

- Ingestion is free and keyless by default, and `make up` needs no credential to
  parse figures. CI still calls neither: the unit tier stubs the transport, and
  the live local test is marked `ollama` and deselected there.
- Quality is now a configuration choice rather than a fixed property. A 7B local
  model is weaker than `gemini-3.6-flash` on merged cells and dense charts, and
  chunk content feeds the eval set — so a lane switch can move eval numbers
  without any code change. `chunks.metadata.vision_model` records which model
  wrote each chunk, so this stays attributable after the fact.
- Local parsing is slower per page and bounded by the `ollama` lane's
  concurrency of 8 rather than gemini's 4. For ingestion, which is a background
  build step, that trade is worth taking.
- The failure policy from 0004 is unchanged and now covers a second way to fail:
  a model that was never pulled. `OllamaVisionProvider` says so in the error
  rather than reporting a page with no figures.
- **The local lane's output quality is not yet verified.** Its transport,
  schema and error handling are covered by unit tests against a stubbed
  transport, but no vision model has been successfully pulled on the
  development machine — both `qwen2.5vl:7b` and `granite3.2-vision:2b` stalled
  mid-download against the Ollama registry. `test_vision_ollama_live.py` exists
  to close this and is marked `ollama`; run it once a pull completes. Until
  then the verified path is Gemini, and `.env` selects that lane.
- Gemini's cassettes are recorded and stay the contract for the paid lane. The
  local lane has no cassette — a local model is reproducible by pulling it, and
  recording a 6GB model's output would fix a quality snapshot we would then have
  to maintain.
