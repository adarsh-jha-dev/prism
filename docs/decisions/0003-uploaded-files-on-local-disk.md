# 0003 — Uploaded files live on local disk, and the row records where

- **Status:** accepted
- **Date:** 2026-09-11

## Context

`POST /collections/{id}/documents` has to put the bytes somewhere before it can
ingest them, and the schema had nowhere to say where they went. Three questions
came with that: what the storage medium is, whether the location is recorded or
derived, and which component owns the `documents` row.

Ingestion runs inside the request for now. A queue is the eventual shape, and
the point of interest is that a queue worker cannot be handed an in-memory
upload — it can only be handed a durable reference. Whatever this endpoint does
becomes that reference.

## Decision

**Local filesystem, at `Settings.storage_dir` (`var/uploads`, a named volume in
compose), keyed by ids: `<collection_id>/<document_id>.pdf`.** No part of a
client-supplied filename reaches a path; the filename is recorded on the row as
a label. Sharding by collection keeps one directory from accumulating every
tenant's uploads. S3 is the deployed shape and changes only `prism.storage`.

**The path is recorded, not derived.** Migration `0002` adds
`documents.storage_path` (root-relative), `size_bytes` and `sha256`, all
nullable — a document ingested from a path rather than an upload has no blob of
its own. Root-relative so the root can move without a backfill.

**The blob is written before the row exists**, and the caller generates the
document id so it can name the file. A crash between the two leaves an
unreferenced file — sweepable — rather than a `documents` row whose bytes never
arrived, which would sit `pending` forever and be visible in the dashboard as a
document that can never be ingested.

**The uploader creates the row; the pipeline advances it.** `create_document`
records a `pending` document; `ingest_document` claims it — `UPDATE ... WHERE
status IN ('pending','failed')`, so the claim *is* the check — and carries it to
`ready` or `failed`. `ingest_document` no longer takes a collection or filename;
it takes a document id and reads the rest from the row.

**A failed ingest keeps its blob.** The bytes are what a retry needs.

## Consequences

- Replacing the synchronous call with an enqueue touches one module. The row and
  its blob are already the durable handoff, and `pending` — a status the CHECK
  constraint has always allowed and nothing wrote — is already the queue's input
  state.
- Two workers cannot double a document's chunks: the conditional claim admits
  one. Retry from `failed` stays safe because the chunk insert and the `ready`
  transition share a transaction, so a failure leaves no chunks to duplicate.
- `sha256` is recorded but nothing reads it yet. It is free at write time (the
  bytes stream through a digest on the way to disk) and impossible to backfill
  cheaply once blobs accumulate.
- A local directory is not durable storage. It survives a container rebuild via
  the compose volume and nothing more; two API replicas would not share it.
  That is the S3 swap, not a change to this design.
- An embedding-model mismatch leaves the document `pending` rather than
  `failed`: the document is not at fault for a misconfigured provider, and
  fixing the configuration should be enough to retry it.
