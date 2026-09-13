"""Ingest the eval corpus into a local collection.

A dev helper, not a service path. The API has no endpoint that creates a tenant
or a collection yet, so this writes those two rows itself — the same thing the
integration tests do, and the reason `make eval` can be run from a clean
database without a psql session.

Re-runnable. A document whose digest is already present in the collection is
skipped rather than ingested twice, because a second copy of a paper would put
near-identical chunks in competition for the same top-k slots and quietly
depress every recall number measured afterwards.

Blobs are not copied into `storage_dir`: the corpus already lives on disk under
the operator's own path, and `documents.storage_path` is nullable exactly for a
document ingested from a path rather than an upload.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from prism.collections import CollectionRef
from prism.config import Settings, get_settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.embeddings import EmbeddingProvider, get_embedding_provider
from prism.eval.golden import Corpus, CorpusDocument
from prism.ingestion import PDF_MIME_TYPE, create_document, ingest_document

__all__ = ["CorpusError", "IngestOutcome", "ingest_corpus", "sha256_of"]

log = structlog.get_logger(__name__)

_READ_CHUNK_BYTES = 1024 * 1024

_SELECT_TENANT = text("SELECT id FROM tenants WHERE name = :name")
_INSERT_TENANT = text("INSERT INTO tenants (id, name) VALUES (:id, :name)")
_SELECT_COLLECTION = text(
    "SELECT id, tenant_id FROM collections WHERE tenant_id = :tenant_id AND name = :name"
)
_INSERT_COLLECTION = text(
    "INSERT INTO collections (id, tenant_id, name, embedding_model, embedding_dim) "
    "VALUES (:id, :tenant_id, :name, :model, :dim) ON CONFLICT DO NOTHING"
)
_SELECT_BY_DIGEST = text(
    "SELECT id, status FROM documents WHERE collection_id = :collection_id AND sha256 = :sha256"
)


class CorpusError(RuntimeError):
    """A corpus file is missing or is not the file the manifest describes."""


@dataclass(frozen=True)
class IngestOutcome:
    collection_id: UUID
    ingested: tuple[str, ...]
    skipped: tuple[str, ...]
    pages: int
    chunks: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(document: CorpusDocument, corpus_dir: Path) -> Path:
    path = corpus_dir / document.filename
    if not path.is_file():
        hint = f" — fetch it from {document.source_url}" if document.source_url else ""
        raise CorpusError(f"{path} is missing{hint}")

    actual = sha256_of(path)
    if actual != document.sha256:
        raise CorpusError(
            f"{path} is not the file the manifest describes: "
            f"manifest {document.sha256[:12]}…, on disk {actual[:12]}…"
        )
    return path


async def _ensure_collection(
    name: str, *, tenant: str, provider: EmbeddingProvider, engine: AsyncEngine
) -> CollectionRef:
    """The collection, creating the tenant and collection if they are absent."""
    async with engine.begin() as conn:
        row = (await conn.execute(_SELECT_TENANT, {"name": tenant})).first()
        if row is None:
            tenant_id = uuid7()
            await conn.execute(_INSERT_TENANT, {"id": tenant_id, "name": tenant})
        else:
            tenant_id = row.id

        await conn.execute(
            _INSERT_COLLECTION,
            {
                "id": uuid7(),
                "tenant_id": tenant_id,
                "name": name,
                "model": provider.model,
                "dim": provider.dim,
            },
        )
        row = (await conn.execute(_SELECT_COLLECTION, {"tenant_id": tenant_id, "name": name})).one()
    return CollectionRef(tenant_id=row.tenant_id, collection_id=row.id)


async def ingest_corpus(
    corpus: Corpus,
    *,
    corpus_dir: Path,
    collection: str,
    tenant: str,
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
) -> IngestOutcome:
    """Ingest every manifest document not already present, by digest.

    Every file is verified against the manifest before anything is written, so a
    corrupt or substituted paper fails the whole run rather than landing half a
    corpus in the index.
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider()
    engine = engine or get_engine()

    paths = {document.filename: _verify(document, corpus_dir) for document in corpus.documents}
    ref = await _ensure_collection(collection, tenant=tenant, provider=provider, engine=engine)
    collection_id = ref.collection_id

    ingested: list[str] = []
    skipped: list[str] = []
    pages = chunks = 0

    for document in corpus.documents:
        path = paths[document.filename]
        async with engine.connect() as conn:
            existing = (
                await conn.execute(
                    _SELECT_BY_DIGEST,
                    {"collection_id": collection_id, "sha256": document.sha256},
                )
            ).first()

        if existing is not None and existing.status == "ready":
            skipped.append(document.filename)
            continue

        document_id = existing.id if existing is not None else uuid7()
        if existing is None:
            await create_document(
                document_id=document_id,
                collection_id=collection_id,
                tenant_id=ref.tenant_id,
                filename=document.filename,
                mime_type=PDF_MIME_TYPE,
                size_bytes=path.stat().st_size,
                sha256=document.sha256,
                engine=engine,
            )

        result = await ingest_document(
            path, document_id=document_id, provider=provider, settings=settings, engine=engine
        )
        ingested.append(document.filename)
        pages += result.pages
        chunks += result.chunks
        log.info(
            "corpus_document_ingested",
            filename=document.filename,
            pages=result.pages,
            chunks=result.chunks,
        )

    return IngestOutcome(
        collection_id=collection_id,
        ingested=tuple(ingested),
        skipped=tuple(skipped),
        pages=pages,
        chunks=chunks,
    )
