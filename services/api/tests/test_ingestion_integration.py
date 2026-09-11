"""Integration: needs `make up`.

The embedding provider is stubbed, so these run without Ollama: what is under
test is the wiring — the rows written, the statuses, and what survives a
failure — not the vectors.
"""

import io
import json
from uuid import UUID

import pytest
from sqlalchemy import text

from pdf_builder import build_pdf
from prism.collections import (
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    assert_collection_compatible,
)
from prism.config import Settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.embeddings import EmbeddingError, EmbeddingProvider, OllamaEmbeddingProvider
from prism.ingestion import (
    PDF_MIME_TYPE,
    ExtractionError,
    IngestionError,
    IngestionResult,
    create_document,
    ingest_document,
)
from stub_provider import DIM, MODEL, StubProvider, vector_for

pytestmark = pytest.mark.integration


def settings(size: int = 1200, overlap: int = 150, batch: int = 64) -> Settings:
    return Settings(
        chunk_size_chars=size,
        chunk_overlap_chars=overlap,
        embed_batch_size=batch,
        embedding_model=MODEL,
        embedding_dim=DIM,
    )


async def ingest(
    source: io.BytesIO,
    *,
    collection_id: UUID,
    filename: str,
    provider: EmbeddingProvider,
    settings: Settings | None = None,
) -> IngestionResult:
    """Record the document, then ingest it — the order the endpoint uses."""
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        filename=filename,
        mime_type=PDF_MIME_TYPE,
    )
    return await ingest_document(
        source, document_id=document_id, provider=provider, settings=settings
    )


async def fetch_document(document_id: UUID) -> dict[str, object] | None:
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT filename, mime_type, status, ingested_at, collection_id "
                    "FROM documents WHERE id = :id"
                ),
                {"id": document_id},
            )
        ).first()
    return None if row is None else dict(row._mapping)


async def fetch_chunks(document_id: UUID) -> list[dict[str, object]]:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT id, collection_id, content, chunk_type, page_number, embedding, "
                "(metadata->>'chunk_index')::int AS chunk_index "
                "FROM chunks WHERE document_id = :id ORDER BY chunk_index"
            ),
            {"id": document_id},
        )
    return [dict(row._mapping) for row in rows]


async def test_a_pdf_becomes_a_ready_document_and_its_chunks(collection_id: UUID) -> None:
    pdf = io.BytesIO(build_pdf([["alpha"], ["beta"], ["gamma"]]))

    result = await ingest(
        pdf,
        collection_id=collection_id,
        filename="three-pages.pdf",
        provider=StubProvider(),
        settings=settings(),
    )

    assert (result.pages, result.chunks) == (3, 3)

    document = await fetch_document(result.document_id)
    assert document is not None
    assert document["filename"] == "three-pages.pdf"
    assert document["mime_type"] == "application/pdf"
    assert document["status"] == "ready"
    assert document["ingested_at"] is not None
    assert document["collection_id"] == collection_id

    chunks = await fetch_chunks(result.document_id)
    assert [c["content"] for c in chunks] == ["alpha", "beta", "gamma"]
    assert [c["page_number"] for c in chunks] == [1, 2, 3]
    assert {c["chunk_type"] for c in chunks} == {"text"}


async def test_chunks_carry_the_collection_id_denormalized(collection_id: UUID) -> None:
    """Scoping has to be a predicate inside the ANN query, so it lives on the row."""
    result = await ingest(
        io.BytesIO(build_pdf([["scoped"]])),
        collection_id=collection_id,
        filename="scoped.pdf",
        provider=StubProvider(),
        settings=settings(),
    )
    ((chunk,)) = await fetch_chunks(result.document_id)
    assert chunk["collection_id"] == collection_id


async def test_reading_order_is_recorded_because_ids_cannot_carry_it(
    collection_id: UUID,
) -> None:
    """A document's chunks share a millisecond, and UUIDv7 is random below that."""
    result = await ingest(
        io.BytesIO(build_pdf([["one"], ["two"], ["three"], ["four"]])),
        collection_id=collection_id,
        filename="ordered.pdf",
        provider=StubProvider(),
        settings=settings(),
    )
    chunks = await fetch_chunks(result.document_id)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]
    assert [c["content"] for c in chunks] == ["one", "two", "three", "four"]
    assert all(UUID(str(c["id"])).version == 7 for c in chunks)


async def test_chunk_index_runs_across_pages_not_within_them(collection_id: UUID) -> None:
    result = await ingest(
        io.BytesIO(build_pdf([["abcdef"], ["ghijkl"]])),
        collection_id=collection_id,
        filename="across.pdf",
        provider=StubProvider(),
        settings=settings(size=3, overlap=0),
    )
    chunks = await fetch_chunks(result.document_id)
    assert [(c["page_number"], c["chunk_index"]) for c in chunks] == [
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 3),
    ]


async def test_embeddings_land_in_the_vector_column(collection_id: UUID) -> None:
    result = await ingest(
        io.BytesIO(build_pdf([["embedded"]])),
        collection_id=collection_id,
        filename="embedded.pdf",
        provider=StubProvider(),
        settings=settings(),
    )
    ((chunk,)) = await fetch_chunks(result.document_id)
    stored = json.loads(str(chunk["embedding"]))
    assert len(stored) == DIM
    assert stored == pytest.approx(vector_for("embedded"), abs=1e-6)


async def test_an_ingested_chunk_is_retrievable_by_nearest_neighbour(
    collection_id: UUID,
) -> None:
    """The point of ingesting at all: the chunk comes back for its own vector."""
    await ingest(
        io.BytesIO(build_pdf([["needle"], ["haystack"]])),
        collection_id=collection_id,
        filename="retrievable.pdf",
        provider=StubProvider(),
        settings=settings(),
    )
    async with get_engine().connect() as conn:
        top = (
            await conn.execute(
                text(
                    "SELECT content FROM chunks WHERE collection_id = :c "
                    "ORDER BY embedding <=> CAST(:p AS vector) LIMIT 1"
                ),
                {"c": collection_id, "p": str(vector_for("needle"))},
            )
        ).scalar_one()
    assert top == "needle"


async def test_a_long_page_becomes_several_chunks_on_the_same_page(
    collection_id: UUID,
) -> None:
    page = "".join(f"line {n} of the page. " for n in range(40))
    result = await ingest(
        io.BytesIO(build_pdf([[page]])),
        collection_id=collection_id,
        filename="long.pdf",
        provider=StubProvider(),
        settings=settings(size=100, overlap=20),
    )
    chunks = await fetch_chunks(result.document_id)
    assert result.chunks > 1
    assert {c["page_number"] for c in chunks} == {1}


async def test_a_blank_page_produces_no_chunks_but_still_counts_as_a_page(
    collection_id: UUID,
) -> None:
    result = await ingest(
        io.BytesIO(build_pdf([["front"], [], ["back"]])),
        collection_id=collection_id,
        filename="gap.pdf",
        provider=StubProvider(),
        settings=settings(),
    )
    assert (result.pages, result.chunks) == (3, 2)
    assert [c["page_number"] for c in await fetch_chunks(result.document_id)] == [1, 3]


async def test_embedding_is_batched(collection_id: UUID) -> None:
    provider = StubProvider()
    await ingest(
        io.BytesIO(build_pdf([["a"], ["b"], ["c"], ["d"], ["e"]])),
        collection_id=collection_id,
        filename="batched.pdf",
        provider=provider,
        settings=settings(batch=2),
    )
    assert [len(batch) for batch in provider.batches] == [2, 2, 1]


async def test_a_scanned_pdf_fails_the_document_rather_than_ingesting_nothing(
    collection_id: UUID,
) -> None:
    """An empty document would surface later as an unexplained retrieval miss."""
    with pytest.raises(ExtractionError, match="no text layer"):
        await ingest(
            io.BytesIO(build_pdf([[], []])),
            collection_id=collection_id,
            filename="scanned.pdf",
            provider=StubProvider(),
            settings=settings(),
        )

    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, status FROM documents "
                    "WHERE collection_id = :c AND filename = 'scanned.pdf'"
                ),
                {"c": collection_id},
            )
        ).one()
    assert row.status == "failed"
    assert await fetch_chunks(row.id) == []


async def test_an_embedding_failure_leaves_a_failed_document_with_no_chunks(
    collection_id: UUID,
) -> None:
    with pytest.raises(EmbeddingError):
        await ingest(
            io.BytesIO(build_pdf([["good"], ["poison"]])),
            collection_id=collection_id,
            filename="halfway.pdf",
            provider=StubProvider(fail_on="poison"),
            settings=settings(),
        )

    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT id, status, ingested_at FROM documents "
                    "WHERE collection_id = :c AND filename = 'halfway.pdf'"
                ),
                {"c": collection_id},
            )
        ).one()
    assert row.status == "failed"
    assert row.ingested_at is None
    assert await fetch_chunks(row.id) == []


async def test_an_unknown_collection_writes_no_document_row() -> None:
    missing = uuid7()
    with pytest.raises(CollectionNotFoundError, match=f"collection {missing} does not exist"):
        await ingest(
            io.BytesIO(build_pdf([["orphan"]])),
            collection_id=missing,
            filename="orphan.pdf",
            provider=StubProvider(),
            settings=settings(),
        )

    async with get_engine().connect() as conn:
        count = (
            await conn.execute(
                text("SELECT count(*) FROM documents WHERE collection_id = :c"), {"c": missing}
            )
        ).scalar_one()
    assert count == 0


async def test_a_provider_the_collection_was_not_built_for_is_refused(
    collection_id: UUID,
) -> None:
    """Mixing embedding models in one index returns rows that mean nothing."""

    class WrongProvider(StubProvider):
        model = "bge-m3"
        dim = 1024

    with pytest.raises(EmbeddingModelMismatchError, match="nomic-embed-text/768-dim"):
        await assert_collection_compatible(collection_id, WrongProvider())


async def test_a_mismatched_provider_leaves_the_document_pending_not_failed(
    collection_id: UUID,
) -> None:
    """A stored document is not at fault for a misconfigured provider: it stays
    claimable, so fixing the configuration is enough to retry it."""
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        filename="mismatched.pdf",
        mime_type=PDF_MIME_TYPE,
    )

    class WrongProvider(StubProvider):
        model = "bge-m3"
        dim = 1024

    with pytest.raises(EmbeddingModelMismatchError):
        await ingest_document(
            io.BytesIO(build_pdf([["mismatched"]])),
            document_id=document_id,
            provider=WrongProvider(),
            settings=settings(),
        )

    document = await fetch_document(document_id)
    assert document is not None
    assert document["status"] == "pending"
    assert await fetch_chunks(document_id) == []


async def test_a_document_is_not_ingested_twice(collection_id: UUID) -> None:
    """Two workers pulling the same queue message must not double its chunks."""
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        filename="once.pdf",
        mime_type=PDF_MIME_TYPE,
    )
    first = await ingest_document(
        io.BytesIO(build_pdf([["only once"]])),
        document_id=document_id,
        provider=StubProvider(),
        settings=settings(),
    )
    assert first.chunks == 1

    with pytest.raises(IngestionError, match="refusing to ingest it twice"):
        await ingest_document(
            io.BytesIO(build_pdf([["only once"]])),
            document_id=document_id,
            provider=StubProvider(),
            settings=settings(),
        )
    assert len(await fetch_chunks(document_id)) == 1


async def test_a_failed_document_can_be_ingested_again(collection_id: UUID) -> None:
    """The blob outlives the failure, so a retry needs no second upload."""
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        filename="retried.pdf",
        mime_type=PDF_MIME_TYPE,
    )
    with pytest.raises(EmbeddingError):
        await ingest_document(
            io.BytesIO(build_pdf([["poison"]])),
            document_id=document_id,
            provider=StubProvider(fail_on="poison"),
            settings=settings(),
        )

    result = await ingest_document(
        io.BytesIO(build_pdf([["poison"]])),
        document_id=document_id,
        provider=StubProvider(),
        settings=settings(),
    )
    assert result.chunks == 1
    document = await fetch_document(document_id)
    assert document is not None
    assert document["status"] == "ready"


async def test_a_non_pdf_document_is_refused_without_being_claimed(
    collection_id: UUID,
) -> None:
    """Only the PDF path exists; a recorded document of another type waits for
    one rather than being marked failed."""
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        filename="diagram.png",
        mime_type="image/png",
    )

    with pytest.raises(IngestionError, match="unsupported mime type 'image/png'"):
        await ingest_document(
            io.BytesIO(build_pdf([["x"]])),
            document_id=document_id,
            provider=StubProvider(),
            settings=settings(),
        )

    document = await fetch_document(document_id)
    assert document is not None
    assert document["status"] == "pending"


async def test_an_unrecorded_document_cannot_be_ingested() -> None:
    missing = uuid7()
    with pytest.raises(IngestionError, match=f"document {missing} does not exist"):
        await ingest_document(
            io.BytesIO(build_pdf([["ghost"]])),
            document_id=missing,
            provider=StubProvider(),
            settings=settings(),
        )


async def test_the_stub_satisfies_the_real_provider_protocol() -> None:
    assert isinstance(StubProvider(), EmbeddingProvider)


@pytest.mark.ollama
async def test_end_to_end_with_the_real_embedding_provider(collection_id: UUID) -> None:
    """Everything at once: PDF in, and the ingested text is findable by meaning."""
    provider = OllamaEmbeddingProvider()
    result = await ingest(
        io.BytesIO(
            build_pdf([["The cat sat on the mat."], ["Quarterly revenue grew twelve percent."]])
        ),
        collection_id=collection_id,
        filename="real.pdf",
        provider=provider,
    )
    assert result.chunks == 2

    probe = await provider.embed_one("A cat is sitting on the rug.")
    async with get_engine().connect() as conn:
        top = (
            await conn.execute(
                text(
                    "SELECT content FROM chunks WHERE collection_id = :c "
                    "ORDER BY embedding <=> CAST(:p AS vector) LIMIT 1"
                ),
                {"c": collection_id, "p": str(probe)},
            )
        ).scalar_one()
    assert "cat" in top
