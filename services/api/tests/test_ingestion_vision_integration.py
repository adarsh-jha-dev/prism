"""Integration: figure chunks as rows. Needs `make up`.

Both providers are stubbed — no Ollama, no paid call.
"""

import io
from uuid import UUID

import pytest
from sqlalchemy import text

from pdf_builder import build_pdf
from prism.collections import CollectionRef
from prism.config import Settings
from prism.core.ids import uuid7
from prism.db import get_engine
from prism.ingestion import PDF_MIME_TYPE, create_document, ingest_document
from prism.vision import ParsedFigure, VisionError
from stub_provider import DIM, MODEL, StubProvider
from stub_vision import CHART, TABLE, StubVision

pytestmark = pytest.mark.integration


def settings(**kwargs: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": MODEL,
        "embedding_dim": DIM,
        "vision_enabled": True,
        "vision_render_dpi": 72,
        "vision_min_path_objects": 6,
    }
    return Settings(**{**defaults, **kwargs})  # type: ignore[arg-type]


async def ingest(
    pdf: bytes,
    *,
    collection_id: UUID,
    vision: StubVision | None,
    config: Settings | None = None,
    collection: CollectionRef,
) -> tuple[UUID, int]:
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        filename="figures.pdf",
        mime_type=PDF_MIME_TYPE,
    )
    result = await ingest_document(
        io.BytesIO(pdf),
        document_id=document_id,
        provider=StubProvider(),
        vision=vision,
        settings=config or settings(),
    )
    return result.document_id, result.figures


async def fetch_chunks(document_id: UUID) -> list[dict[str, object]]:
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT content, chunk_type, page_number, metadata, "
                "(metadata->>'chunk_index')::int AS chunk_index "
                "FROM chunks WHERE document_id = :id ORDER BY chunk_index"
            ),
            {"id": document_id},
        )
    return [dict(row._mapping) for row in rows]


async def fetch_status(document_id: UUID) -> str | None:
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text("SELECT status FROM documents WHERE id = :id"), {"id": document_id}
            )
        ).first()
    return None if row is None else str(row.status)


async def test_a_table_page_writes_a_table_chunk(
    collection_id: UUID, collection: CollectionRef
) -> None:
    pdf = build_pdf([["Revenue by region"]], rules={0: 8})

    document_id, figures = await ingest(
        pdf, collection_id=collection_id, collection=collection, vision=StubVision([TABLE])
    )

    assert figures == 1
    text_chunk, table = await fetch_chunks(document_id)
    assert text_chunk["chunk_type"] == "text"
    assert table["chunk_type"] == "table"
    assert table["content"] == "| region | p95 |"
    assert table["page_number"] == 1


async def test_the_check_constraint_accepts_every_kind_vision_produces(
    collection_id: UUID, collection: CollectionRef
) -> None:
    # figure/table/equation are already in chunks_chunk_type_check.
    equation = ParsedFigure(kind="equation", content=r"E = mc^2")
    document_id, _ = await ingest(
        build_pdf([["x"]], rules={0: 8}),
        collection_id=collection_id,
        collection=collection,
        vision=StubVision([TABLE, CHART, equation]),
    )
    chunks = await fetch_chunks(document_id)
    assert [c["chunk_type"] for c in chunks] == ["text", "table", "figure", "equation"]


async def test_chunk_index_runs_unbroken_across_text_and_figures(
    collection_id: UUID, collection: CollectionRef
) -> None:
    pdf = build_pdf([["page one"], ["page two"], ["page three"]], rules={1: 8})
    document_id, _ = await ingest(
        pdf,
        collection_id=collection_id,
        collection=collection,
        vision=StubVision([TABLE, CHART]),
    )

    chunks = await fetch_chunks(document_id)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3, 4]
    assert [(c["page_number"], c["chunk_type"]) for c in chunks] == [
        (1, "text"),
        (2, "text"),
        (2, "table"),
        (2, "figure"),
        (3, "text"),
    ]


async def test_a_figure_chunk_records_that_a_model_wrote_it(
    collection_id: UUID, collection: CollectionRef
) -> None:
    document_id, _ = await ingest(
        build_pdf([["x"]], rules={0: 8}),
        collection_id=collection_id,
        collection=collection,
        vision=StubVision([TABLE]),
    )
    text_chunk, table = await fetch_chunks(document_id)

    assert table["metadata"] == {
        "chunk_index": 1,
        "source": "vision",
        "vision_model": "gemini-3.6-flash",
        "caption": "Table 1.",
    }
    assert text_chunk["metadata"] == {"chunk_index": 0}


async def test_figure_chunks_are_embedded_like_any_other(
    collection_id: UUID, collection: CollectionRef
) -> None:
    document_id, _ = await ingest(
        build_pdf([["x"]], rules={0: 8}),
        collection_id=collection_id,
        collection=collection,
        vision=StubVision([TABLE]),
    )
    async with get_engine().connect() as conn:
        missing = (
            await conn.execute(
                text("SELECT count(*) FROM chunks WHERE document_id = :id AND embedding IS NULL"),
                {"id": document_id},
            )
        ).scalar_one()
    assert missing == 0


async def test_vision_off_is_exactly_the_text_pipeline(
    collection_id: UUID, collection: CollectionRef
) -> None:
    vision = StubVision([TABLE])
    document_id, figures = await ingest(
        build_pdf([["x"]], rules={0: 8}),
        collection_id=collection_id,
        collection=collection,
        vision=vision,
        config=settings(vision_enabled=False),
    )

    assert (vision.calls, figures) == (0, 0)
    chunks = await fetch_chunks(document_id)
    assert [c["chunk_type"] for c in chunks] == ["text"]


async def test_a_prose_document_never_reaches_the_provider(
    collection_id: UUID, collection: CollectionRef
) -> None:
    vision = StubVision([TABLE])
    _, figures = await ingest(
        build_pdf([["just words here"]]),
        collection_id=collection_id,
        collection=collection,
        vision=vision,
    )
    assert (vision.calls, figures) == (0, 0)


async def test_a_vision_failure_fails_the_document_and_writes_no_chunks(
    collection_id: UUID, collection: CollectionRef
) -> None:
    document_id = uuid7()
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        filename="figures.pdf",
        mime_type=PDF_MIME_TYPE,
    )

    with pytest.raises(VisionError, match="stub provider is down"):
        await ingest_document(
            io.BytesIO(build_pdf([["Table 1"]], rules={0: 8})),
            document_id=document_id,
            provider=StubProvider(),
            vision=StubVision([], fail=True),
            settings=settings(),
        )

    assert await fetch_status(document_id) == "failed"
    assert await fetch_chunks(document_id) == []


async def test_a_failed_document_can_be_ingested_again_once_vision_recovers(
    collection_id: UUID, collection: CollectionRef
) -> None:
    document_id = uuid7()
    pdf = build_pdf([["Table 1"]], rules={0: 8})
    await create_document(
        document_id=document_id,
        collection_id=collection_id,
        tenant_id=collection.tenant_id,
        filename="figures.pdf",
        mime_type=PDF_MIME_TYPE,
    )

    with pytest.raises(VisionError):
        await ingest_document(
            io.BytesIO(pdf),
            document_id=document_id,
            provider=StubProvider(),
            vision=StubVision([], fail=True),
            settings=settings(),
        )

    result = await ingest_document(
        io.BytesIO(pdf),
        document_id=document_id,
        provider=StubProvider(),
        vision=StubVision([TABLE]),
        settings=settings(),
    )

    assert result.figures == 1
    assert await fetch_status(document_id) == "ready"
    assert [c["chunk_type"] for c in await fetch_chunks(document_id)] == ["text", "table"]
