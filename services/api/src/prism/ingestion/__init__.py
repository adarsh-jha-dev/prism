"""Document ingestion: a file in, embedded chunk rows out."""

from prism.collections import CollectionNotFoundError, EmbeddingModelMismatchError
from prism.ingestion.chunking import chunk_text
from prism.ingestion.pdf import ExtractionError, extract_pages
from prism.ingestion.pipeline import (
    PDF_MIME_TYPE,
    IngestionError,
    IngestionResult,
    PlannedChunk,
    create_document,
    ingest_document,
    parse_figures,
    plan_document_chunks,
)
from prism.ingestion.render import RenderError, render_figure_pages

__all__ = [
    "PDF_MIME_TYPE",
    "CollectionNotFoundError",
    "EmbeddingModelMismatchError",
    "ExtractionError",
    "IngestionError",
    "IngestionResult",
    "PlannedChunk",
    "RenderError",
    "chunk_text",
    "create_document",
    "extract_pages",
    "ingest_document",
    "parse_figures",
    "plan_document_chunks",
    "render_figure_pages",
]
