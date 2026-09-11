"""Document ingestion: a file in, embedded chunk rows out."""

from prism.ingestion.chunking import chunk_text
from prism.ingestion.pdf import ExtractionError, extract_pages
from prism.ingestion.pipeline import (
    PDF_MIME_TYPE,
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    IngestionError,
    IngestionResult,
    assert_collection_ingestable,
    create_document,
    ingest_document,
)

__all__ = [
    "PDF_MIME_TYPE",
    "CollectionNotFoundError",
    "EmbeddingModelMismatchError",
    "ExtractionError",
    "IngestionError",
    "IngestionResult",
    "assert_collection_ingestable",
    "chunk_text",
    "create_document",
    "extract_pages",
    "ingest_document",
]
