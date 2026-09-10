"""Document ingestion: a file in, embedded chunk rows out."""

from prism.ingestion.chunking import chunk_text
from prism.ingestion.pdf import ExtractionError, extract_pages
from prism.ingestion.pipeline import (
    PDF_MIME_TYPE,
    IngestionError,
    IngestionResult,
    ingest_document,
)

__all__ = [
    "PDF_MIME_TYPE",
    "ExtractionError",
    "IngestionError",
    "IngestionResult",
    "chunk_text",
    "extract_pages",
    "ingest_document",
]
