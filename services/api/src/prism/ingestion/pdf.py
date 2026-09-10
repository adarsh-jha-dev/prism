"""PDF text extraction.

Text layer only. A scanned page carries no text to extract, and OCR belongs to
the multimodal ingestion path rather than here — so a PDF with no text layer is
an error, not an empty document.
"""

from pathlib import Path
from typing import IO

from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

__all__ = ["ExtractionError", "extract_pages"]


class ExtractionError(RuntimeError):
    """A document's text could not be extracted.

    Never softened into an empty document: silently ingesting zero content
    surfaces much later as an unexplained retrieval miss.
    """


def extract_pages(source: str | Path | IO[bytes]) -> list[tuple[int, str]]:
    """Extract the text layer of a PDF as `(page_number, text)`, in page order.

    Page numbers are 1-based, matching `chunks.page_number` and the page numbers
    shown in the operator dashboard. Text is returned verbatim — no stripping,
    normalization, or joining — so that character offsets into it stay usable.

    A page with no text yields an empty string; blank pages are legitimate. A
    document in which *every* page is empty raises ExtractionError, as do an
    unreadable file, a password-protected one, and a PDF with no pages. A
    missing path raises OSError, unchanged — that is a caller bug, not a
    malformed document.
    """
    try:
        reader = PdfReader(source)
    except PyPdfError as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            opened = reader.decrypt("")
        except (PyPdfError, DependencyError, NotImplementedError) as exc:
            raise ExtractionError(f"could not decrypt PDF: {exc}") from exc
        if not opened:
            raise ExtractionError("PDF is password-protected")

    try:
        pages = [(number, page.extract_text()) for number, page in enumerate(reader.pages, start=1)]
    except PyPdfError as exc:
        raise ExtractionError(f"could not extract text: {exc}") from exc

    if not pages:
        raise ExtractionError("PDF has no pages")
    if not any(text.strip() for _, text in pages):
        raise ExtractionError(f"no text layer in any of {len(pages)} pages; PDF may be scanned")

    return pages
