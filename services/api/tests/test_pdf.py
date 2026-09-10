import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from pdf_builder import build_pdf
from prism.ingestion.pdf import ExtractionError, extract_pages


def test_single_page() -> None:
    pdf = io.BytesIO(build_pdf([["Only page"]]))
    assert extract_pages(pdf) == [(1, "Only page")]


def test_page_numbers_are_one_based_and_ordered() -> None:
    pdf = io.BytesIO(build_pdf([["first"], ["second"], ["third"]]))
    pages = extract_pages(pdf)
    assert [number for number, _ in pages] == [1, 2, 3]
    assert [text for _, text in pages] == ["first", "second", "third"]


def test_text_is_associated_with_its_own_page() -> None:
    pdf = io.BytesIO(build_pdf([["alpha"], ["beta"], ["gamma"]]))
    assert dict(extract_pages(pdf))[2] == "beta"


def test_multiple_lines_on_one_page_are_preserved() -> None:
    pdf = io.BytesIO(build_pdf([["line one", "line two", "line three"]]))
    ((_, text),) = extract_pages(pdf)
    assert text.splitlines() == ["line one", "line two", "line three"]


def test_blank_page_between_text_pages_yields_empty_string() -> None:
    pdf = io.BytesIO(build_pdf([["front"], [], ["back"]]))
    assert extract_pages(pdf) == [(1, "front"), (2, ""), (3, "back")]


def test_document_with_no_text_layer_raises() -> None:
    pdf = io.BytesIO(build_pdf([[], [], []]))
    with pytest.raises(ExtractionError, match="no text layer in any of 3 pages"):
        extract_pages(pdf)


def test_whitespace_only_document_counts_as_no_text_layer() -> None:
    pdf = io.BytesIO(build_pdf([["   "], ["\t"]]))
    with pytest.raises(ExtractionError, match="no text layer"):
        extract_pages(pdf)


def test_accepts_a_path(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(build_pdf([["from disk"]]))
    assert extract_pages(target) == [(1, "from disk")]


def test_accepts_a_path_string(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(build_pdf([["from disk"]]))
    assert extract_pages(str(target)) == [(1, "from disk")]


def test_parentheses_and_backslashes_survive_extraction() -> None:
    pdf = io.BytesIO(build_pdf([[r"cost (usd) is c:\rate"]]))
    assert extract_pages(pdf) == [(1, r"cost (usd) is c:\rate")]


def test_unreadable_file_raises_extraction_error() -> None:
    with pytest.raises(ExtractionError, match="could not read PDF"):
        extract_pages(io.BytesIO(b"this is not a PDF at all"))


def test_empty_file_raises_extraction_error() -> None:
    with pytest.raises(ExtractionError, match="could not read PDF"):
        extract_pages(io.BytesIO(b""))


def test_missing_path_raises_oserror_not_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        extract_pages(tmp_path / "nope.pdf")


def test_returns_plain_tuples_of_int_and_str() -> None:
    pdf = io.BytesIO(build_pdf([["x"]]))
    ((number, text),) = extract_pages(pdf)
    assert isinstance(number, int)
    assert isinstance(text, str)


def _encrypt(pdf: bytes, user_password: str, owner_password: str = "owner") -> io.BytesIO:
    writer = PdfWriter(clone_from=io.BytesIO(pdf))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def test_password_protected_pdf_raises() -> None:
    encrypted = _encrypt(build_pdf([["secret"]]), user_password="hunter2")
    with pytest.raises(ExtractionError, match="password-protected"):
        extract_pages(encrypted)


def test_encrypted_with_empty_user_password_extracts_normally() -> None:
    # Encrypted purely to restrict editing; still readable without a password.
    encrypted = _encrypt(build_pdf([["readable"]]), user_password="")
    assert extract_pages(encrypted) == [(1, "readable")]
