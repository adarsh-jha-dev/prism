"""A hand-built PDF, shared by the extraction and ingestion tests.

Hand-built rather than committed fixture files so the expected text is
visible in the test that asserts on it.
"""


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Each page is a list of text lines; [] is a page with no text layer."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages))), len(pages)
        ).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, lines in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        if lines:
            operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
            for line_number, line in enumerate(lines):
                if line_number:
                    operators.append("T*")
                operators.append(f"({_escape(line)}) Tj")
            operators.append("ET")
            content = "\n".join(operators).encode()
            objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (
                len(content),
                content,
            )
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        else:
            objects[page_id] = (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>"
            )

    out = bytearray(b"%PDF-1.7\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"

    xref_offset = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n" % size
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        out += b"%010d 00000 n \n" % offsets.get(number, 0)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_offset)
    return bytes(out)
