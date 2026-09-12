"""A hand-built PDF, shared by the extraction, rendering and ingestion tests.

Hand-built rather than committed fixture files so the expected text — and the
exact number of drawn rules — is visible in the test that asserts on it.
"""


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    pages: list[list[str]],
    *,
    rules: dict[int, int] | None = None,
    images: dict[int, int] | None = None,
) -> bytes:
    """Each page is a list of text lines; [] is a page with no text layer.

    `rules` and `images` map a 0-based page index to a count of stroked
    rectangles or embedded 1x1 raster images to place on that page. Each
    stroked rectangle is one pdfium path object.
    """
    rules = rules or {}
    images = images or {}

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    next_id = 4
    page_ids: list[int] = []

    for index, lines in enumerate(pages):
        image_ids = []
        for _ in range(images.get(index, 0)):
            # 1x1 RGB: the smallest unambiguous raster image object.
            objects[next_id] = (
                b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\n"
                b"stream\n\x20\x40\x60\nendstream"
            )
            image_ids.append(next_id)
            next_id += 1

        operators: list[str] = []
        if lines:
            operators += ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
            for line_number, line in enumerate(lines):
                if line_number:
                    operators.append("T*")
                operators.append(f"({_escape(line)}) Tj")
            operators.append("ET")

        # Each `re ... S` pair closes one path object.
        for rule in range(rules.get(index, 0)):
            operators += ["0.5 w", f"72 {640 - rule * 18} 400 12 re", "S"]

        for slot, image_id in enumerate(image_ids):
            operators += ["q", f"120 0 0 90 72 {420 - slot * 110} cm", f"/Im{image_id} Do", "Q"]

        content_id = next_id
        next_id += 1
        content = "\n".join(operators).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)

        xobjects = (
            "/XObject << {} >> ".format(" ".join(f"/Im{i} {i} 0 R" for i in image_ids))
            if image_ids
            else ""
        )
        page_id = next_id
        next_id += 1
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> {xobjects}>> "
            f"/Contents {content_id} 0 R >>"
        ).encode()
        page_ids.append(page_id)

    objects[2] = "<< /Type /Pages /Kids [{}] /Count {} >>".format(
        " ".join(f"{page_id} 0 R" for page_id in page_ids), len(pages)
    ).encode()

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
