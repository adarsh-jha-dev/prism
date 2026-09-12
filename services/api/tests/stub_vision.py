"""A vision provider that never leaves the process."""

from prism.vision import ParsedFigure, VisionError

TABLE = ParsedFigure(kind="table", content="| region | p95 |", caption="Table 1.")
CHART = ParsedFigure(kind="figure", content="a bar chart of revenue by quarter")


class StubVision:
    model = "gemini-3.6-flash"

    def __init__(self, figures: list[ParsedFigure], fail: bool = False) -> None:
        self.calls = 0
        self._figures = figures
        self._fail = fail

    async def parse_page(self, png: bytes) -> list[ParsedFigure]:
        self.calls += 1
        if self._fail:
            raise VisionError("stub provider is down")
        return list(self._figures)
