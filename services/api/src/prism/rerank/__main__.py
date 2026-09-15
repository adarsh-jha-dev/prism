"""`python -m prism.rerank fetch` — download the pinned reranker weights."""

import sys

from prism.config import get_settings
from prism.rerank.base import RerankError
from prism.rerank.weights import fetch_weights, weights_dir


def main() -> int:
    if sys.argv[1:] != ["fetch"]:
        print("usage: python -m prism.rerank fetch", file=sys.stderr)
        return 2
    settings = get_settings()
    try:
        fetched = fetch_weights(settings)
    except RerankError as exc:
        print(f"RerankError: {exc}", file=sys.stderr)
        return 1
    for path in fetched:
        print(f"  fetched  {path}")
    print(f"{weights_dir(settings)} is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
