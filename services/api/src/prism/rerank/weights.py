"""Where the reranker weights live, and fetching them at a pinned digest.

The floor is calibrated against one revision and one quantization, so a file
that does not match its digest is refused rather than loaded.
"""

import hashlib
from pathlib import Path

import httpx

from prism.config import Settings
from prism.rerank.base import RerankError

__all__ = ["artifact_paths", "fetch_weights", "weights_dir"]

_PINNED: dict[tuple[str, str], dict[str, str]] = {
    ("6f5ff65298512715a1e669753bc754d2bc8f367b", "int8"): {
        "onnx/model_int8.onnx": "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
        "tokenizer.json": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    },
}

_READ_CHUNK_BYTES = 1024 * 1024


def weights_dir(settings: Settings) -> Path:
    return settings.reranker_dir / settings.reranker_model / settings.reranker_revision


def _pinned(settings: Settings) -> dict[str, str]:
    key = (settings.reranker_revision, settings.reranker_quantization)
    if key not in _PINNED:
        raise RerankError(
            f"no digests pinned for revision {key[0]} at {key[1]} — pin them, "
            "then re-fit rerank_score_floor on the golden set"
        )
    return _PINNED[key]


def artifact_paths(settings: Settings) -> tuple[Path, Path]:
    """The model and tokenizer files, in that order."""
    root = weights_dir(settings)
    return (
        root / f"onnx/model_{settings.reranker_quantization}.onnx",
        root / "tokenizer.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(settings: Settings) -> list[Path]:
    """Download every pinned artifact not already present. Returns what was fetched."""
    base = f"https://huggingface.co/{settings.reranker_repo}/resolve/{settings.reranker_revision}"
    fetched: list[Path] = []
    for name, expected in _pinned(settings).items():
        target = weights_dir(settings) / name
        if target.is_file() and _sha256(target) == expected:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        with (
            httpx.stream("GET", f"{base}/{name}", follow_redirects=True, timeout=60.0) as response,
            partial.open("wb") as handle,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes(_READ_CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)

        if digest.hexdigest() != expected:
            partial.unlink()
            raise RerankError(f"{name}: sha256 {digest.hexdigest()[:12]}…, pinned {expected[:12]}…")
        partial.replace(target)
        fetched.append(target)
    return fetched
