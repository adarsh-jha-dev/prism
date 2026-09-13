"""`python -m prism.eval ingest` and `python -m prism.eval recall`.

Both are dev entry points and both talk to the local stack. Neither can reach a
paid provider: retrieval embeds on the `ollama` lane and nothing here generates.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from prism.config import get_settings
from prism.db import get_engine
from prism.embeddings import EmbeddingError, get_embedding_provider
from prism.eval.golden import GoldenSetError, load_corpus, load_golden_set
from prism.eval.ingest import CorpusError, ingest_corpus
from prism.eval.report import build_report, render_json, render_text
from prism.eval.runner import (
    CollectionNotResolvedError,
    collection_stats,
    resolve_collection,
    run_golden_set,
)

DEFAULT_TENANT = "eval"
DEFAULT_KS = (1, 3, 5, 10)


def _parser() -> argparse.ArgumentParser:
    # Shared options hang off each subcommand, not off the top-level parser:
    # defined in both places argparse lets the subparser's default overwrite a
    # value already given, and `eval ingest --golden x` is the order anyone types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=Path("eval/corpus.yaml"))
    common.add_argument("--golden", type=Path, default=Path("eval/golden.yaml"))
    common.add_argument("--corpus-dir", type=Path, default=Path("eval/corpus"))
    common.add_argument("--tenant", default=DEFAULT_TENANT)

    parser = argparse.ArgumentParser(prog="python -m prism.eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "ingest", parents=[common], help="ingest the corpus into the golden set's collection"
    )

    recall = sub.add_parser(
        "recall", parents=[common], help="run the golden set and report recall@k"
    )
    recall.add_argument("-k", "--at", type=int, nargs="+", default=list(DEFAULT_KS), dest="ks")
    recall.add_argument("--json", type=Path, default=None, help="also write the run as JSON")
    recall.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RECALL",
        help="exit non-zero if recall at the largest k falls below this",
    )
    return parser


async def _ingest(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.manifest)
    golden = load_golden_set(args.golden, corpus)
    outcome = await ingest_corpus(
        corpus,
        corpus_dir=args.corpus_dir,
        collection=golden.collection,
        tenant=args.tenant,
        provider=get_embedding_provider(),
    )

    print(f"collection {golden.collection} ({outcome.collection_id})")
    for filename in outcome.ingested:
        print(f"  ingested  {filename}")
    for filename in outcome.skipped:
        print(f"  present   {filename}")
    print(f"{outcome.pages} pages, {outcome.chunks} chunks ingested this run")
    return 0


async def _recall(args: argparse.Namespace) -> int:
    ks = sorted({k for k in args.ks if k > 0})
    if not ks:
        print("at least one positive k is required", file=sys.stderr)
        return 2

    corpus = load_corpus(args.manifest)
    golden = load_golden_set(args.golden, corpus)
    if not golden.answerable:
        print("the golden set has no answerable questions to measure", file=sys.stderr)
        return 2

    settings = get_settings()
    ref = await resolve_collection(golden.collection, tenant=args.tenant)
    results = await run_golden_set(golden, collection=ref, k=max(ks), settings=settings)
    report = build_report(
        golden=golden,
        results=results,
        stats=await collection_stats(ref.collection_id),
        settings=settings,
        ks=ks,
    )

    print(render_text(report))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(render_json(report) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.fail_under is not None:
        achieved = report.answerable[-1].recall
        if achieved < args.fail_under:
            print(
                f"\nrecall@{ks[-1]} {achieved:.3f} is below --fail-under {args.fail_under:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


async def _run(args: argparse.Namespace) -> int:
    try:
        if args.command == "ingest":
            return await _ingest(args)
        return await _recall(args)
    except (GoldenSetError, CorpusError, CollectionNotResolvedError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except EmbeddingError as exc:
        print(f"{exc}\nIs Ollama running with the embedding model pulled?", file=sys.stderr)
        return 2
    finally:
        await get_engine().dispose()


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
