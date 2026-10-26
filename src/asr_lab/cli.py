"""The ``asr-lab`` command line.

Usage::

    uv run asr-lab transcribe --manifest data/fleurs_it/test.jsonl --output runs/whisper-small-it
    uv run asr-lab score --run runs/whisper-small-it
    uv run asr-lab errors --run runs/whisper-small-it
    uv run asr-lab compare --a runs/A --b runs/B
    uv run asr-lab render-results --spec experiments/E3/tables.toml --document RESULTS.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from asr_lab.evaluation.comparison import (
    METRICS,
    compare_runs,
    format_comparison,
    save_comparison,
)
from asr_lab.evaluation.error_analysis import analyze_run, format_errors
from asr_lab.evaluation.normalize import NORMALIZER_NAMES, get_normalizer_by_name
from asr_lab.evaluation.scoring import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    format_summary,
    score_run,
)
from asr_lab.inference.faster_whisper import FasterWhisperModel
from asr_lab.inference.run import transcribe_manifest
from asr_lab.reporting import render_results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asr-lab", description="Multilingual ASR lab.")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="transcribe a manifest with Whisper and save the predictions"
    )
    transcribe.add_argument("--manifest", required=True, type=Path)
    transcribe.add_argument("--output", required=True, type=Path, help="run directory")
    transcribe.add_argument("--model-size", default="small")
    transcribe.add_argument("--compute-type", default="int8")
    transcribe.add_argument("--beam-size", type=int, default=5)
    transcribe.add_argument("--cpu-threads", type=int, default=4)
    transcribe.add_argument("--batch-size", type=int, default=8)
    transcribe.add_argument(
        "--limit", type=int, default=None, help="only the first N utterances of the manifest"
    )

    score = commands.add_parser(
        "score", help="compute WER and CER with confidence intervals for a run directory"
    )
    score.add_argument("--run", required=True, type=Path, help="run directory")
    score.add_argument(
        "--manifest", type=Path, default=None, help="default: the manifest recorded in run.json"
    )
    score.add_argument(
        "--normalizer",
        default=None,
        choices=NORMALIZER_NAMES,
        help="one normalizer for every utterance (default: each utterance's language normalizer)",
    )
    score.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    score.add_argument("--seed", type=int, default=DEFAULT_SEED)

    errors = commands.add_parser(
        "errors", help="most frequent errors and worst utterances of a run directory"
    )
    errors.add_argument("--run", required=True, type=Path, help="run directory")
    errors.add_argument(
        "--manifest", type=Path, default=None, help="default: the manifest recorded in run.json"
    )
    errors.add_argument(
        "--normalizer",
        default=None,
        choices=NORMALIZER_NAMES,
        help="one normalizer for every utterance (default: each utterance's language normalizer)",
    )
    errors.add_argument("--top", type=int, default=20, help="how many frequent errors to list")
    errors.add_argument("--worst", type=int, default=5, help="how many worst utterances to show")

    compare = commands.add_parser(
        "compare", help="paired bootstrap comparison of two scorings of the same utterances"
    )
    compare.add_argument("--a", required=True, type=Path, help="run directory of system A")
    compare.add_argument("--b", required=True, type=Path, help="run directory of system B")
    compare.add_argument("--normalizer-a", default=None, choices=NORMALIZER_NAMES)
    compare.add_argument("--normalizer-b", default=None, choices=NORMALIZER_NAMES)
    compare.add_argument("--metric", default="wer", choices=sorted(METRICS))
    compare.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    compare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    compare.add_argument("--output", type=Path, default=None, help="save the result as JSON")

    render = commands.add_parser(
        "render-results", help="regenerate the tables of the results document from JSON results"
    )
    render.add_argument("--spec", required=True, type=Path, action="append", help="TOML spec")
    render.add_argument("--document", type=Path, default=Path("RESULTS.md"))
    render.add_argument(
        "--check", action="store_true", help="only check that the document is up to date"
    )
    render.add_argument(
        "--allow-missing-markers",
        action="store_true",
        help="skip tables the document has no markers for (e.g. the README)",
    )

    args = parser.parse_args(argv)
    if args.command == "transcribe":
        model = FasterWhisperModel(
            args.model_size,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            cpu_threads=args.cpu_threads,
        )
        path = transcribe_manifest(
            model, args.manifest, args.output, batch_size=args.batch_size, limit=args.limit
        )
        print(f"Predictions: {path}")
    elif args.command == "score":
        normalizer = get_normalizer_by_name(args.normalizer) if args.normalizer else None
        scores, path = score_run(
            args.run,
            args.manifest,
            normalizer=normalizer,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        print(format_summary(scores))
        print(f"Scores: {path}")
    elif args.command == "errors":
        normalizer = get_normalizer_by_name(args.normalizer) if args.normalizer else None
        utterances, report, path = analyze_run(
            args.run, args.manifest, normalizer=normalizer, top=args.top, worst=args.worst
        )
        print(format_errors(utterances, report, show=args.worst))
        print(f"Errors: {path}")
    elif args.command == "compare":
        result = compare_runs(
            args.a,
            args.b,
            normalizer_a=get_normalizer_by_name(args.normalizer_a) if args.normalizer_a else None,
            normalizer_b=get_normalizer_by_name(args.normalizer_b) if args.normalizer_b else None,
            metric=args.metric,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        print(format_comparison(result))
        if args.output:
            save_comparison(result, args.output)
            print(f"Comparison: {args.output}")
    elif args.command == "render-results":
        changed = render_results(
            args.spec,
            args.document,
            check=args.check,
            allow_missing=args.allow_missing_markers,
        )
        if args.check and changed:
            print(f"{args.document} is out of date: run without --check to regenerate it")
            sys.exit(1)
        print(f"{args.document}: {'updated' if changed else 'up to date'}")


if __name__ == "__main__":
    main()
