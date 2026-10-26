"""The ``asr-lab`` command line.

Usage::

    uv run asr-lab transcribe --manifest data/fleurs_it/test.jsonl --output runs/whisper-small-it
    uv run asr-lab score --run runs/whisper-small-it
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from asr_lab.evaluation.normalize import get_normalizer
from asr_lab.evaluation.scoring import SCORES_FILE, format_summary, score_run
from asr_lab.inference.faster_whisper import FasterWhisperModel
from asr_lab.inference.run import transcribe_manifest


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
        "--normalizer-lang",
        default=None,
        help="use one language's normalizer for every utterance (default: each utterance's)",
    )
    score.add_argument("--n-resamples", type=int, default=1000)
    score.add_argument("--seed", type=int, default=0)

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
        normalizer = get_normalizer(args.normalizer_lang) if args.normalizer_lang else None
        scores = score_run(
            args.run,
            args.manifest,
            normalizer=normalizer,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        print(format_summary(scores))
        print(f"Scores: {args.run / SCORES_FILE}")


if __name__ == "__main__":
    main()
