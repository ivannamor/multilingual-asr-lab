"""The ``asr-lab`` command line.

Usage::

    uv run asr-lab transcribe --manifest data/fleurs_it/test.jsonl --output runs/whisper-small-it
    uv run asr-lab transcribe --manifest data/fleurs_it/test.jsonl --output runs/parakeet-it \
        --family tdt --model-id nvidia/parakeet-tdt-0.6b-v3 --revision 541d1f9 --device cuda
    uv run asr-lab score --run runs/whisper-small-it
    uv run asr-lab errors --run runs/whisper-small-it
    uv run asr-lab compare --a runs/A --b runs/B
    uv run asr-lab select --spec experiments/E2/sweep.toml --output selection.json
    uv run asr-lab plot-selection --selection selection.json --output pareto.png
    uv run asr-lab plan-sessions --spec experiments/E10/simulation.toml --output sessions.jsonl
    uv run asr-lab render-sessions --manifest sessions.jsonl
    uv run asr-lab usage --run runs/e10 --spec experiments/E10/simulation.toml --output usage.json
    uv run asr-lab ingest-voxpopuli --spec experiments/E6/quality.toml
    uv run asr-lab quality --spec experiments/E6/quality.toml --run runs/e6 --output results/
    uv run asr-lab quality --spec experiments/E6/quality.toml --run runs/e6 --output results/ \
        --export-collapsed --second-run runs/e6-small
    uv run asr-lab select-data --spec experiments/E7/finetune.toml --export-high-wer
    uv run asr-lab select-data --spec experiments/E7/finetune.toml --output selections.json
    uv run asr-lab finetune --spec experiments/E7/finetune.toml --selections selections.json \
        --strategy random --seed 0 --output models/e7/random-s0
    uv run asr-lab finetune-results --spec experiments/E7/finetune.toml \
        --records experiments/E7/results/training --output finetune.json
    uv run asr-lab render-results --spec experiments/E3/tables.toml --document RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from asr_lab.analytics.sessions import load_simulation_spec, plan_sessions, render_sessions
from asr_lab.analytics.usage import analyze_usage, load_user_model, save_usage
from asr_lab.data.manifest import write_manifest
from asr_lab.data.quality import (
    analyze_quality,
    export_collapsed,
    export_high_wer,
    save_features,
    save_report,
    second_opinion,
)
from asr_lab.data.voxpopuli import ingest_voxpopuli, load_data_spec
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
from asr_lab.inference.faster_whisper import DEFAULT_TEMPERATURES, FasterWhisperModel
from asr_lab.inference.run import transcribe_manifest
from asr_lab.inference.transformers_models import FAMILIES
from asr_lab.plots import plot_selection, plot_usage
from asr_lab.reporting import render_results
from asr_lab.sweep import format_selection, load_spec, save_selection, select_configuration
from asr_lab.tracking import log_comparison, log_scores
from asr_lab.training.lora import fine_tune, load_training_config
from asr_lab.training.results import finetune_results, save_results
from asr_lab.training.selection import (
    load_selection_spec,
    save_selections,
    select_data,
    selected_entries,
)

_MLFLOW_HELP = (
    "also log the result to this MLflow experiment (needs `uv sync --extra tracking`; "
    "stored where MLFLOW_TRACKING_URI points, by default mlflow.db)"
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asr-lab", description="Multilingual ASR lab.")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="transcribe a manifest with a model and save the predictions"
    )
    transcribe.add_argument("--manifest", required=True, type=Path)
    transcribe.add_argument("--output", required=True, type=Path, help="run directory")
    transcribe.add_argument(
        "--family",
        default="whisper",
        choices=["whisper", *FAMILIES],
        help="whisper (faster-whisper), ctc (wav2vec2) or tdt (Parakeet); the last two need "
        "`uv sync --extra hf`",
    )
    transcribe.add_argument("--model-size", default="small", help="Whisper checkpoint")
    transcribe.add_argument("--model-id", default=None, help="ctc, tdt: repository on the Hub")
    transcribe.add_argument("--revision", default=None, help="ctc, tdt: commit of the repository")
    transcribe.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    transcribe.add_argument(
        "--compute-type",
        default=None,
        help="Whisper default: int8 on CPU, float16 on CUDA; ctc, tdt: float32 on CPU, float16 "
        "on CUDA",
    )
    transcribe.add_argument("--beam-size", type=int, default=5)
    transcribe.add_argument("--cpu-threads", type=int, default=4)
    decoding = transcribe.add_argument_group("decoding (defaults: faster-whisper's)")
    decoding.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPERATURES),
        help="fallback temperatures; '--temperature 0' disables the fallback",
    )
    decoding.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    decoding.add_argument("--log-prob-threshold", type=float, default=-1.0)
    decoding.add_argument("--no-speech-threshold", type=float, default=0.6)
    decoding.add_argument(
        "--vad", action="store_true", help="remove non-speech audio (Silero VAD) before decoding"
    )
    decoding.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=None,
        help="skip silences longer than this (seconds) around a likely hallucination",
    )
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
    score.add_argument("--mlflow", metavar="EXPERIMENT", default=None, help=_MLFLOW_HELP)

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
    compare.add_argument("--mlflow", metavar="EXPERIMENT", default=None, help=_MLFLOW_HELP)

    select = commands.add_parser(
        "select",
        help="choose a decoding configuration from the runs of a sweep "
        "(fastest not significantly worse than the best)",
    )
    select.add_argument("--spec", required=True, type=Path, help="TOML sweep spec")
    select.add_argument("--split", default="dev")
    select.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    select.add_argument("--seed", type=int, default=DEFAULT_SEED)
    select.add_argument("--output", type=Path, default=None, help="save the result as JSON")

    plot = commands.add_parser(
        "plot-selection", help="plot WER against processing time for the runs of a sweep"
    )
    plot.add_argument("--selection", required=True, type=Path, help="JSON saved by select")
    plot.add_argument("--output", required=True, type=Path, help="PNG file")

    plan_cmd = commands.add_parser(
        "plan-sessions", help="draw simulated usage sessions from a TOML spec (experiment E10)"
    )
    plan_cmd.add_argument("--spec", required=True, type=Path, help="TOML simulation spec")
    plan_cmd.add_argument("--output", required=True, type=Path, help="manifest of the sessions")

    render_cmd = commands.add_parser(
        "render-sessions",
        help="write the audio of planned sessions (noise and telephone channel applied)",
    )
    render_cmd.add_argument("--manifest", required=True, type=Path, help="written by plan-sessions")

    usage = commands.add_parser(
        "usage",
        help="usage analytics of transcribed sessions: errors per segment, priorities, "
        "confidence review, simulated corrections (experiment E10)",
    )
    usage.add_argument("--run", required=True, type=Path, help="run directory of the sessions")
    usage.add_argument(
        "--spec", required=True, type=Path, help="TOML simulation spec with a [users] table"
    )
    usage.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    usage.add_argument("--seed", type=int, default=DEFAULT_SEED)
    usage.add_argument("--output", required=True, type=Path, help="JSON report")
    usage.add_argument("--figure", type=Path, default=None, help="also plot WER against SNR (PNG)")

    ingest_vp = commands.add_parser(
        "ingest-voxpopuli",
        help="download the VoxPopuli row groups of a quality spec and write their manifest (E6)",
    )
    ingest_vp.add_argument("--spec", required=True, type=Path, help="TOML quality spec")

    quality = commands.add_parser(
        "quality",
        help="quality features, filter rules and clusters of training utterances (experiment E6)",
    )
    quality.add_argument("--spec", required=True, type=Path, help="TOML quality spec")
    quality.add_argument(
        "--run",
        type=Path,
        default=None,
        help="transcription of the spec's manifest: adds the model's WER and the clusters",
    )
    quality.add_argument(
        "--output", required=True, type=Path, help="folder for features.jsonl and quality.json"
    )
    quality.add_argument(
        "--export-collapsed",
        action="store_true",
        help="write the manifest of the collapsed high-WER utterances (needs --run)",
    )
    quality.add_argument(
        "--second-run",
        type=Path,
        default=None,
        help="a second model's transcription of the collapsed utterances (needs --run)",
    )

    select_data_cmd = commands.add_parser(
        "select-data",
        help="choose training data with each strategy of a fine-tuning spec (experiment E7)",
    )
    select_data_cmd.add_argument("--spec", required=True, type=Path, help="TOML fine-tuning spec")
    select_data_cmd.add_argument("--output", type=Path, default=None, help="selections JSON")
    select_data_cmd.add_argument(
        "--export-high-wer",
        action="store_true",
        help="only write the manifest of the high-WER utterances, for the second model",
    )

    finetune = commands.add_parser(
        "finetune",
        help="fine-tune Whisper with LoRA on one selection and export it (needs the training "
        "extra and, in practice, a GPU)",
    )
    finetune.add_argument("--spec", required=True, type=Path, help="TOML fine-tuning spec")
    finetune.add_argument(
        "--selections", required=True, type=Path, help="the output of select-data"
    )
    finetune.add_argument(
        "--strategy", required=True, help="a strategy of the spec, or 'base' (no training)"
    )
    finetune.add_argument("--seed", type=int, default=0)
    finetune.add_argument("--output", required=True, type=Path, help="model directory")
    finetune.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    finetune_res = commands.add_parser(
        "finetune-results",
        help="WER, contrasts between strategies and forgetting of the fine-tuned models (E7)",
    )
    finetune_res.add_argument("--spec", required=True, type=Path, help="TOML fine-tuning spec")
    finetune_res.add_argument(
        "--runs", type=Path, default=Path("experiments/runs"), help="folder of the run directories"
    )
    finetune_res.add_argument(
        "--records", required=True, type=Path, help="folder of the training records"
    )
    finetune_res.add_argument("--output", required=True, type=Path, help="results JSON")

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
    if args.command == "compare" and args.mlflow and not args.output:
        parser.error("--mlflow needs --output: MLflow mirrors a saved comparison file")
    if args.command == "transcribe":
        if args.family == "whisper":
            model = FasterWhisperModel(
                args.model_size,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                cpu_threads=args.cpu_threads,
                temperature=args.temperature,
                compression_ratio_threshold=args.compression_ratio_threshold,
                log_prob_threshold=args.log_prob_threshold,
                no_speech_threshold=args.no_speech_threshold,
                vad_filter=args.vad,
                hallucination_silence_threshold=args.hallucination_silence_threshold,
            )
        else:
            if not (args.model_id and args.revision):
                parser.error(f"--family {args.family} needs --model-id and --revision")
            model = FAMILIES[args.family](
                args.model_id,
                revision=args.revision,
                device=args.device,
                dtype=args.compute_type,
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
        if args.mlflow:
            print(f"MLflow run: {log_scores(scores, path, args.mlflow)}")
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
            if args.mlflow:
                print(f"MLflow run: {log_comparison(result, args.output, args.mlflow)}")
    elif args.command == "select":
        selection = select_configuration(
            load_spec(args.spec), args.split, n_resamples=args.n_resamples, seed=args.seed
        )
        print(format_selection(selection))
        if args.output:
            save_selection(selection, args.output)
            print(f"Selection: {args.output}")
    elif args.command == "plot-selection":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        print(f"Figure: {plot_selection(selection, args.output)}")
    elif args.command == "plan-sessions":
        count = write_manifest(args.output, plan_sessions(load_simulation_spec(args.spec)))
        print(f"Sessions: {count} in {args.output}")
    elif args.command == "render-sessions":
        written = render_sessions(args.manifest)
        print(f"Audio: {written} session(s) written next to {args.manifest}")
    elif args.command == "usage":
        report = analyze_usage(
            args.run, load_user_model(args.spec), n_resamples=args.n_resamples, seed=args.seed
        )
        save_usage(report, args.output)
        overall = report["overall"]
        print(
            f"{report['sessions']} sessions, WER {overall['wer']:.2%} "
            f"[{overall['ci_low']:.2%}, {overall['ci_high']:.2%}]"
        )
        print(f"Usage report: {args.output}")
        if args.figure:
            print(f"Figure: {plot_usage(report, args.figure)}")
    elif args.command == "ingest-voxpopuli":
        print(f"Manifest: {ingest_voxpopuli(load_data_spec(args.spec))}")
    elif args.command == "quality":
        if (args.export_collapsed or args.second_run) and args.run is None:
            parser.error("--export-collapsed and --second-run need --run")
        rows, report = analyze_quality(args.spec, args.run)
        if args.export_collapsed:  # first: the second run's manifest
            print(f"Collapsed utterances: {export_collapsed(rows, args.spec)}")
        if args.second_run:
            report["second_opinion"] = second_opinion(args.spec, args.run, args.second_run)
        save_features(rows, args.output / "features.jsonl")
        save_report(report, args.output / "quality.json")
        flagged = next(row for row in report["flags"] if row["rule"] == "any")
        print(
            f"{report['utterances']} utterances, {report['hours']:.2f} h; flagged by a rule: "
            f"{flagged['utterances']} ({flagged['share']:.1%}), {flagged['hours']:.2f} h"
        )
        if report["clusters"]:
            for cluster in report["clusters"]["clusters"]:
                print(
                    f"cluster {cluster['cluster']}: {cluster['utterances']} utterances, "
                    f"{cluster['description']}"
                )
        for model in report.get("second_opinion", {}).get("models", []):
            print(
                f"{model['model']} on the collapsed utterances: WER {model['wer']:.2%}, "
                f"{model['collapsed']} still collapsed"
            )
        print(f"Quality report: {args.output / 'quality.json'}")
    elif args.command == "select-data":
        if args.export_high_wer:
            selection = load_selection_spec(args.spec)
            rows, _ = analyze_quality(selection.quality_spec, selection.first_run)
            print(f"High-WER utterances: {export_high_wer(rows, selection.quality_spec)}")
            return
        if args.output is None:
            parser.error("select-data needs --output (or --export-high-wer)")
        report = select_data(args.spec)
        save_selections(report, args.output)
        for selection in report["selections"]:
            print(
                f"{selection['strategy']}, seed {selection['seed']}: "
                f"{selection['utterances']} utterances, {selection['hours']:.2f} h, "
                f"{selection['speakers']} speakers, {selection['flagged']} flagged"
            )
        print(f"Selections: {args.output}")
    elif args.command == "finetune":
        if args.strategy == "base":
            manifest, entries = Path(), []
        else:
            manifest, entries = selected_entries(args.selections, args.strategy, args.seed)
        model_dir = fine_tune(
            load_training_config(args.spec),
            entries,
            manifest,
            args.output,
            seed=args.seed,
            description={"strategy": args.strategy, "selections": args.selections.as_posix()},
            device=args.device,
        )
        print(f"Model: {model_dir}")
    elif args.command == "finetune-results":
        report = finetune_results(args.spec, args.runs, args.records)
        save_results(report, args.output)
        for subset in report["subsets"]:
            systems = ", ".join(
                f"{name} {result['wer']:.2%}" for name, result in subset["systems"].items()
            )
            print(f"{subset['name']} ({subset['n_utterances']} utterances): {systems}")
        print(f"Fine-tuning results: {args.output}")
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
