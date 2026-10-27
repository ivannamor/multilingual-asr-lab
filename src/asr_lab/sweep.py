"""Decoding sweeps: transcribe a split with several decoding configurations, then
choose one with a rule fixed in advance.

A sweep is described by a TOML spec (e.g. ``experiments/E2/sweep.toml``): one
Whisper checkpoint, the languages and their normalizers, path templates for
manifests and run directories, and the configurations, each listing only the
decoding options that differ from faster-whisper's defaults.

The selection rule, "the fastest configuration not significantly worse than the
best one":

1. in each language, the *best* configuration has the lowest corpus WER;
2. a configuration is *eligible* if, in every language, a paired bootstrap
   does not find it significantly worse than that language's best;
3. the eligible configuration with the shortest total processing time wins.
   If none is eligible, the first configuration of the spec (the baseline)
   is kept: no change is safe in every language.

One configuration is chosen for all languages. The rule reads only the split it
is given (the dev set), so the test set cannot influence the choice.
"""

from __future__ import annotations

import inspect
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr_lab.evaluation.comparison import compare_runs
from asr_lab.evaluation.metrics import ErrorCounts, error_rate, utterance_counts, word_tokens
from asr_lab.evaluation.normalize import Normalizer, get_normalizer_by_name
from asr_lab.evaluation.scoring import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    load_run,
    normalized_pairs,
)
from asr_lab.inference.base import SAMPLE_RATE, AudioSample
from asr_lab.inference.faster_whisper import FasterWhisperModel
from asr_lab.inference.run import RUN_FILE, transcribe_manifest

RULE = (
    "fastest configuration (total processing time) that a paired bootstrap does not find "
    "significantly worse than the best configuration of any language; the first one if none"
)
# Constructor arguments of the adapter that are not decoding options.
_NOT_DECODING = {"self", "model_size", "device", "compute_type", "cpu_threads", "model"}
DECODING_OPTIONS = frozenset(inspect.signature(FasterWhisperModel).parameters) - _NOT_DECODING


@dataclass(frozen=True)
class DecodingConfig:
    """A named set of decoding options (only those that differ from the defaults)."""

    name: str
    options: dict[str, Any]
    selectable: bool = True  # False for a repeat of another configuration


@dataclass(frozen=True)
class SweepSpec:
    model_size: str
    languages: tuple[str, ...]
    normalizers: dict[str, str]  # language -> normalizer name
    manifest_template: str  # with {lang} and {split}
    run_dir_template: str  # with {config}, {lang} and {split}
    configs: tuple[DecodingConfig, ...]

    def manifest(self, lang: str, split: str) -> Path:
        return Path(self.manifest_template.format(lang=lang, split=split))

    def run_dir(self, config: str, lang: str, split: str) -> Path:
        return Path(self.run_dir_template.format(config=config, lang=lang, split=split))

    def config(self, name: str) -> DecodingConfig:
        for config in self.configs:
            if config.name == name:
                return config
        raise ValueError(f"No configuration named {name!r}")

    def normalizer(self, lang: str) -> Normalizer:
        return get_normalizer_by_name(self.normalizers[lang])


def load_spec(path: Path) -> SweepSpec:
    """Read and check a sweep spec.

    Raises:
        ValueError: on duplicate names, unknown options or normalizers, a language
            without a normalizer, or no selectable configuration.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    configs = tuple(
        DecodingConfig(item["name"], dict(item.get("options", {})), item.get("selectable", True))
        for item in data["config"]
    )
    names = [config.name for config in configs]
    if len(set(names)) < len(names):
        raise ValueError(f"{path}: configuration names must be unique, got {names}")
    for config in configs:
        unknown = sorted(config.options.keys() - DECODING_OPTIONS)
        if unknown:
            raise ValueError(f"{path}: unknown decoding options {unknown} in {config.name!r}")
    if not any(config.selectable for config in configs):
        raise ValueError(f"{path}: no selectable configuration")
    languages = tuple(data["languages"])
    normalizers = dict(data["normalizers"])
    missing = [lang for lang in languages if lang not in normalizers]
    if missing:
        raise ValueError(f"{path}: no normalizer for {missing}")
    for name in normalizers.values():
        get_normalizer_by_name(name)  # raises if unknown
    return SweepSpec(
        model_size=data["model_size"],
        languages=languages,
        normalizers=normalizers,
        manifest_template=data["manifest"],
        run_dir_template=data["run_dir"],
        configs=configs,
    )


def run_sweep(
    spec: SweepSpec,
    split: str,
    *,
    configs: Sequence[str] | None = None,
    device: str = "cuda",
    limit: int | None = None,
    model: FasterWhisperModel | None = None,
    progress: bool = True,
) -> list[Path]:
    """Transcribe the split of every language with each configuration (all by default).

    The checkpoint is loaded once and shared by all configurations, which run one
    after the other in the order given. Complete runs are skipped and interrupted
    ones resume (see ``transcribe_manifest``).

    Args:
        model: the loaded checkpoint to use (by default ``spec.model_size`` on
            ``device``); tests pass one with fake weights.

    Returns:
        The run directories, in the order they were transcribed.
    """
    model = model or FasterWhisperModel(spec.model_size, device=device)
    _warm_up(model, spec.languages[0])
    run_dirs = []
    for name in configs or [config.name for config in spec.configs]:
        adapter = model.with_decoding(**spec.config(name).options)
        for lang in spec.languages:
            run_dir = spec.run_dir(name, lang, split)
            if progress:
                print(run_dir)
            transcribe_manifest(
                adapter, spec.manifest(lang, split), run_dir, limit=limit, progress=progress
            )
            run_dirs.append(run_dir)
    return run_dirs


def _warm_up(model: FasterWhisperModel, lang: str) -> None:
    """Transcribe one second of silence, so that the first timed run does not pay for
    initialising the device (CUDA kernels, memory pools)."""
    model.transcribe([AudioSample("warm-up", np.zeros(SAMPLE_RATE, dtype=np.float32), lang)])


def select_configuration(
    spec: SweepSpec,
    split: str = "dev",
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Apply the selection rule (see the module docstring) to the runs of a split.

    Returns:
        Every configuration's WER, speed and comparisons with the best ones, the
        best configuration of each language and the selected one.

    Raises:
        ValueError: if a run is missing, incomplete or was transcribed with other
            settings than its configuration.
    """
    infos = {
        config.name: {lang: _checked_run(spec, config, lang, split) for lang in spec.languages}
        for config in spec.configs
    }
    wer = {
        config.name: {
            lang: _corpus_wer(spec.run_dir(config.name, lang, split), spec.normalizer(lang))
            for lang in spec.languages
        }
        for config in spec.configs
    }
    selectable = [config.name for config in spec.configs if config.selectable]
    # Ties go to the configuration listed first.
    best = {lang: min(selectable, key=lambda name: wer[name][lang]) for lang in spec.languages}

    results = {}
    for config in spec.configs:
        processing = sum(info["processing_seconds"] for info in infos[config.name].values())
        audio = sum(info["audio_seconds"] for info in infos[config.name].values())
        result: dict[str, Any] = {
            "options": config.options,
            "selectable": config.selectable,
            "wer": wer[config.name],
            "processing_seconds": processing,
            "audio_seconds": audio,
            "rtf": processing / audio,
        }
        if config.selectable:
            result["vs_best"] = {
                lang: _versus(spec, best[lang], config.name, lang, split, n_resamples, seed)
                for lang in spec.languages
            }
            result["eligible"] = not any(
                versus["significant"] and versus["delta"] > 0
                for versus in result["vs_best"].values()
            )
        results[config.name] = result

    eligible = [name for name in selectable if results[name]["eligible"]]
    selected = (
        min(eligible, key=lambda name: results[name]["processing_seconds"])
        if eligible
        else spec.configs[0].name
    )
    return {
        "rule": RULE,
        "split": split,
        "model_size": spec.model_size,
        "metric": "wer",
        "languages": list(spec.languages),
        "normalizers": spec.normalizers,
        "n_resamples": n_resamples,
        "seed": seed,
        "best": best,
        "selected": selected,
        "none_eligible": not eligible,
        "configs": results,
    }


def _checked_run(spec: SweepSpec, config: DecodingConfig, lang: str, split: str) -> dict[str, Any]:
    """Read a run's run.json, checking that it is complete and used the configuration."""
    run_dir = spec.run_dir(config.name, lang, split)
    path = run_dir / RUN_FILE
    if not path.exists():
        raise ValueError(f"Missing run {run_dir}: transcribe it first")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("finished_at") is None:
        raise ValueError(f"Run {run_dir} is incomplete: run the sweep again to finish it")
    # The settings the configuration stands for; model=object() skips loading weights.
    expected = FasterWhisperModel(spec.model_size, model=object(), **config.options)
    recorded = info.get("model_config") or {}
    keys = ["model_size", "beam_size", *expected.decoding]
    mismatched = [key for key in keys if recorded.get(key) != expected.config[key]]
    if mismatched:
        raise ValueError(f"Run {run_dir} was transcribed with different {mismatched}")
    return info


def _corpus_wer(run_dir: Path, normalizer: Normalizer) -> float:
    run = load_run(run_dir)
    pairs = normalized_pairs(run.entries, run.hypotheses, normalizer)
    counts = utterance_counts(pairs.references, pairs.hypotheses, word_tokens)
    return error_rate(sum(counts, ErrorCounts()))


def _versus(
    spec: SweepSpec, best: str, name: str, lang: str, split: str, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Paired comparison of a configuration (B) with the best one of the language (A)."""
    normalizer = spec.normalizer(lang)
    result = compare_runs(
        spec.run_dir(best, lang, split),
        spec.run_dir(name, lang, split),
        normalizer_a=normalizer,
        normalizer_b=normalizer,
        n_resamples=n_resamples,
        seed=seed,
    )
    ci = result["confidence_intervals"][result["headline_unit"]]
    return {
        "best": best,
        "delta": result["delta"],  # positive when the configuration is worse
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "prob_better": ci["prob_b_better"],
        "significant": result["significant"],
        "unit": result["headline_unit"],
    }


def save_selection(selection: dict[str, Any], path: Path) -> None:
    """Write a selection as JSON (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def format_selection(selection: dict[str, Any]) -> str:
    """A plain-text table: WER per language (* = best), RTF and eligibility."""
    languages = selection["languages"]
    header = f"{'configuration':<18}" + "".join(f"{lang:>9}" for lang in languages)
    lines = [
        f"Selection on {selection['split']} ({selection['model_size']}, WER)",
        header + f"{'RTF':>8}  eligible",
    ]
    for name, result in selection["configs"].items():
        cells = "".join(
            f"{result['wer'][lang]:>8.2%}" + ("*" if selection["best"][lang] == name else " ")
            for lang in languages
        )
        eligible = ("yes" if result["eligible"] else "no") if result["selectable"] else "-"
        lines.append(f"{name:<18}{cells}{result['rtf']:>8.3f}  {eligible}")
    lines.append(f"* best of the language. Rule: {selection['rule']}.")
    note = " (none eligible: baseline kept)" if selection["none_eligible"] else ""
    lines.append(f"Selected: {selection['selected']}{note}")
    return "\n".join(lines)
