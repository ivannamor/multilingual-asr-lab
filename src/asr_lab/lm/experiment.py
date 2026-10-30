"""The steps of experiment E4, from the spec: text and models, log probabilities, grid, test."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from asr_lab.inference.transformers_models import Wav2Vec2CTCModel
from asr_lab.lm.corpus import (
    build_language_model,
    load_alphabet,
    load_corpora,
    load_spec,
    prepare_text,
    save_report,
)
from asr_lab.lm.logprobs import export_log_probs
from asr_lab.lm.tune import decode_test_set, lm_name, lm_paths, tune_test_set
from asr_lab.provenance import utc_now

SPLITS = ("dev", "test")


def checkpoints(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The CTC checkpoint of each language, pinned in the spec of E1b."""
    models = tomllib.loads(Path(spec["acoustic_models"]).read_text(encoding="utf-8"))
    return models["models"]["ctc"]


def logprobs_path(folder: Path, test_set: dict[str, Any], split: str) -> Path:
    """Log probabilities of one split of a test set.

    >>> logprobs_path(Path("lp"), {"name": "fleurs-it"}, "dev").as_posix()
    'lp/fleurs-it-dev.npz'
    """
    return folder / f"{test_set['name']}-{split}.npz"


def language_models(spec: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Every (corpus, order, name) of the experiment."""
    return [
        (corpus["name"], order, lm_name(corpus["name"], order))
        for corpus in spec["corpora"]
        for order in spec["text"]["orders"]
    ]


def run_lm_prepare(spec_path: Path, kenlm_bin: Path, models_dir: Path, report_path: Path) -> Path:
    """Write the text of each source and language, and build its language models."""
    spec = load_spec(spec_path)
    ctc = checkpoints(spec)
    report = []
    for lang in sorted({test_set["lang"] for test_set in spec["test_sets"]}):
        alphabet = load_alphabet(ctc[lang]["id"], ctc[lang]["revision"])
        manifests = [
            Path(test_set[split])
            for test_set in spec["test_sets"]
            if test_set["lang"] == lang
            for split in SPLITS
        ]
        for corpus in load_corpora(spec):
            text, _ = lm_paths(models_dir, corpus.name, lang, 0)
            description = prepare_text(
                corpus,
                lang,
                alphabet,
                manifests,
                text,
                n_words=spec["text"]["n_words"],
                seed=spec["text"]["seed"],
            )
            description["models"] = [
                build_language_model(
                    text,
                    lm_paths(models_dir, corpus.name, lang, order)[1],
                    order=order,
                    kenlm_bin=kenlm_bin,
                )
                for order in spec["text"]["orders"]
            ]
            report.append(description)
    save_report(
        {"created": utc_now(), "kenlm_commit": spec["text"]["kenlm_commit"], "text": report},
        report_path,
    )
    return report_path


def run_lm_logprobs(spec_path: Path, output: Path, device: str) -> list[Path]:
    """Save the log probabilities of the dev and test sets, one checkpoint per language."""
    spec = load_spec(spec_path)
    ctc = checkpoints(spec)
    written = []
    for lang in sorted({test_set["lang"] for test_set in spec["test_sets"]}):
        model = Wav2Vec2CTCModel(ctc[lang]["id"], revision=ctc[lang]["revision"], device=device)
        for test_set in spec["test_sets"]:
            if test_set["lang"] == lang:
                for split in SPLITS:
                    path = logprobs_path(output, test_set, split)
                    export_log_probs(model, Path(test_set[split]), path)
                    written.append(path)
    return written


def run_lm_tune(
    spec_path: Path, models_dir: Path, logprobs_dir: Path, output: Path, workers: int
) -> Path:
    """The grid of every language model on the dev set of every test set."""
    spec = load_spec(spec_path)
    results = {}
    for test_set in spec["test_sets"]:
        results[test_set["name"]] = {}
        for corpus, order, name in language_models(spec):
            _, lm_path = lm_paths(models_dir, corpus, test_set["lang"], order)
            logprobs = logprobs_path(logprobs_dir, test_set, "dev")
            results[test_set["name"]][name] = tune_test_set(
                spec, test_set, lm_path, logprobs, workers=workers
            )
            selected = results[test_set["name"]][name]["selected"]
            print(f"{test_set['name']} {name}: {selected}", flush=True)
    save_report({"created": utc_now(), "decoding": spec["decoding"], "test_sets": results}, output)
    return output


def run_lm_decode(
    spec_path: Path, models_dir: Path, logprobs_dir: Path, selection_path: Path, runs_dir: Path
) -> list[Path]:
    """Decode each test set once with every language model, with the weights chosen on dev."""
    spec = load_spec(spec_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["test_sets"]
    written = []
    for test_set in spec["test_sets"]:
        for corpus, order, name in language_models(spec):
            _, lm_path = lm_paths(models_dir, corpus, test_set["lang"], order)
            written.append(
                decode_test_set(
                    spec,
                    test_set,
                    name,
                    lm_path,
                    logprobs_path(logprobs_dir, test_set, "test"),
                    selection[test_set["name"]][name]["selected"],
                    runs_dir / f"e4-{name}-{test_set['name']}",
                )
            )
    return written
