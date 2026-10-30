"""The text of the language models, cleaned to what a CTC model can write, and the models.

An n-gram language model only helps with words a CTC model can spell, so its text must use the same
alphabet: lowercase, no digits, no punctuation. A sentence with digits or foreign letters is
dropped rather than repaired, since a changed sentence would teach the model word sequences nobody
wrote.

**No leakage.** A sentence that appears in a dev or test manifest, after the same cleaning, is
removed from the text, and how many there were is reported. This catches identical sentences only,
which is why the sources are chosen for not sharing their origin with the test sets: FLEURS comes
from Wikimedia projects, and a language model built on Wikipedia would contain its neighbours.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import tomllib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from asr_lab.data.manifest import read_manifest
from asr_lab.provenance import file_sha256

_PUNCTUATION = re.compile(r"[^\w\s']|_", re.UNICODE)
_DIGIT = re.compile(r"\d")
_SPACES = re.compile(r"\s+")
_APOSTROPHES = str.maketrans(dict.fromkeys(map(chr, (0x2019, 0x2018, 0x02BC, 0x60)), "'"))


@dataclass(frozen=True)
class Corpus:
    """One text source: for each language, the parquet files and the field holding the sentence."""

    name: str
    repo: str
    revision: str
    files: dict[str, dict[str, str]]  # language -> {"config": folder, "field": "translation.it"}


def load_corpora(spec: dict[str, Any]) -> list[Corpus]:
    """The ``[[corpora]]`` tables of a language-model spec."""
    return [
        Corpus(item["name"], item["repo"], item["revision"], item["files"])
        for item in spec["corpora"]
    ]


def clean_sentence(text: str, alphabet: frozenset[str]) -> str | None:
    """A sentence in the alphabet of the CTC model, or ``None`` if it cannot be written in it.

    >>> alphabet = frozenset("abcdefghijklmnopqrstuvwxyz'")
    >>> clean_sentence("L'ha detto ieri!", alphabet)
    "l'ha detto ieri"
    >>> clean_sentence("Nel 1999", alphabet) is None
    True
    """
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES).lower()
    if _DIGIT.search(text):
        return None
    text = _SPACES.sub(" ", _PUNCTUATION.sub(" ", text)).strip()
    if not text or any(char not in alphabet for char in text.replace(" ", "")):
        return None
    return text


def forbidden_sentences(manifests: Iterable[Path], alphabet: frozenset[str]) -> set[str]:
    """The sentences of dev and test manifests, cleaned, to keep out of a language model."""
    return {
        cleaned
        for manifest in manifests
        for entry in read_manifest(manifest)
        if (cleaned := clean_sentence(entry.text, alphabet))
    }


def select_sentences(
    lines: Iterable[str],
    alphabet: frozenset[str],
    forbidden: set[str],
    *,
    n_words: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Clean the lines, drop the forbidden ones and draw sentences up to ``n_words`` words.

    The budget is in words because sources differ in sentence length: parliamentary sentences are
    about four times longer than subtitles, so the same number of sentences would compare the
    sizes of two language models as well as their domains.

    Returns:
        The sentences, and how many lines each step removed.

    Raises:
        ValueError: if the sentences left have fewer than ``n_words`` words.
    """
    counts = {"read": 0, "not_writable": 0, "in_dev_or_test": 0, "duplicates": 0}
    kept: dict[str, None] = {}  # a dict keeps the order, so that the draw is reproducible
    matches: set[str] = set()
    for line in lines:
        counts["read"] += 1
        cleaned = clean_sentence(line, alphabet)
        if cleaned is None:
            counts["not_writable"] += 1
        elif cleaned in forbidden:
            counts["in_dev_or_test"] += 1
            matches.add(cleaned)
        elif cleaned in kept:
            counts["duplicates"] += 1
        else:
            kept[cleaned] = None
    counts["distinct_in_dev_or_test"] = len(matches)
    counts["available_sentences"] = len(kept)
    counts["available_words"] = sum(sentence.count(" ") + 1 for sentence in kept)
    if counts["available_words"] < n_words:
        raise ValueError(f"only {counts['available_words']} words remain, {n_words} requested")
    drawn = random.Random(seed).sample(list(kept), len(kept))
    sentences, words = [], 0
    for sentence in drawn:
        if words >= n_words:
            break
        sentences.append(sentence)
        words += sentence.count(" ") + 1
    return sentences, counts


def stream_corpus(corpus: Corpus, lang: str) -> Iterator[str]:
    """The sentences of a language in a corpus, read row group by row group from the Hub."""
    from huggingface_hub import HfFileSystem  # network access only when preparing the text

    source = corpus.files[lang]
    column, *keys = source["field"].split(".")
    fs = HfFileSystem()
    pattern = f"datasets/{corpus.repo}@{corpus.revision}/{source['config']}/*.parquet"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise ValueError(f"no parquet file matches {pattern}")
    for path in paths:
        with fs.open(path) as file:
            for batch in pq.ParquetFile(file).iter_batches(columns=[column]):
                for value in batch.column(0).to_pylist():
                    for key in keys:
                        value = value[key]
                    if value:
                        yield value


def prepare_text(
    corpus: Corpus,
    lang: str,
    alphabet: frozenset[str],
    forbidden_manifests: Iterable[Path],
    output: Path,
    *,
    n_words: int,
    seed: int,
) -> dict[str, Any]:
    """Write the text of a language model, one sentence per line, and return its description."""
    forbidden = forbidden_sentences(forbidden_manifests, alphabet)
    sentences, counts = select_sentences(
        stream_corpus(corpus, lang), alphabet, forbidden, n_words=n_words, seed=seed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sentences) + "\n", encoding="utf-8", newline="\n")
    return {
        "corpus": corpus.name,
        "repo": corpus.repo,
        "revision": corpus.revision,
        "lang": lang,
        "seed": seed,
        **counts,
        "sentences": len(sentences),
        "words": sum(sentence.count(" ") + 1 for sentence in sentences),
        "forbidden_sentences": len(forbidden),
    }


def build_language_model(
    text: Path, output: Path, *, order: int, kenlm_bin: Path
) -> dict[str, Any]:
    """Count the n-grams of a text with KenLM and write a binary model and its vocabulary.

    ``kenlm_bin`` is the folder holding KenLM's ``lmplz`` and ``build_binary``, which are compiled
    separately: the Python package only reads models.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    arpa = output.with_suffix(".arpa")
    with text.open("rb") as stdin, arpa.open("wb") as stdout:
        subprocess.run(
            [str(kenlm_bin / "lmplz"), "-o", str(order), "-S", "30%", "-T", str(output.parent)],
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    subprocess.run(
        [str(kenlm_bin / "build_binary"), "-S", "30%", str(arpa), str(output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    arpa.unlink()
    return {"order": order, "text_sha256": file_sha256(text), "bytes": output.stat().st_size}


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write a description of the language models as JSON (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_alphabet(model_id: str, revision: str) -> frozenset[str]:
    """The characters a CTC checkpoint can write, read from its vocabulary at a pinned revision.

    The hyphen is left out: the models rarely write it, and a word with a hyphen would be unknown
    to the language model, which splits it in two like the scoring does.
    """
    from huggingface_hub import hf_hub_download  # network access only when preparing the text

    vocabulary = hf_hub_download(model_id, "vocab.json", revision=revision)
    tokens = json.loads(Path(vocabulary).read_text(encoding="utf-8"))
    return frozenset(token for token in tokens if len(token) == 1 and token not in "|-")


def load_spec(path: Path) -> dict[str, Any]:
    """Read a language-model spec (e.g. ``experiments/E4/lm.toml``)."""
    return tomllib.loads(path.read_text(encoding="utf-8"))
