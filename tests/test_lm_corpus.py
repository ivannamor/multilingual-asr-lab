"""The text of the language models: cleaning, leakage filter, reproducible draw."""

from __future__ import annotations

from pathlib import Path

import pytest

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.lm.corpus import (
    Corpus,
    clean_sentence,
    forbidden_sentences,
    load_corpora,
    prepare_text,
    select_sentences,
)

ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyzàèéìòù'")


def test_clean_sentence_matches_the_ctc_alphabet() -> None:
    assert clean_sentence("L\u2019Unione, oggi!", ALPHABET) == "l'unione oggi"
    assert clean_sentence("  più   città ", ALPHABET) == "più città"
    assert clean_sentence("là-bas", ALPHABET) == "là bas"


@pytest.mark.parametrize("text", ["Nel 1999", "", "   ", "!!!", "Москва e Roma", "naïve"])
def test_sentences_the_model_cannot_write_are_dropped(text: str) -> None:
    assert clean_sentence(text, ALPHABET) is None


def test_select_removes_dev_and_test_sentences_and_counts_them() -> None:
    lines = ["Una frase.", "UNA FRASE", "Frase di test!", "Altra frase", "Nel 1999", "Terza frase"]
    sentences, counts = select_sentences(lines, ALPHABET, {"frase di test"}, n_words=4, seed=0)
    assert "frase di test" not in sentences
    assert counts == {
        "read": 6,
        "not_writable": 1,
        "in_dev_or_test": 1,
        "duplicates": 1,
        "distinct_in_dev_or_test": 1,
        "available_sentences": 3,
        "available_words": 6,
    }


def test_the_draw_is_reproducible_and_depends_on_the_seed() -> None:
    lines = [f"frase {word}" for word in "abcdefghijklmnopqrstuvwxyz"]
    first = select_sentences(lines, ALPHABET, set(), n_words=10, seed=1)[0]
    assert first == select_sentences(lines, ALPHABET, set(), n_words=10, seed=1)[0]
    assert first != select_sentences(lines, ALPHABET, set(), n_words=10, seed=2)[0]


def test_too_few_sentences_is_an_error() -> None:
    with pytest.raises(ValueError, match="only 2"):
        select_sentences(["una frase"], ALPHABET, set(), n_words=3, seed=0)


def test_the_budget_is_in_words_not_in_sentences() -> None:
    lines = ["una frase molto molto lunga davvero", "corta", "altra", "ancora", "basta"]
    sentences, _ = select_sentences(lines, ALPHABET, set(), n_words=6, seed=0)
    assert sum(sentence.count(" ") + 1 for sentence in sentences) >= 6
    assert sum(sentence.count(" ") + 1 for sentence in sentences[:-1]) < 6


def test_forbidden_sentences_are_cleaned_like_the_corpus(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    entry = ManifestEntry("u", "u.wav", 1.0, "Una Frase, di test.", "it", "s", "t")
    write_manifest(manifest, [entry])
    assert forbidden_sentences([manifest], ALPHABET) == {"una frase di test"}


def test_prepare_text_writes_and_describes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "asr_lab.lm.corpus.stream_corpus", lambda corpus, lang: ["Una frase", "Altra frase"]
    )
    manifest = tmp_path / "test.jsonl"
    write_manifest(manifest, [ManifestEntry("u", "u.wav", 1.0, "Altra frase", "it", "s", "t")])
    corpus = Corpus("c", "repo", "rev", {"it": {"config": "x", "field": "y"}})
    output = tmp_path / "lm" / "text.txt"
    report = prepare_text(corpus, "it", ALPHABET, [manifest], output, n_words=2, seed=0)
    assert output.read_text(encoding="utf-8") == "una frase\n"
    assert report["in_dev_or_test"] == 1
    assert report["words"] == 2
    assert report["revision"] == "rev"


def test_load_corpora_reads_the_spec() -> None:
    spec = {"corpora": [{"name": "c", "repo": "r", "revision": "v", "files": {"it": {}}}]}
    assert load_corpora(spec) == [Corpus("c", "r", "v", {"it": {}})]
