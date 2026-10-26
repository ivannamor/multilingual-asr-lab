# Multilingual ASR Lab

[![CI](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml)

A reproducible framework to evaluate, compare and improve automatic speech recognition (ASR)
models on Italian, German and English, with a focus on statistically sound evaluation and
data-centric optimization.

> **Status:** early development. The evaluation core (alignment and error counts) is in place;
> datasets, model adapters and experiments are coming next.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and git. uv installs the right Python version
(3.11) automatically.

```bash
git clone https://github.com/ivannamor/multilingual-asr-lab.git
cd multilingual-asr-lab
uv sync                      # create .venv and install locked dependencies
uv run pre-commit install    # optional: run linters before every commit
uv run pytest                # run the test suite (including the examples below)
```

## Data

Datasets are converted into JSONL manifests (one utterance per line) with 16 kHz mono 16-bit
WAV audio. Audio paths are relative to the manifest, so a dataset folder can be moved freely.
Data is downloaded from the Hugging Face Hub and is not stored in git.

```bash
# FLEURS Italian test set (865 utterances, ~660 MB download, cached by huggingface_hub)
uv run python -m asr_lab.data.ingest --lang it --split test --output data/fleurs_it
```

FLEURS has no speaker ids: each utterance gets a placeholder `speaker_id`, and
`metadata["sentence_id"]` groups the utterances that read the same sentence.

## Usage

Open a Python console with `uv run python`.

### Word and character error rate

`wer` and `cer` take a list of reference transcripts and a list of model outputs (hypotheses)
and return **corpus-level** rates: total errors divided by total reference words (or
characters), not the mean of per-utterance rates.

```python
>>> from asr_lab.evaluation.metrics import wer, cer
>>> references = ["il gatto dorme", "oggi piove"]
>>> hypotheses = ["il cane", "oggi piove"]
>>> wer(references, hypotheses)
0.4
>>> round(cer(references, hypotheses), 3)
0.375

```

Here the first utterance has 2 errors out of 3 words and the second has none, so the corpus WER
is 2 / 5 = 0.4. A WER above 1.0 is possible when the model inserts many extra words.

**Empty references.** The error rate of a single utterance with an empty reference (e.g. a
silent clip) is mathematically undefined. `error_rate` returns `0.0` when the model also
outputs nothing and `inf` otherwise, so hallucinations on silence rank as the worst
utterances instead of being hidden. This differs from jiwer, which divides by `max(N, 1)`.
Corpus-level WER and CER are unaffected: inserted words are added to the corpus error count.
For silent clips, a dedicated hallucination metric is more informative than WER.

### Text normalization

References and hypotheses must go through the **same** normalizer before scoring, otherwise
casing, punctuation and typographic variants are counted as recognition errors. Each
normalizer has a versioned name that should be reported with every result.

```python
>>> from asr_lab.evaluation.normalize import get_normalizer
>>> normalize = get_normalizer("it")
>>> normalize.name
'it-v1'
>>> reference = "L'uomo è arrivato."
>>> hypothesis = "l\u2019uomo è arrivato"  # typographic apostrophe
>>> round(wer([reference], [hypothesis]), 3)
0.667
>>> wer([normalize(reference)], [normalize(hypothesis)])
0.0

```

| Normalizer | Steps |
|---|---|
| `none` | no change |
| `basic-v1`, `en-v1` | Unicode NFC, unify apostrophes, lowercase, remove punctuation (apostrophes inside words are kept: "don't"), collapse whitespace |
| `it-v1` | as above, then split elisions: "l'uomo" → "l uomo" |
| `de-v1` | as above, then "ß" → "ss" (umlauts are kept) |

Numbers are not normalized yet: "venti" and "20" still count as different words.

### Confidence intervals and system comparison

A WER measured on a test set is an estimate: a different sample of utterances would give a
different number. `bootstrap_ci` resamples utterances with replacement to get a 95% confidence
interval, and `paired_bootstrap` compares two systems on the same utterances to tell whether
their difference is larger than this sampling noise.

```python
>>> from asr_lab.evaluation.metrics import utterance_counts
>>> from asr_lab.evaluation.stats import bootstrap_ci, paired_bootstrap
>>> references = ["il gatto dorme sul divano"] * 60 + ["oggi piove molto"] * 40
>>> system_a = ["il gatto dorme sul divano"] * 60 + ["oggi piove molta"] * 40
>>> system_b = ["il gatto dorme sul divano"] * 60 + ["oggi piove molto"] * 30 + ["oggi piove molta"] * 10
>>> counts_a = utterance_counts(references, system_a)
>>> counts_b = utterance_counts(references, system_b)
>>> ci = bootstrap_ci(counts_a)
>>> print(f"WER A = {ci.estimate:.1%} [{ci.low:.1%}, {ci.high:.1%}]")
WER A = 9.5% [7.1%, 12.2%]
>>> comparison = paired_bootstrap(counts_a, counts_b)
>>> print(f"B - A = {comparison.delta:+.1%} [{comparison.low:+.1%}, {comparison.high:+.1%}]")
B - A = -7.1% [-9.7%, -4.8%]
>>> comparison.significant
True

```

Utterances of the same speaker are correlated; pass `groups=speaker_ids` to resample whole
speakers instead of single utterances, which gives wider and more honest intervals. Results are
reproducible: the random generator is seeded (`seed=0` by default).

### Inspecting the alignment

```python
>>> from asr_lab.evaluation.metrics import align, count_errors
>>> pairs = align("il gatto dorme".split(), "il cane".split())
>>> for p in pairs:
...     print(f"{p.op.value:<13} {p.ref!s:<6} {p.hyp}")
match         il     il
deletion      gatto  None
substitution  dorme  cane

>>> count_errors(pairs)
ErrorCounts(hits=1, substitutions=1, deletions=1, insertions=0)

```

All examples in this README are executed by the test suite, so they always match the code.
