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

## Usage

Align a reference transcript with a model hypothesis and count the errors
(open a Python console with `uv run python`):

```python
>>> from asr_lab.evaluation.metrics import align, count_errors
>>> pairs = align("il gatto dorme".split(), "il cane".split())
>>> for p in pairs:
...     print(f"{p.op.value:<13} {p.ref!s:<6} {p.hyp}")
match         il     il
deletion      gatto  None
substitution  dorme  cane

>>> counts = count_errors(pairs)
>>> counts
ErrorCounts(hits=1, substitutions=1, deletions=1, insertions=0)
>>> print(f"WER = {counts.errors / counts.ref_length:.1%}")
WER = 66.7%

```

The same functions work at character level, which is how the character error rate (CER) is
computed:

```python
>>> count_errors(align(list("kitten"), list("sitting")))
ErrorCounts(hits=4, substitutions=2, deletions=0, insertions=1)

```

All examples in this README are executed by the test suite, so they always match the code.
