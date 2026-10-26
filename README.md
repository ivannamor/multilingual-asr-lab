# Multilingual ASR Lab

[![CI](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml)

A reproducible framework to evaluate, compare and improve automatic speech recognition (ASR)
models on Italian, German and English, with a focus on statistically sound evaluation and
data-centric optimization.

> **Status:** early development. The evaluation pipeline (FLEURS ingestion, Whisper inference,
> WER/CER with versioned normalizers and bootstrap confidence intervals) works end to end and has
> produced the first baseline; the experiments comparing models and settings are next.

Design decisions and their rationale are recorded in [docs/design.md](docs/design.md).

## Results

| Model | Dataset | Utterances / words | Normalizer | WER [95% CI] | CER [95% CI] | RTF |
|---|---|---|---|---|---|---|
| Whisper-small (faster-whisper, int8, beam 5) | FLEURS it test | 865 / 21,255 | `it-v1` | **7.46%** [6.81, 8.08] | 2.11% [1.87, 2.37] | 0.11 |

Corpus-level rates. Confidence intervals: percentile bootstrap, 1,000 resamples, resampling the
346 distinct FLEURS sentences (each is read by two or three speakers, so readings of the same
sentence are correlated; resampling single utterances would give an interval about 18% narrower).
RTF = processing time / audio duration on a 12-core ARM64 laptop CPU, 4 threads, no GPU.
Error analysis (`asr-lab errors`): most errors are substitutions (1,223 vs 169 deletions and 193
insertions), spread over 885 distinct word pairs. Part of them are normalization artefacts rather
than recognition errors (abbreviations and numbers such as "km" vs "chilometri", elided forms such
as "delle" vs "dell'"), which experiment E3 will quantify; others are rare proper nouns ("Geospiza"
recognized as "joe s pizza"). Whisper's known subtitle hallucination appears at the end of 5 of the
865 utterances ("Sottotitoli e revisione a cura di QTSS", text learned from subtitled videos).
To reproduce: [Reproducing the baseline](#reproducing-the-baseline).

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

## Command line

### Reproducing the baseline

```bash
uv sync
uv run python -m asr_lab.data.ingest --lang it --split test --output data/fleurs_it
uv run asr-lab transcribe --manifest data/fleurs_it/test.jsonl --output runs/whisper-small-int8-fleurs-it
uv run asr-lab score --run runs/whisper-small-int8-fleurs-it
uv run asr-lab errors --run runs/whisper-small-int8-fleurs-it   # most frequent errors
```

Expected output of the last command (transcription takes about 25 minutes on a laptop CPU;
timings vary with the machine):

```
faster-whisper/small-int8 on data/fleurs_it/test.jsonl
865 utterances, 21255 words, 128744 characters; normalizer it-v1
WER 7.46%  95% CI [6.81%, 8.08%]  (resampling sentence_id, n=346)
CER 2.11%  95% CI [1.87%, 2.37%]  (resampling sentence_id, n=346)
```

### Run directories

A run directory holds:
- `predictions.jsonl`: one raw (not normalized) hypothesis per utterance;
- `run.json`: model name and decoding settings, manifest SHA-256, git commit, environment, audio
  duration, processing time and real-time factor;
- `scores_<normalizer>.json`: one file per scoring setting, e.g. `scores_it-v1.json`, so scoring
  the same predictions with another normalizer never overwrites a result.
- `errors_<normalizer>.json`: most frequent substitutions, deletions and insertions, and the
  utterances with the most errors (written by `asr-lab errors`, which also prints aligned
  reference/hypothesis pairs in the style of NIST sclite).

Runs are resumable: running the same `transcribe` command again skips the utterances already
transcribed, and running different settings in the same directory is an error. Run directories
are not stored in git.

`asr-lab score` normalizes references and hypotheses with each utterance's language normalizer
(or `--normalizer-lang`), computes corpus WER and CER, and bootstraps confidence intervals for
every meaningful resampling unit: utterances, readings of the same sentence
(`metadata.sentence_id`) and speakers with several utterances. The widest interval is reported,
because correlated utterances make narrower intervals over-optimistic. The scores file also
records the number of utterances and reference words, S/D/I counts, the model and its settings,
and the manifest hash.

### Options

Every command lists its options with `--help`, e.g. `uv run asr-lab transcribe --help`.

| Command | Option | Default | Meaning |
|---|---|---|---|
| `python -m asr_lab.data.ingest` | `--lang` | (required) | `it`, `de` or `en` |
| | `--split` | `test` | `train`, `dev` or `test` |
| | `--output` | (required) | dataset folder (manifest + audio) |
| | `--max-utterances`, `--seed` | all, `0` | seeded random subset |
| `asr-lab transcribe` | `--manifest`, `--output` | (required) | input manifest, run directory |
| | `--model-size` | `small` | any faster-whisper model, e.g. `large-v3-turbo` |
| | `--compute-type` | `int8` | weight precision, e.g. `float32` |
| | `--beam-size` | `5` | `1` = greedy decoding |
| | `--cpu-threads` | `4` | more threads were slower on the development machine |
| | `--batch-size` | `8` | utterances written to disk at a time |
| | `--limit` | all | first N utterances of the manifest |
| `asr-lab score` | `--run` | (required) | run directory |
| | `--manifest` | from `run.json` | override the manifest |
| | `--normalizer-lang` | per utterance | one language's normalizer for every utterance |
| | `--n-resamples`, `--seed` | `1000`, `0` | bootstrap settings (non-default values appear in the file name) |
| `asr-lab errors` | `--run` | (required) | run directory |
| | `--manifest`, `--normalizer-lang` | as `score` | as `score` |
| | `--top`, `--worst` | `20`, `5` | frequent errors listed, worst utterances shown |

## Python API

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

### Transcribing with Whisper

From Python, models share one interface (`ASRModel.transcribe`). The Whisper adapter uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper); weights are downloaded on first use.
This example is not executed by the test suite because it needs the model and the FLEURS data.

```python
from pathlib import Path

from asr_lab.data.manifest import read_manifest
from asr_lab.inference.base import load_sample
from asr_lab.inference.faster_whisper import FasterWhisperModel

manifest = Path("data/fleurs_it/test.jsonl")
samples = [load_sample(entry, manifest) for entry in read_manifest(manifest)[:3]]
model = FasterWhisperModel("small", compute_type="int8", beam_size=5, cpu_threads=4)
for hypothesis in model.transcribe(samples):
    print(hypothesis.utterance_id, hypothesis.text)
print(model.name, model.config)
```

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

All Python examples with a `>>>` prompt are executed by the test suite, so they always match
the code.
