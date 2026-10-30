# Multilingual ASR Lab

[![CI](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ivannamor/multilingual-asr-lab/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivannamor/multilingual-asr-lab/blob/main/notebooks/quickstart.ipynb)

A reproducible framework to evaluate, compare and improve automatic speech recognition (ASR)
models on Italian, German and English, with a focus on statistically sound evaluation and
data-centric optimization.

> **Status:** early development. The evaluation pipeline (FLEURS ingestion, Whisper inference,
> WER/CER with versioned normalizers and bootstrap confidence intervals) works end to end and has
> produced the first results (baseline, decoding settings, text normalization, training-data
> quality, data selection for fine-tuning and usage analytics on simulated traffic); more
> experiments are next.

Design decisions and their rationale are recorded in [docs/design.md](docs/design.md).

## Results

**The full results, method and discussion are in [RESULTS.md](RESULTS.md).** FLEURS test sets
(Italian, German and English; 1.8 to 3.5 hours each), corpus-level rates, 95% bootstrap confidence
intervals resampling the distinct sentences. Large models, float16 on a Colab T4 GPU:

<!-- BEGIN e1-gpu -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Italian · Whisper-small | `it-v1` | 865 / 21,255 | 7.43% [6.80%, 8.09%] | 1,225 / 176 / 178 | 2.07% [1.85%, 2.29%] |
| Italian · Whisper-large-v3-turbo | `it-v1` | 865 / 21,255 | 2.64% [2.23%, 3.07%] | 450 / 52 / 59 | 0.85% [0.67%, 1.04%] |
| Italian · Whisper-large-v3 | `it-v1` | 865 / 21,255 | 2.26% [1.93%, 2.64%] | 387 / 52 / 42 | 0.75% [0.59%, 0.93%] |
| German · Whisper-small | `de-v1` | 862 / 18,703 | 9.13% [8.43%, 9.90%] | 1,303 / 168 / 237 | 2.54% [2.29%, 2.80%] |
| German · Whisper-large-v3-turbo | `de-v1` | 862 / 18,703 | 4.45% [3.91%, 5.01%] | 612 / 89 / 131 | 1.44% [1.20%, 1.69%] |
| German · Whisper-large-v3 | `de-v1` | 862 / 18,703 | 4.02% [3.52%, 4.56%] | 540 / 101 / 111 | 1.44% [1.18%, 1.74%] |
| English · Whisper-small | `en-v1` | 647 / 14,467 | 6.59% [5.93%, 7.27%] | 696 / 104 / 154 | 2.70% [2.39%, 3.03%] |
| English · Whisper-large-v3-turbo | `en-v1` | 647 / 14,467 | 5.00% [4.45%, 5.55%] | 517 / 110 / 97 | 2.13% [1.85%, 2.42%] |
| English · Whisper-large-v3 | `en-v1` | 647 / 14,467 | 4.62% [4.08%, 5.17%] | 488 / 100 / 80 | 2.05% [1.75%, 2.37%] |
<!-- END e1-gpu -->

Whisper-base and Whisper-small, int8 on a laptop CPU:

<!-- BEGIN e1-baseline -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Italian · Whisper-small | `it-v1` | 865 / 21,255 | 7.46% [6.81%, 8.08%] | 1,223 / 169 / 193 | 2.11% [1.87%, 2.37%] |
| Italian · Whisper-base | `it-v1` | 865 / 21,255 | 16.09% [15.13%, 17.12%] | 2,595 / 402 / 422 | 4.16% [3.84%, 4.49%] |
| German · Whisper-small | `de-v1` | 862 / 18,703 | 9.31% [8.58%, 10.11%] | 1,335 / 165 / 242 | 2.58% [2.32%, 2.88%] |
| German · Whisper-base | `de-v1` | 862 / 18,703 | 18.07% [16.91%, 19.31%] | 2,564 / 283 / 532 | 5.28% [4.92%, 5.73%] |
| English · Whisper-small | `en-v1` | 647 / 14,467 | 6.66% [5.98%, 7.36%] | 708 / 101 / 154 | 2.75% [2.43%, 3.08%] |
| English · Whisper-base | `en-v1` | 647 / 14,467 | 10.03% [9.09%, 10.99%] | 1,047 / 140 / 264 | 4.39% [3.95%, 4.82%] |
<!-- END e1-baseline -->

Key findings so far:
- **Whisper-large-v3 reaches 2.26% WER in Italian, 4.02% in German and 4.62% in English**, and the
  language ranking reverses compared with the small models: as recognition errors disappear, a
  growing share of what is left is writing conventions (numbers, British vs American spelling)
  that a simple normalizer counts as errors. large-v3-turbo is about twice as fast and 0.4 points
  worse in every language (experiment E1).
- **The gain from a larger model depends on the language**: Whisper-small removes about half of
  Whisper-base's word errors in Italian and German, but only a third in English (experiment E1).
  The German test set has only male speakers and the Italian one is 96% female, so language and
  gender cannot be separated in these comparisons.
- **Text normalization alone changes the WER of the same Whisper-small Italian transcripts from
  13.87% to 7.15%**: almost half of the raw word errors are casing, punctuation and formatting, not
  recognition errors (experiment E3).
- **A "standard" normalizer can inflate WER**: Whisper's own basic normalizer deletes text in
  parentheses, which FLEURS speakers read aloud, adding 1.87 points of WER (95% CI [1.03, 2.79]).
- **Model rankings are stable, relative gains are not**: Whisper-small beats Whisper-base by
  8.6-9.3 WER points under every normalizer, but its relative error reduction ranges from 40% to
  55% depending on normalization.
- **Decoding settings matter little for WER, voice activity detection matters for
  hallucinations**: in a sweep on the dev sets, beam size, temperature fallback and Whisper's
  hallucination-silence threshold move large-v3-turbo's WER by at most a quarter of a point, while
  voice activity detection removes the invented "Vielen Dank" endings (12 to 0 on the German test
  set) for 6% more processing time (experiment E2). The configuration is chosen on dev by a rule
  written in code before the runs, and the test sets are used once:

![WER against processing time for each decoding configuration, on the dev sets](reports/e2_pareto.png)

- **In simulated usage, competing speakers are the problem** (experiment E10; the traffic and the
  users' corrections are simulated, the noise, telephone degradations and transcripts are real).
  large-v3-turbo goes from 3.56% WER in quiet rooms to 45.47% with babble at 0 dB on a headset,
  where it sometimes transcribes a background voice instead of the speaker. Reviewing the least
  confident 5% of sessions catches 34% of all word errors. The one-page
  [usage summary](reports/e10_summary.md) turns this into investment priorities.

![WER against the signal-to-noise ratio, per language, noise type and channel](reports/e10_wer_vs_snr.png)

- **A high WER against a training transcript is not proof that the transcript is wrong**
  (experiment E6, 10 hours of Italian VoxPopuli). A pre-registered rule, WER of large-v3-turbo
  above 50%, flags 12.4% of the utterances. Examined after the fact, about half of them (163) are the
  model collapsing, writing "Grazie a tutti." for a whole speech: Whisper-small transcribes 159 of
  them, and listening confirms the speech is there. Most of the rest are real label noise, segment
  boundaries shifted into the neighbouring speech. Filtering on one model's WER would discard hard
  but valid training data together with the broken transcripts.
- **Fine-tuning can cost more than it gives, and the data decide how much** (experiment E7).
  Whisper-small fine-tuned with LoRA on 5 hours of that parliamentary speech, chosen four ways with
  three seeds each, never beat the base model in-domain and forgot out of domain: FLEURS WER rose
  from 7.05% to 8.47%-9.14%, and the models started to open encyclopedic sentences with "Signor
  Presidente, onorevoli colleghi". Filtering the suspect transcripts or balancing the speakers
  limited the damage significantly compared with random sampling (0.55 to 0.80 WER points
  in-domain, after Holm's correction).

To reproduce: `experiments/E1/run.sh`, `experiments/E2/run.sh`, `experiments/E3/run.sh`,
`experiments/E6/run.sh`, `experiments/E7/run.sh` and `experiments/E10/run.sh` (see
[Experiments](#experiments)).

## Try it in your browser

The [quickstart notebook](notebooks/quickstart.ipynb) runs the whole pipeline on a few FLEURS
utterances: transcription with Whisper, WER with a confidence interval, the most frequent errors and
the effect of normalization. Open it in **Google Colab** with the badge above (free GPU, nothing to
install), change the settings cell (language, model, beam size, normalizer, number of utterances)
and run all cells. Locally: `uv run --with jupyter jupyter lab notebooks/quickstart.ipynb`.

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

### Experiments

Each experiment has a folder in [`experiments/`](experiments/) with a script that regenerates all
its numbers (`run.sh`, and `run.ps1` for Windows) and a TOML spec of the tables it contributes to
[RESULTS.md](RESULTS.md). The transcripts of the runs used in the results are committed in
`experiments/runs/`, so every score and comparison can be recomputed without running a model:

```bash
bash experiments/E1/run.sh    # baseline: Whisper base to large-v3 on Italian, German, English
bash experiments/E2/run.sh    # decoding settings: selection on dev, test comparisons, figure
bash experiments/E3/run.sh    # text normalization: rescoring, paired comparisons, tables
bash experiments/E6/run.sh    # training-data quality: features, rules, clusters, second opinion
bash experiments/E7/run.sh    # fine-tuning data: second model, four strategies, results
bash experiments/E10/run.sh   # usage analytics on simulated sessions: segments, priorities, review
```

### Run directories

A run directory holds:
- `predictions.jsonl`: one raw (not normalized) hypothesis per utterance;
- `run.json`: model name and decoding settings, manifest SHA-256, git commit, environment, audio
  duration, processing time and real-time factor;
- `scores_<normalizer>.json`: one file per scoring setting, e.g. `scores_it-v1.json`, so scoring
  the same predictions with another normalizer never overwrites a result.
- `errors_<normalizer>.json`: most frequent substitutions, deletions and insertions, the
  utterances ending with an invented closing phrase ("Vielen Dank", "Sottotitoli a cura di QTSS"),
  and the utterances with the most errors (written by `asr-lab errors`, which also prints aligned
  reference/hypothesis pairs in the style of NIST sclite).

Runs are resumable: running the same `transcribe` command again skips the utterances already
transcribed (and leaves a complete run untouched), and running different settings in the same
directory is an error. Rescoring unchanged predictions leaves the scores file as it is, so
rerunning an experiment script does not modify the committed results. Run directories are not
stored in git, except those used in the results (`experiments/runs/`).

`asr-lab score` normalizes references and hypotheses with each utterance's language normalizer
(or the one named by `--normalizer`), computes corpus WER and CER, and bootstraps confidence intervals for
every meaningful resampling unit: utterances, readings of the same sentence
(`metadata.sentence_id`) and speakers with several utterances. The widest interval is reported,
because correlated utterances make narrower intervals over-optimistic. The scores file also
records the number of utterances and reference words, S/D/I counts, the model and its settings,
and the manifest hash.

### Experiment tracking (MLflow)

Scores and comparisons can also be logged to [MLflow](https://mlflow.org), to browse, filter and
plot them in its web interface. The JSON files stay the source of truth (tables are generated
from them), and logging the same file again replaces its MLflow run instead of duplicating it.
MLflow is an optional dependency:

```bash
uv sync --extra tracking --extra plots                      # install the tracking and plots extras
uv run asr-lab score --run runs/whisper-small-it --mlflow E1
uv run asr-lab compare --a runs/A --b runs/B --output cmp.json --mlflow E1
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://127.0.0.1:5000
```

Each scores run records the WER and CER with their confidence intervals, S/D/I counts, the
real-time factor, every model and decoding setting as a parameter, and the git commits of the
transcription and of the scoring; the JSON file is attached as an artifact. Runs are stored where
`MLFLOW_TRACKING_URI` points, by default in `mlflow.db` (not committed).

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
| | `--device` | `cpu` | `cpu` or `cuda` (NVIDIA GPU) |
| | `--compute-type` | `int8` on CPU, `float16` on GPU | weight precision, e.g. `float32` |
| | `--beam-size` | `5` | `1` = greedy decoding |
| | `--cpu-threads` | `4` | more threads were slower on the development machine |
| | `--temperature` | `0 0.2 0.4 0.6 0.8 1` | fallback temperatures, tried in order when an output looks broken; `0` alone disables the fallback |
| | `--compression-ratio-threshold`, `--log-prob-threshold` | `2.4`, `-1.0` | when the fallback is triggered |
| | `--no-speech-threshold` | `0.6` | output nothing for a window that is probably not speech |
| | `--vad` | off | remove non-speech audio (Silero VAD) before decoding |
| | `--hallucination-silence-threshold` | off | skip silences longer than this (seconds) around a likely hallucination; enables word timestamps |
| | `--batch-size` | `8` | utterances written to disk at a time |
| | `--limit` | all | first N utterances of the manifest |
| `asr-lab score` | `--run` | (required) | run directory |
| | `--manifest` | from `run.json` | override the manifest |
| | `--normalizer` | per utterance language | one normalizer by name for every utterance: `none`, `basic-v1`, `whisper-basic`, `it-v1`, `it-v2`, `de-v1`, `en-v1` |
| | `--n-resamples`, `--seed` | `1000`, `0` | bootstrap settings (non-default values appear in the file name) |
| | `--mlflow` | off | also log the scores to this MLflow experiment |
| `asr-lab errors` | `--run` | (required) | run directory |
| | `--manifest`, `--normalizer` | as `score` | as `score` |
| | `--top`, `--worst` | `20`, `5` | frequent errors listed, worst utterances shown |
| `asr-lab compare` | `--a`, `--b` | (required) | run directories of systems A and B (can be the same run) |
| | `--normalizer-a`, `--normalizer-b` | per utterance language | normalizer of each side |
| | `--metric` | `wer` | `wer` or `cer` |
| | `--output` | none | save the comparison as JSON |
| | `--mlflow` | off | also log the saved comparison to this MLflow experiment (needs `--output`) |
| `asr-lab select` | `--spec` | (required) | TOML sweep spec, e.g. `experiments/E2/sweep.toml`; picks the fastest configuration not significantly worse than the best one in any language |
| | `--split` | `dev` | split whose runs are compared (never the test set) |
| | `--n-resamples`, `--seed` | `1000`, `0` | bootstrap settings |
| | `--output` | none | save the selection as JSON |
| `asr-lab plot-selection` | `--selection`, `--output` | (required) | JSON saved by `select`, PNG file; WER against processing time, one panel per language (needs `uv sync --extra plots`) |
| `asr-lab plan-sessions` | `--spec`, `--output` | (required) | TOML simulation spec (e.g. `experiments/E10/simulation.toml`), session manifest to write; draws simulated usage sessions from FLEURS recordings |
| `asr-lab render-sessions` | `--manifest` | (required) | writes each session's audio next to the manifest: background noise at its SNR, then its channel (headset or telephone); resumable |
| `asr-lab usage` | `--run`, `--spec`, `--output` | (required) | run directory of transcribed sessions, simulation spec with a `[users]` table, JSON report: WER per segment, priorities, confidence review, background-voice transcripts, simulated corrections |
| | `--figure` | none | also plot WER against SNR per language (PNG, needs `uv sync --extra plots`) |
| | `--n-resamples`, `--seed` | `1000`, `0` | bootstrap settings |
| `asr-lab ingest-voxpopuli` | `--spec` | (required) | TOML quality spec (e.g. `experiments/E6/quality.toml`); downloads its row groups of VoxPopuli at a pinned revision and writes their manifest |
| `asr-lab quality` | `--spec`, `--output` | (required) | quality spec, folder for `features.jsonl` and `quality.json`: per-utterance features (duration, speaking rate, estimated SNR, clipping, duplicates) and the spec's filter rules |
| | `--run` | none | transcription of the spec's manifest by a strong model: adds its WER against each transcript and clusters the utterances |
| | `--export-collapsed` | off | write the manifest of the high-WER utterances where the model wrote almost nothing (`<data>/collapsed.jsonl`), for a second model |
| | `--second-run` | none | a second model's transcription of those utterances: compares both models on them |
| `asr-lab select-data` | `--spec` | (required) | TOML fine-tuning spec (e.g. `experiments/E7/finetune.toml`): its strategies draw the same hours of training data from the E6 pool |
| | `--output` | none | selections JSON: per strategy and seed, the utterance ids and what they contain |
| | `--export-high-wer` | off | only write the manifest of the pool's high-WER utterances (`<data>/high_wer.jsonl`), for the second model of the two-source filter |
| `asr-lab finetune` | `--spec`, `--selections`, `--strategy`, `--output` | (required) | fine-tune Whisper with LoRA on one selection (`base`: no training), merge and convert it to CTranslate2 in `<output>/ct2`, with a training record (needs `uv sync --extra training` and, in practice, a GPU) |
| | `--seed` | `0` | which selection, and the seed of the training |
| | `--device` | `cuda` | `cpu` or `cuda` |
| `asr-lab finetune-results` | `--spec`, `--records`, `--output` | (required) | fine-tuning spec, folder of the training records, results JSON: WER of the base model and of every strategy (seeds pooled) on each test set and on the test speakers absent from the pool, the spec's contrasts and the comparisons with the base model (paired bootstrap, Holm), errors at the segment edges |
| | `--runs` | `experiments/runs` | folder of the run directories (`e7-<model>-<test set>`) |
| `asr-lab render-results` | `--spec` | (required, repeatable) | TOML spec of generated tables, e.g. `experiments/E3/tables.toml` |
| | `--document` | `RESULTS.md` | document whose tables are regenerated |
| | `--check` | off | only check that the document is up to date |
| | `--allow-missing-markers` | off | skip tables the document does not show (README) |

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
| `it-v2` | `it-v1` plus numbers, symbols and unambiguous units in words: "2.000" → "duemila", "3,5" → "tre virgola cinque", "10:30" → "dieci e trenta", "$5" → "cinque dollari", "km" → "chilometri", "vent'anni" → "venti anni" |
| `de-v1` | as above, then "ß" → "ss" (umlauts are kept) |
| `whisper-basic` | reimplementation of OpenAI Whisper's `BasicTextNormalizer`: lowercase, remove text in brackets and parentheses, replace every mark, symbol and punctuation character (apostrophes included) with a space |

The default for Italian is `it-v1`, which does not normalize numbers ("venti" and "20" count as
different words); `it-v2` does, and is available by name while experiment E3 measures its effect.

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
