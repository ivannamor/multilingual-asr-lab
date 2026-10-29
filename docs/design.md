# Design decisions

Short records of the decisions that shape this project, in the spirit of
[Architecture Decision Records](https://adr.github.io/): the context, the decision, the
alternatives that were rejected and the consequences. New entries are added when a decision is
made; superseded entries are marked, not deleted.

| # | Decision | Area |
|---|---|---|
| 1 | [Every dataset is converted to a JSONL manifest](#1-every-dataset-is-converted-to-a-jsonl-manifest) | data |
| 2 | [Error rates are corpus-level and computed from per-utterance counts](#2-error-rates-are-corpus-level-and-computed-from-per-utterance-counts) | evaluation |
| 3 | [Own alignment implementation, cross-checked against jiwer](#3-own-alignment-implementation-cross-checked-against-jiwer) | evaluation |
| 4 | [The error rate of an empty reference is infinite](#4-the-error-rate-of-an-empty-reference-is-infinite) | evaluation |
| 5 | [Text normalizers are explicit, composable and versioned](#5-text-normalizers-are-explicit-composable-and-versioned) | evaluation |
| 6 | [Confidence intervals by bootstrap over per-utterance counts](#6-confidence-intervals-by-bootstrap-over-per-utterance-counts) | statistics |
| 7 | [FLEURS utterances get a per-utterance speaker placeholder](#7-fleurs-utterances-get-a-per-utterance-speaker-placeholder) | data |
| 8 | [Models share one interface; audio is loaded outside the models](#8-models-share-one-interface-audio-is-loaded-outside-the-models) | inference |
| 9 | [Documentation examples are executed by the test suite](#9-documentation-examples-are-executed-by-the-test-suite) | engineering |
| 10 | [Lesson learned: never seek backwards in a compressed archive](#10-lesson-learned-never-seek-backwards-in-a-compressed-archive) | engineering |
| 11 | [Local Whisper inference with faster-whisper, int8, 4 threads](#11-local-whisper-inference-with-faster-whisper-int8-4-threads) | inference |
| 12 | [One run directory per configuration, resumable, with provenance](#12-one-run-directory-per-configuration-resumable-with-provenance) | engineering |
| 13 | [Report the widest confidence interval across resampling units](#13-report-the-widest-confidence-interval-across-resampling-units) | statistics |
| 14 | [Scores files are named after their settings](#14-scores-files-are-named-after-their-settings) | engineering |
| 15 | [Whisper's basic normalizer is reimplemented, not imported](#15-whispers-basic-normalizer-is-reimplemented-not-imported) | evaluation |
| 16 | [it-v2 writes numbers in words, but is not the default yet](#16-it-v2-writes-numbers-in-words-but-is-not-the-default-yet) | evaluation |
| 17 | [Results tables are generated and checked against committed results](#17-results-tables-are-generated-and-checked-against-committed-results) | engineering |
| 18 | [A notebook for trying the pipeline, not for presenting results](#18-a-notebook-for-trying-the-pipeline-not-for-presenting-results) | engineering |
| 19 | [Large models on a Colab GPU, with a bridge run and predictions brought back as a zip](#19-large-models-on-a-colab-gpu-with-a-bridge-run-and-predictions-brought-back-as-a-zip) | inference |
| 20 | [Every decoding option is explicit and recorded](#20-every-decoding-option-is-explicit-and-recorded) | inference |
| 21 | [MLflow mirrors the JSON results, it does not replace them](#21-mlflow-mirrors-the-json-results-it-does-not-replace-them) | engineering |

---

## 1. Every dataset is converted to a JSONL manifest

**Context.** The project uses several datasets (FLEURS, Common Voice, VoxPopuli, ...), each
with its own layout, and several consumers (inference, evaluation, data quality, clustering).

**Decision.** Each dataset is converted once into a manifest: one JSON object per line with
`utterance_id`, `audio_filepath`, `duration`, `text`, `lang`, `speaker_id`, `source` and an
optional `metadata` dictionary (NeMo-style). Every other module reads only manifests.
- `utterance_id` links predictions to references, instead of relying on line order.
- `audio_filepath` is relative to the manifest and uses forward slashes, so a dataset folder can
  be moved between machines and operating systems unchanged.
- `text` is the original transcript; normalization happens at evaluation time.
- Files are always written as UTF-8 with LF line endings, whatever the operating system.
- Reading and writing validate every line and report errors as `file:line`; unknown fields are
  rejected (they are usually typos), extra information goes into `metadata`.

**Alternatives rejected.** Reading each dataset in its native format in every module (N × M
adapters); Hugging Face `datasets` objects as the exchange format (heavy, and ties every module
to one library); CSV (quoting issues with transcripts, no nested metadata).

**Consequences.** Adding a dataset means writing one ingest function. Manifests are plain text
and can be inspected with standard tools.

## 2. Error rates are corpus-level and computed from per-utterance counts

**Context.** WER can be reported as total errors over total reference words (corpus level) or
as the mean of per-utterance WERs.

**Decision.** WER and CER are always corpus-level: Σ(S + D + I) / ΣN. The alignment produces
one `ErrorCounts` (hits, substitutions, deletions, insertions) per utterance; corpus rates sum
these counts and divide once.

**Alternatives rejected.** The mean of per-utterance WERs gives a one-word utterance the same
weight as a fifty-word one (e.g. 1/1 and 0/99 errors give a mean of 50% but a corpus WER of 1%).

**Consequences.** Per-utterance counts are reused without re-aligning by bootstrap resampling
(decision 6) and, later, by slicing (WER per speaker, duration, noise level).

## 3. Own alignment implementation, cross-checked against jiwer

**Context.** Mature libraries (jiwer) compute WER. The alignment is also the basis of error
analysis (which words are confused with which).

**Decision.** Levenshtein alignment is implemented in the project (dynamic programming plus
backtrace, unit costs) and returns the full alignment, not only counts. Corpus WER and CER are
cross-checked against jiwer on randomly generated corpora in the test suite.

**Ties.** Several alignments can have the same minimum cost. The backtrace prefers
match/substitution, then deletion, then insertion. This never changes WER, but it can change how
errors are split into S, D and I, so different tools may report different S/D/I for the same
text. Only total rates are compared with jiwer.

**Alternatives rejected.** Using jiwer alone: it hides the alignment that error analysis needs,
and any tie-breaking rule would be implicit.

**Consequences.** Alignments are available for error analysis and the dashboard; agreement with
a reference implementation is tested continuously.

## 4. The error rate of an empty reference is infinite

**Context.** An utterance with an empty reference (silence, noise) has N = 0, so its WER is
mathematically undefined. Tools disagree: jiwer 4 divides by max(N, 1); other tools return NaN
or raise.

**Decision.** `error_rate` returns 0.0 when there are no errors and `inf` otherwise.

**Alternatives rejected.**
- `max(N, 1)`: turns an insertion count into a fake rate; a one-word hallucination on silence
  scores 1.0, the same as getting every word of a normal utterance wrong.
- NaN: silently skipped by most aggregations (pandas, Polars, `np.nanmean`), which would hide
  hallucinations from summaries, the opposite of what an evaluation should do.
- Raising an error: blocks pipelines on datasets that deliberately contain silent clips.

**Consequences.** Hallucinations on silence rank first when sorting utterances by error rate.
Corpus-level rates are unaffected (inserted words are simply added to the error count) unless
every reference is empty. `inf` is not valid JSON, so per-utterance results are stored as counts,
not rates. Silent clips are evaluated with a dedicated hallucination metric rather than WER.

## 5. Text normalizers are explicit, composable and versioned

**Context.** Casing, punctuation, typographic apostrophes and Unicode encodings can be counted
as recognition errors. On the same FLEURS Italian transcripts, two reasonable normalizations
disagree on about 6% of words with no recognition errors involved.

**Decision.** A normalizer is a named pipeline of small single-purpose steps (Unicode NFC,
apostrophe unification, lowercasing, punctuation removal, language-specific steps, whitespace
collapsing). Names carry a version (`it-v1`, `de-v1`, `en-v1`, `basic-v1`, `none`). Changing a
rule creates a new version; existing versions never change. References and hypotheses always go
through the same normalizer, and its name is reported with every result.

Language-specific choices: Italian elisions are split (`l'uomo` → `l uomo`) so that every
apostrophe variant gives the same tokens; German `ß` becomes `ss`, umlauts are kept; accents are
kept everywhere (`è` and `e` are different words). Numbers are not normalized yet.

**Alternatives rejected.** A single opaque normalization function (results could not be traced
to the rules used); Whisper's normalizers only (English-centric, not versioned by this project).

**Consequences.** WER values computed with different normalizers are not comparable, and the
reports say so. The effect of normalization is measured explicitly (experiment E3).

## 6. Confidence intervals by bootstrap over per-utterance counts

**Context.** A WER measured on a few hundred utterances is an estimate; differences between
systems are often a fraction of a point.

**Decision.** Percentile bootstrap (default 1,000 resamples, 95%, fixed seed) for confidence
intervals, and paired bootstrap for comparisons: each replicate draws the same utterances for
both systems. Resampling works on per-utterance counts, vectorized with numpy (10,000 replicates
over 1,000 utterances take about 0.1 s). An optional `groups` argument resamples whole speakers,
because utterances of the same speaker are correlated (Bisani & Ney, 2004).

**Alternatives rejected.** Reporting point estimates only; unpaired comparisons (they add the
sampling noise of both systems); analytical intervals (WER is a ratio of correlated sums).

**Consequences.** Every reported number has an interval; comparisons report the interval of the
difference and the fraction of replicates in which the new system wins. The bootstrap measures
sampling noise, not whether the test set is representative. Sweeps with many configurations need
a multiple-comparison correction.

## 7. FLEURS utterances get a per-utterance speaker placeholder

**Context.** FLEURS has no speaker ids. Each sentence is read by two or three speakers
(Italian test split: 865 utterances, 346 distinct sentences, 827 female and 38 male voices).

**Decision.** `speaker_id` is a unique placeholder per utterance, and `metadata.sentence_id`
groups the readings of the same sentence. The original FLEURS normalized transcript is kept in
metadata for comparison.

**Alternatives rejected.** A single `"unknown"` speaker: speaker-level bootstrap would see one
group and return a zero-width interval, i.e. false certainty.

**Consequences.** Speaker-level resampling on FLEURS reduces to utterance-level resampling.
Grouping by `sentence_id` captures the correlation between readings of the same text (a hard
proper noun is hard for every reader). The gender imbalance is reported with results.

## 8. Models share one interface; audio is loaded outside the models

**Context.** The project compares models from different families and libraries (Whisper,
wav2vec2 with CTC, NVIDIA Parakeet).

**Decision.** Every adapter subclasses `ASRModel` and implements only `name` and
`_transcribe_batch(samples) -> hypotheses`. The public `transcribe` (template method) checks
that the adapter returned exactly one hypothesis per sample, in the same order. Audio is loaded
and validated once (`load_sample`: mono, float32, 16 kHz) and passed to models as arrays.
Hypotheses carry raw text; n-best lists and scores are optional.

**Alternatives rejected.** Letting each adapter read files and check its own output: duplicated
code, and a single forgotten check would silently misalign predictions and references, producing
wrong WERs with no visible error.

**Consequences.** Adding a model means writing one adapter. Adapters can be tested with
synthetic arrays. Raw predictions are saved and can be re-scored with any normalizer without
re-running inference.

## 9. Documentation examples are executed by the test suite

**Context.** Usage examples in documentation go stale when code changes, especially after
indirect changes (e.g. a normalizer change alters a WER shown in the README).

**Decision.** Examples in docstrings and in the README are written as doctests and run by
pytest in CI (`--doctest-modules`, `--doctest-glob=README.md`). Expected values are computed by
running the code, never by hand.

**Consequences.** Documentation cannot silently disagree with the code; in practice the doctests
already caught two hand-written values that were wrong. Results on real datasets cannot be
doctests (the data is not in the repository) and live in experiment reports instead.

## 10. Lesson learned: never seek backwards in a compressed archive

**Context.** The first real FLEURS ingest (865 WAV files inside a 658 MB `tar.gz`) was still
running after 36 minutes, at 97% CPU, and slowing down as it progressed.

**Cause.** A `tar.gz` is a single compressed stream: seeking backwards restarts decompression
from the beginning of the archive. The audio decoder was given each file directly from the
archive and seeked back to re-read the header, so each file cost a decompression of everything
before it: total work grew quadratically with the number of files.

**Fix.** The archive is opened in stream mode (`r|gz`), which forbids seeking, and each file is
read into memory before decoding. The full ingest now takes about 9 seconds.

**Lesson.** Unit tests on a tiny synthetic archive checked correctness but could not reveal the
cost on real data: every pipeline is run once on real data before it is considered done. A job
that slows down as it progresses usually repeats work.

## 11. Local Whisper inference with faster-whisper, int8, 4 threads

**Context.** Development runs on an ARM64 CPU with no GPU. Two Whisper implementations were
measured with Whisper-small on real FLEURS Italian utterances (real-time factor = processing time
/ audio duration; lower is faster):

| | faster-whisper (CTranslate2, int8) | transformers + PyTorch (float32) |
|---|---|---|
| RTF, greedy | **0.27** | 1.56 (4 threads), 1.89 (12 threads) |
| RTF, beam size 5 | **0.34** | 2.61 (12 threads) |
| Installed dependencies | about 315 MB | about 5.1 GB (the ARM64 PyTorch wheel ships CUDA libraries) |

Thread count was also measured for faster-whisper (beam size 5): RTF 0.35 with 4 threads, 1.18
with 8 and 1.63 with 12. Beyond 4 threads synchronization costs outweigh the extra cores on this
machine (WSL shares cores with Windows).

**Decision.** Local Whisper inference uses faster-whisper with int8 weights, beam size 5
(Whisper's default) and 4 threads. The language is taken from the manifest instead of being
detected, and `condition_on_previous_text` is disabled (a known source of repetition loops and
hallucinations). Every setting that can change the output is exposed by the adapter's `config`
and stored with the results.

**Alternatives rejected.** transformers for local inference: about six times slower and a much
heavier install. It will still be used where it is needed (wav2vec2 models, fine-tuning on a GPU),
as an optional dependency.

**Consequences.** The full FLEURS Italian test set (3.5 hours of audio) takes about 70 minutes
with beam size 5. int8 quantization may change transcripts slightly compared with the original
float32 weights; the model name includes the precision (`faster-whisper/small-int8`) and the
effect can be measured (float32 is one setting away). The 8 idle cores can later run several
model instances in parallel.

## 12. One run directory per configuration, resumable, with provenance

**Context.** Transcribing the full FLEURS Italian test set takes about an hour on CPU, and every
reported number must be traceable to the model, settings, data and code that produced it.

**Decision.** `asr-lab transcribe` writes a run directory with `predictions.jsonl` (raw
hypotheses, appended batch by batch) and `run.json`: model name and full decoding config,
manifest path and SHA-256, git commit, Python version and platform, audio duration, processing
time and real-time factor. A run is identified by model, config, manifest hash and utterance
limit: running again with the same identity resumes (already transcribed utterances are skipped),
running with a different identity in the same directory is an error. Elapsed time is saved even
when a run is interrupted, so the real-time factor of a resumed run stays correct.

The git commit is read from the `.git` directory rather than by running `git`, because the code
never runs shell commands (same behaviour on Linux and Windows). As a consequence uncommitted
changes cannot be detected, and `run.json` records `"git_dirty": "unknown"`.

**Alternatives rejected.** Recomputing everything on each run (an hour per attempt); a results
database or MLflow at this stage (added later for experiment sweeps; plain files are enough and
easy to inspect); overwriting a run directory silently when settings change (mixes predictions
from different configurations).

**Consequences.** Interrupted runs cost nothing; one directory is always one configuration; scores
can be recomputed from saved predictions with any normalizer. Model loading time is not included
in the real-time factor.

## 13. Report the widest confidence interval across resampling units

**Context.** Bootstrap intervals depend on what is resampled. Utterances that share a speaker,
or that read the same sentence (FLEURS has two or three readings per sentence), have correlated
errors; resampling them one by one treats them as independent and gives intervals that are too
narrow. Which grouping matters most depends on the dataset.

**Decision.** `asr-lab score` computes an interval for every meaningful unit: utterances always;
`metadata.sentence_id` when every utterance has one; speakers when at least one speaker has
several utterances (FLEURS speaker ids are per-utterance placeholders, see decision 7, so this
unit is skipped there). The widest interval is reported as the headline; all of them are stored in
the scores file. The headline is printed with its unit and number of units, e.g.
`WER 5.96% 95% CI [2.38%, 10.35%] (resampling utterance, n=10)`.

**Alternatives rejected.** Always resampling utterances (over-optimistic with correlated data);
choosing one unit per dataset by hand (easy to forget, and an implicit choice in every report).

**Consequences.** The reported uncertainty is conservative and the choice is automatic and
visible. With few units (e.g. a small number of speakers) intervals become wide, which is the
honest outcome rather than a defect.

## 14. Scores files are named after their settings

**Context.** Scoring is cheap and deterministic, so the same predictions are scored many times:
with different normalizers (experiment E3) or bootstrap settings. A single `scores.json` per run
directory would be overwritten each time.

**Decision.** The file name contains every scoring setting that changes the result, leaving out
defaults: `scores_it-v1.json`, `scores_basic-v1.json`, `scores_de-v1+it-v1_r5000_s1.json` (several
normalizers for a multilingual manifest, 5,000 resamples, seed 1).

**Alternatives rejected.** One file overwritten on every call (results are lost silently);
timestamped files (identical copies pile up, and it is unclear which one is current).

**Consequences.** Same settings give the same file with the same content; different settings
never collide. A cross-run view (e.g. MLflow) can be built on top of these files later.

## 15. Whisper's basic normalizer is reimplemented, not imported

**Context.** OpenAI Whisper's `BasicTextNormalizer` is the default normalization for non-English
text in many papers and leaderboards, so results are easier to compare when it is available. It
ships with the `openai-whisper` package, which depends on PyTorch (about 5 GB on this machine).

**Decision.** The normalizer is reimplemented in a few lines as `whisper-basic` (lowercase; remove
text in square/angle brackets and in parentheses; Unicode NFKC; replace every mark, symbol and
punctuation character with a space; collapse whitespace) and tested on known cases. Normalizers
can be selected by name on the command line (`--normalizer whisper-basic`).

**Alternatives rejected.** Depending on `openai-whisper` for one function; leaving it out (losing
comparability with published numbers).

**Consequences.** Small differences from the original are possible and are declared; tokens are
the same on the tested cases. A first use already showed a pitfall: FLEURS references contain
spoken text in parentheses (103 of 865 Italian test utterances), which `whisper-basic` deletes,
turning correctly transcribed words into insertions (experiment E3).

## 16. it-v2 writes numbers in words, but is not the default yet

**Context.** With `it-v1`, a correctly recognized number written differently in the reference and
in the hypothesis ("20 anni" / "vent'anni", "3 km" / "tre chilometri") counts as an error. Error
analysis of Whisper-small on FLEURS Italian found 56 substitutions involving digits or symbols.

**Decision.** `it-v2` = `it-v1` plus: Italian number formats written in words with `num2words`
(thousands dot "2.000", decimal comma "3,5", times "10:30", digits attached to letters such as
"M16" untouched); currency moved before the amount and spelled out ("$5" → "cinque dollari");
percent; only unambiguous units (`km`, `km/h`, `cm`, `mm`, `kg`; not `ha`, which is mostly the
verb "has"); elided tens restored ("vent'anni" → "venti anni"). Numbers are converted before
punctuation is removed. The default Italian normalizer stays `it-v1`.

**Alternatives rejected.** Converting spoken numbers to digits (inverse text normalization):
harder to make reliable for Italian than number-to-words. Making `it-v2` the default immediately:
it would silently change every Italian number already reported; the switch, if any, will follow
experiment E3 and be versioned.

**Consequences.** On Whisper-small (same predictions) WER goes from 7.46% to 7.15%, a significant
paired difference (-0.31 points, 95% CI [-0.48, -0.15]), and digit/symbol substitutions drop from
56 to 10. The rules were chosen partly by looking at errors on the test set; they apply equally to
every system and do not favour a model, but this is declared in the E3 report. Ordinals, fractions,
Roman numerals and alternative spoken forms ("ventun anni", "dieci e mezza") are not handled.

## 17. Results tables are generated and checked against committed results

**Context.** Results are presented in a long document (`RESULTS.md`) and summarized in the README.
Numbers typed by hand go stale or are copied wrongly (two hand-written values in the README were
already caught by doctests), and experiment outputs lived only in a local, untracked `runs/`
folder.

**Decision.** Every experiment has a folder in `experiments/` with a script (`run.sh` and
`run.ps1`) that regenerates all its numbers, and a TOML spec listing the tables it contributes.
`asr-lab render-results` builds the tables from the JSON results and writes them between
`<!-- BEGIN name -->` / `<!-- END name -->` markers, leaving the prose untouched. A test renders
the documents in check mode and fails if any table differs from the committed results. The
transcripts (`predictions.jsonl`, about 250 KB per run) and the scores, comparisons and error
reports of the runs used in the results are committed in `experiments/runs/` and
`experiments/<E>/results/`; `runs/` stays a local scratch area. Paired comparisons are produced
by `asr-lab compare` and saved as JSON, never computed ad hoc.

**Alternatives rejected.** A Jupyter notebook as the results document (hard to review in pull
requests, outputs can be stale or depend on execution order; a notebook is planned instead as a
hands-on quickstart); hand-written tables; not versioning predictions (every check would require
re-running inference).

**Consequences.** Anyone can verify every table from a clone in seconds. Prose numbers are still
written by hand and must be checked when a chapter is written. The repository grows by a few
hundred KB per run, acceptable for text predictions (audio is never committed).

## 18. A notebook for trying the pipeline, not for presenting results

**Context.** Results are presented in `RESULTS.md` (decision 17). A reader may also want to run the
pipeline and change a setting, ideally without installing anything.

**Decision.** `notebooks/quickstart.ipynb` runs the pipeline on a small random subset of FLEURS:
install the package from GitHub, settings in one cell (language, model, beam size, normalizer,
number of utterances), ingest, transcribe, score with a confidence interval, error analysis,
normalizers compared, one paired comparison. It contains no logic: every step calls `asr_lab`. A
"Open in Colab" badge opens it on Google Colab with a free GPU; the Whisper adapter gained a
`device` option (`cpu` or `cuda`, with `int8` and `float16` as the respective default precision).
Outputs are never committed (`nbstripout` pre-commit hook). Tests check that the notebook has no
outputs, that every code cell parses and that every `asr_lab` name it imports still exists; it is
not executed in CI, which would need a model and a dataset download.

**Alternatives rejected.** A notebook as the results document (decision 17); executing the
notebook in CI (slow and dependent on external downloads).

**Consequences.** Trying the project takes a click and a few minutes. Code that runs only on a GPU
(Colab) is not exercised by CI; the notebook was executed locally on CPU before release.

**Update (GPU on Colab).** The first Colab run failed on the GPU with `libcublas.so.12 is not
found`: CTranslate2, used by faster-whisper, is built for CUDA 12, while the current Colab image
ships a newer CUDA (and Python 3.13). The notebook now has a GPU setup cell, run only when a GPU is
present: it installs the CUDA 12 libraries from pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`),
loads them with `ctypes` (they are not on the system library path), and checks the GPU with
Whisper-tiny. Any failure falls back to the CPU with a message instead of stopping the notebook.
Verified on a Colab T4: 30 utterances with Whisper-small in about one second, against a few minutes
on the CPU; the fallback path was exercised on a machine without a GPU.

## 19. Large models on a Colab GPU, with a bridge run and predictions brought back as a zip

**Context.** Whisper-large-v3 and large-v3-turbo are too slow for the development machine (an
ARM64 CPU, no GPU), so they run on a Colab T4 GPU in float16, while the existing runs used int8 on
the CPU. A comparison between a CPU int8 run and a GPU float16 run changes the model, the precision
and the hardware at once.

**Decision.** `notebooks/e1_large_colab.ipynb` transcribes the three FLEURS test sets with
Whisper-small, large-v3-turbo and large-v3 on the GPU. Whisper-small is a *bridge* run: its paired
comparison with the CPU int8 run of the same model measures the effect of precision and hardware
alone, and the model comparisons (small → turbo → large-v3) are all made between GPU runs. The
notebook works inside a clone of the repository, so that every `run.json` records the git commit,
and builds the manifests with the same ingest code, so that their SHA-256 matches the local ones.
It only transcribes: it writes a zip with `predictions.jsonl` and `run.json` of each run, which is
unzipped into `experiments/runs/`; scores, comparisons and tables are computed locally by
`experiments/E1/run.sh`, exactly as for the CPU runs.

**Alternatives rejected.** Running the large models on the CPU in int8 (homogeneous with the other
runs, but hours per language); comparing CPU and GPU runs directly (confounded); pushing results
from Colab to GitHub (needs credentials in the notebook); saving to Google Drive (convenient for
long sessions, but an extra copy step and a Google account dependency).

**Consequences.** Real-time factors measured on the GPU are not comparable with the CPU ones and
are reported separately. If Colab images change so that the manifests differ, scoring stops with a
SHA-256 mismatch instead of producing wrong numbers.

## 20. Every decoding option is explicit and recorded

**Context.** The Whisper adapter passed only the beam size, `condition_on_previous_text` and
`vad_filter` to faster-whisper, and recorded only those. Everything else came from the library's
defaults without being recorded, including the temperature fallback: when an output looks broken
(compression ratio above 2.4 or average log-probability below -1.0), faster-whisper decodes again
by *sampling* at temperatures 0.2 to 1.0. The runs of E1 therefore used a fallback that their
`run.json` did not mention, and that is not deterministic when it is triggered. Experiment E2
varies these options, so they must be part of a run's identity.

**Decision.** The adapter takes every decoding option that E2 varies (temperatures, the two
fallback thresholds, `no_speech_threshold`, VAD, `hallucination_silence_threshold`) and passes all
of them explicitly, together with the ones that are never varied (`best_of`, `patience`,
`length_penalty`, `condition_on_previous_text`); all of them are recorded in `model_config`. The
defaults are faster-whisper 1.2.1's, so existing results do not change. Runs saved before the new
fields existed are still recognized as the same configuration: an adapter declares the values its
older runs used implicitly (`ASRModel.implicit_config`), and a missing field is compared with
that value. A real change of setting is still rejected.

**Alternatives rejected.** Rewriting the committed `run.json` files (it would edit provenance
after the fact); treating the old runs as a different configuration (they would have to be
transcribed again for nothing); recording only non-default values (the record would depend on the
library version's defaults).

**Consequences.** `run.json` describes the decoding completely. The implicit values are correct
only for faster-whisper 1.2.1, the version recorded in every existing run; a later upgrade that
changes a default must not reuse them for its own runs, which have every field recorded anyway.

## 21. MLflow mirrors the JSON results, it does not replace them

**Context.** Experiment E2 produces dozens of runs (decoding settings × languages). Comparing them
by opening JSON files is slow; an experiment tracker gives a table of runs, filters and plots.
MLflow is the most common open-source choice. But the results already have a complete,
version-controlled record (run directories, scores files, generated tables, decision 17).

**Decision.** `asr-lab score` and `asr-lab compare` take `--mlflow EXPERIMENT`, which logs the
saved result to MLflow (`asr_lab/tracking.py`): metrics with their confidence intervals, every
model and decoding setting as a parameter, real-time factor, git commits, and the JSON file as an
artifact. The JSON files remain the source of truth: no table or document is generated from
MLflow. Each result file maps to one MLflow run, found by a tag with the file's path, so logging it
again replaces that run instead of creating a duplicate. MLflow is an optional extra
(`tracking`): the Colab notebooks and anyone who only runs the pipeline do not install it. The
store is MLflow's default local SQLite database (`mlflow.db`, not committed), or whatever
`MLFLOW_TRACKING_URI` points to. CI installs all extras, so the tracking tests run there.

**Alternatives rejected.** MLflow as the primary store (results would depend on a database
outside git, and the tables could no longer be checked against committed files); logging during
transcription (scores do not exist yet then, and a run's identity is already in `run.json`); a
mandatory dependency (MLflow and its dependencies are large, for a feature that is optional).

**Consequences.** The MLflow store can be deleted and rebuilt at any time by rescoring with
`--mlflow`. Comparisons are logged only when saved with `--output`, since MLflow mirrors files.

## 22. Decoding options are chosen on the dev set by a rule written in code

**Context.** Experiment E2 tunes Whisper-large-v3-turbo's decoding options (beam size,
temperature fallback, voice activity detection, hallucination silence threshold) for speed and
accuracy. Tuning on the test set would make its result optimistic; choosing by eye on the dev set
leaves room for choices that are hard to reproduce, and for a look at the test set in between.

**Decision.** A sweep is a TOML spec (`experiments/E2/sweep.toml`): one checkpoint, the languages
with their normalizers, and named configurations that each change one option from the E1 setup.
`asr_lab.sweep.run_sweep` transcribes them all with one loaded model (`with_decoding` shares the
weights), after a warm-up so that no configuration pays for initialising the GPU.
`select_configuration` then applies a rule fixed before any result exists: in each language the
configuration with the lowest WER is the best; a configuration is eligible if a paired bootstrap
does not find it significantly worse than the best of any language; the eligible configuration
with the shortest total processing time wins, and if none is eligible the baseline is kept. One
configuration serves every language, as one deployment would. The Colab notebook runs the dev
sweep, the rule and the test run of the selected configuration in one session, without anyone
looking at the dev results in between. Every run must have exactly its configuration's settings
in `run.json`, or the selection refuses it. A repeat of the baseline, which is never selectable,
measures how much two identical runs differ in WER and speed.

**Alternatives rejected.** One configuration per language (three times the choices and the risk
of overfitting the dev set, for a deployment that would need three setups); a full grid of
combinations (dozens of runs; one factor at a time shows each option's effect, at the cost of
missing interactions); choosing the lowest dev WER (ignores speed, the reason to tune decoding);
a correction for multiple comparisons in the rule (uncorrected tests reject more configurations,
which errs on the side of accuracy).

**Consequences.** The test set is used once. The best dev WER is optimistic, since it is the
minimum of several noisy estimates, so the test comparison with E1 is the result to report.
Speeds are measured on a shared Colab GPU: they are comparable within one session only, and the
repeated baseline shows how much they vary.

**After E2.** The rule selected the hallucination-silence threshold, which differs from the
baseline in four dev transcripts, because its German interval against voice activity detection
reached +0.56 points without excluding zero. "Not significantly worse" is not "as good": the next
sweep should fix a non-inferiority margin in advance (eligible only if the upper bound of the
difference is below it). The rule was not changed after the results.

## 23. Hallucinated closing phrases are counted in the error report; figures are an optional extra

**Context.** E1 found that the large models end some German transcripts with "Vielen Dank", a
phrase learned from subtitled videos. It costs little WER but is very visible to a user, and E2
tests settings meant to reduce it, so it has to be counted the same way for every run. The E1
counts came from a one-off script. E2 also needs tables that are not scores or comparisons (the
selection, counts per configuration) and a figure of WER against speed.

**Decision.** `asr_lab.evaluation.hallucination` matches a short list of closing phrases per
language at the end of the normalized hypothesis, when the normalized reference does not end the
same way; `asr-lab errors` writes the matches to the error report
(`invented_closing_phrases`), so the count is recomputed from the saved predictions like every
other number. `asr-lab render-results` gained two table kinds, `selection` (from `select`'s JSON)
and `closing-phrases` (one error report per cell), so these tables are generated and checked by
the test suite too. The figure is drawn by `asr-lab plot-selection` with matplotlib, an optional
extra (`plots`), imported only when a figure is drawn; every number in it is also in a table.

**Alternatives rejected.** A generic hallucination detector (insertions at the end of an
utterance): it would also count genuine recognition errors, and the known phrases are what users
notice; a separate command for the count (one more file per run, while the error report already
holds the alignment); matplotlib as a core dependency (the pipeline and the Colab notebooks do not
need it); an interactive figure (GitHub shows static images only).

**Consequences.** The counts are a lower bound: invented endings not in the list are missed, and
adding a phrase changes past counts, so the list is part of the method and is versioned with the
code. Rerunning `asr-lab errors` on older runs only adds the new field.

## 24. Simulated usage sessions with real degradations

**Context.** Experiment E10 analyses usage logs as a product team would: error and correction
rates per segment, confidence calibration, where to invest. No real usage logs are available.
Fully simulated logs (text and errors generated by a script) would only show back what the
simulator was told to produce, so any finding would be circular.

**Decision.** Only the traffic is simulated; everything that produces errors is real. A TOML spec
(`experiments/E10/simulation.toml`) sets the mix of languages, channels and environments, and
`asr_lab.analytics.sessions.plan_sessions` draws 3,000 sessions from the FLEURS dev sets with a
fixed seed. Each session records its recording, condition and noise seed in the manifest
metadata. `render_sessions` then degrades the real audio (`asr_lab.data.augment`): pink noise
(stationary sources such as fans or traffic) or babble (four other FLEURS speakers of the same
language, reading other sentences) mixed at a fixed SNR, and a narrowband telephone channel (8 kHz,
G.711 mu-law) applied after the noise, as in a call from a noisy room. The model transcribes the
degraded audio on a Colab GPU, so errors and confidence are measured, not drawn. The session
manifest is committed (text only, about 2 MB); the audio (about 1.2 GB) is not, since it is
rebuilt exactly from the manifest, and the notebook checks that the plan drawn again is identical
to the committed one before transcribing. Sessions keep FLEURS's `sentence_id`, so the bootstrap
groups sessions built from the same sentence.

**Alternatives rejected.** Fully synthetic logs (circular); recorded noise corpora such as MUSAN
(an 11 GB download and a licence to check, for a demonstration that babble from FLEURS speakers
and shaped noise already make); the test sets as source (they stay untouched for the model
comparisons, and nothing in E10 is tuned, so the dev sets suffice); a "domain" field as in the
original plan (FLEURS is all Wikipedia text, so a domain label would have no real effect).

**Consequences.** Effects of noise, channel and language on errors are real for these
conditions, but their sizes depend on the simulated SNRs and on read speech; the traffic shares,
and therefore any "priority" computed from them, are assumptions and are reported as such. The
same analysis code would run unchanged on a manifest of real sessions.

## 25. Usage analytics: priorities as avoidable errors, confidence as a review budget

**Context.** E10 has to turn per-session errors into answers a product team can act on: where
the errors come from, where to invest, whether the model can flag its own mistakes, and what
the product's own telemetry would show. Each of these needs a definition that is simple to
explain and does not depend on arbitrary weights.

**Decision.** `asr_lab.analytics.usage` (command `asr-lab usage`) computes, from the saved
predictions and each session's language normalizer:
- **WER per segment** (language, channel, environment, SNR), corpus-level, with bootstrap
  intervals that resample sentences, and each segment's share of all word errors;
- **priority** of a segment (language × channel × environment) = the errors it would lose if
  it were as accurate as quiet headset sessions of the same language, as a share of all errors.
  It multiplies traffic (simulated) by the gap to the best condition (measured), in a unit that
  adds up across segments, instead of a matrix with arbitrary frequency and severity scores;
- **confidence as a review budget**: sessions sorted by Whisper's average log-probability, and
  the share of all word errors found by reviewing the least confident 5-30%, next to a random
  review and the best possible one (an oracle), plus the Spearman correlation with session WER;
- **background capture**: a babble session whose transcript is closer (in WER) to one of its
  background voices than to the speaker is flagged, counted per condition and followed through
  the review. Added after the first look at the worst transcripts, which repeated a background
  sentence; it is a definition, not a tuned threshold, and changes no other number;
- **corrections by simulated users**: each word error is noticed with a fixed probability and a
  transcript with a noticed error is corrected; the correction rate, the only signal real
  telemetry has, is compared with the true WER and the share of sessions with errors.
Tables are generated (`usage` table kind) in RESULTS.md and in a one-page summary
(`reports/e10_summary.md`), which the document test also checks.

**Alternatives rejected.** A frequency × severity matrix with hand-set severity levels (hard to
defend, and not additive); expected calibration error of word confidences (faster-whisper gives
segment log-probabilities, not calibrated word probabilities, and a review budget answers the
product question directly); a correction model fitted to data (there is none: the model is a
declared assumption, and its parameters are in the spec).

**Consequences.** Priorities depend on the simulated traffic mix and should be recomputed with
real traffic; error rates per condition and the confidence results are measured. The correction
analysis only shows what a correction rate can and cannot reveal under the stated user model.

## 26. Training-data quality: cheap features, pre-registered rules, clusters to read

**Context.** Experiment E6 looks for problems in a sample of training data before it is used:
transcripts that do not match the audio, empty transcripts, distorted or noisy audio, duplicates.
The plan also listed learned embeddings (WavLM, ECAPA, sentence encoders) with UMAP and HDBSCAN;
they need a GPU, several hours and are hard to interpret in a short report.

**Decision.** A reduced version on cheap, interpretable features:
- **Data.** Italian VoxPopuli (European Parliament speech with the parliament's transcripts, CC0),
  three row groups of the training set (about 3,000 utterances, 10 hours), read with ranged
  requests from the Hub's parquet files at a pinned revision
  (`asr_lab.data.voxpopuli`, `asr-lab ingest-voxpopuli`). It is also the planned domain of the
  fine-tuning experiment (E7). Ingestion keeps every utterance, empty transcripts included.
- **Features** (`asr_lab.data.quality`): duration; speaking rate in letters per second of the
  normalized transcript; a blind SNR estimate (energy ratio between the loudest and the quietest
  25 ms frames); clipping; repeated transcripts and audio; and the WER of large-v3-turbo (E2
  settings, transcribed on Colab) against each transcript, with its length ratio. The
  normalizer is `it-v2`, so that digits against words are not counted as label noise.
- **Rules** with thresholds written in `experiments/E6/quality.toml` before any feature was
  computed. Each rule is reported on its own and as a union, in utterances and hours.
- **Clusters** (`asr_lab.data.clustering`): k-means on standardized features, the number of
  clusters chosen by the silhouette score among 2-8, each cluster described by the features it
  stands out on and shown with the utterances closest to its centre.

**Alternatives rejected.** Learned embeddings with UMAP and HDBSCAN (cost and interpretability,
left for later); DNSMOS for audio quality (an extra model for audio that is mostly clean); filtering
inside ingestion (it would hide what the checks are meant to find); Common Voice (its transcripts
are prompts read aloud, so there is little label noise to find).

**Consequences.** A high WER of the strong model is evidence, not proof, of a wrong transcript:
it also follows hard audio and, in VoxPopuli, the parliament's edited transcripts, which drop
hesitations and repair sentences. The clusters are there to separate these cases by listening to
their examples; the rules are what a pipeline would apply.

**After E6.** The WER rule flagged 12.4% of the sample, and reading its transcripts showed two
mechanisms with the same symptom: large-v3-turbo collapsing ("Grazie a tutti." for a whole
speech) and segment boundaries shifted into the neighbouring speech. Two post hoc steps were
added, kept apart from the pre-registered spec: the flagged utterances are split by the length of
the model's transcript (`high_wer_kind`, thresholds as constants in the code, not in
`quality.toml`), and a second model (Whisper-small on the CPU) transcribes the collapsed ones
(`asr-lab quality --export-collapsed --second-run`). The exported manifest keeps the order of the
data manifest, so its SHA-256 is stable and the committed second run is checked against it. The
per-utterance features (`features.jsonl`, 1.4 MB) are rebuilt by the script and not committed;
the report `quality.json` holds every number of the chapter. For the fine-tuning experiment, a
transcript should be suspect only when two sources disagree with it.
