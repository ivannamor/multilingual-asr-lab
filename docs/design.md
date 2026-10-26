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
