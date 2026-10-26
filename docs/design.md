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
