# ASR Lab: results

This document collects every experiment run so far, with its method, results and limitations. It
grows with each experiment. All tables are generated from the JSON results committed in
[`experiments/`](experiments/) by `asr-lab render-results`, and a test fails if a table no longer
matches them, so the numbers below cannot drift from the data. Each chapter ends with the command
that regenerates it.

**Contents**
1. [Executive summary](#executive-summary)
2. [Setup](#setup): data, models, metrics, statistics, hardware
3. [E1. Baseline](#e1-baseline)
4. [E3. How much does text normalization change WER?](#e3-how-much-does-text-normalization-change-wer)
5. [Limitations and next steps](#limitations-and-next-steps)

## Executive summary

- **Baseline.** Whisper-small (faster-whisper, int8, beam size 5) reaches **7.46% WER** (95% CI
  [6.81, 8.08]) and 2.11% CER on the FLEURS Italian test set; Whisper-base reaches 16.09%.
- **Normalization is a large hidden variable.** On the same Whisper-small transcripts, WER ranges
  from 13.87% with no normalization to 7.15% with the most complete Italian normalizer. Almost
  half of the raw word errors are casing, punctuation and formatting.
- **A widely used normalizer can make results worse.** Whisper's own basic normalizer deletes text
  in parentheses. In FLEURS that text is read aloud (103 of 865 references), so correctly
  transcribed words become insertions: +1.87 WER points (95% CI [1.03, 2.79]) compared with a
  simple lowercase-and-punctuation normalizer.
- **Rankings are stable, relative improvements are not.** Whisper-small beats Whisper-base by 8.6
  to 9.3 points under every normalizer, but the relative error reduction it would be credited with
  ranges from 40% (no normalization) to 55%.
- **Numbers matter.** Writing numbers, symbols and units in words (`it-v2`) lowers Whisper-small's
  WER by 0.31 points (95% CI [0.15, 0.48]) and removes most digit/symbol substitutions (56 to 10).

## Setup

### Data

FLEURS (Google), Italian test split, downloaded from the Hugging Face Hub and converted to 16 kHz
mono 16-bit WAV with a JSONL manifest.

| | |
|---|---|
| Utterances | 865 (3.52 hours of audio) |
| Distinct sentences | 346: each sentence is read by two or three speakers |
| Duration per utterance | 5.3 s to 52.9 s, median 13.9 s |
| Speaker gender | 827 female, 38 male utterances |
| Speaker ids | not provided by FLEURS |
| References | original transcripts with casing and punctuation; 103 contain text in parentheses that is read aloud |

The gender imbalance is large: no result in this document should be read as representative of
male voices. Because speaker ids are missing, readings of the same sentence are the only known
source of correlation between utterances (see Statistics).

### Models

| | |
|---|---|
| Models | Whisper-small and Whisper-base (OpenAI), through faster-whisper 1.2.1 (CTranslate2) |
| Precision | int8 weights |
| Decoding | beam size 5, language forced to Italian, `condition_on_previous_text` disabled, no voice activity detection |
| Hardware | 12-core ARM64 laptop CPU (Snapdragon X Elite), WSL2 Ubuntu, no GPU, 4 threads |
| Speed | real-time factor 0.11 (small, 24 min for 3.5 h of audio) and 0.07 (base, 14 min) |

faster-whisper was chosen after a benchmark on this machine: it was about six times faster than
the PyTorch implementation, and more than 4 threads made it slower
([decision 11](docs/design.md#11-local-whisper-inference-with-faster-whisper-int8-4-threads)).

### Metrics

- **Word and character error rate at corpus level**: total substitutions, deletions and insertions
  divided by the total number of reference words (or characters), never the mean of per-utterance
  rates. The alignment is implemented in the project and cross-checked against jiwer.
- **Normalization** is applied identically to references and hypotheses before scoring. Every
  result states its normalizer; the normalizers are described in [E3](#normalizers-compared).
- Raw predictions are saved, so every score can be recomputed with another normalizer or
  statistic without running the models again.

### Statistics

- **Confidence intervals**: percentile bootstrap, 1,000 resamples, fixed seed. Readings of the same
  sentence have correlated errors (a hard proper noun is hard for every reader), so intervals are
  computed both resampling utterances and resampling the 346 sentences; the widest is reported. In
  practice the sentence-level interval is about 22% wider than the utterance-level one.
- **Comparisons** between two systems, or two normalizers, use the paired bootstrap: each
  replicate draws the same sentences for both sides, and the interval of the difference is
  reported. Two separate intervals can overlap while the paired difference is clearly significant
  (see `it-v1 → it-v2` below): overlap is not a test.
- "B better in" is the fraction of bootstrap replicates in which B has the lower error rate.

## E1. Baseline

<!-- BEGIN e1-baseline -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Whisper-small (faster-whisper, int8, beam 5) | `it-v1` | 865 / 21,255 | 7.46% [6.81%, 8.08%] | 1,223 / 169 / 193 | 2.11% [1.87%, 2.37%] |
| Whisper-base (faster-whisper, int8, beam 5) | `it-v1` | 865 / 21,255 | 16.09% [15.13%, 17.12%] | 2,595 / 402 / 422 | 4.16% [3.84%, 4.49%] |
<!-- END e1-baseline -->

Whisper-small makes fewer than half the word errors of Whisper-base. Most of its errors are
substitutions (1,223, against 169 deletions and 193 insertions), spread over 885 distinct word
pairs: there is no single dominant error to fix. The error analysis (`asr-lab errors`) shows four
recurring patterns:

- **Formatting rather than recognition**: abbreviations and numbers ("km" vs "chilometri"), elided
  forms ("delle" vs "dell'"). Experiment E3 measures how much of the WER they account for.
- **"e" (and) and "è" (is)**, which sound almost the same, confused 22 times: a context error that a
  language model could help with.
- **Rare proper nouns** replaced by frequent words that sound alike: *Geospiza* became
  "joe s pizza", *Martelly* became "Martelli".
- **Hallucination**: in 5 of 865 utterances Whisper appends "Sottotitoli (e revisione) a cura di
  QTSS" ("subtitles by QTSS"), text learned from subtitled videos in its training data, which
  never occurs in the references.

E1 will be extended to German and English and to other model families (larger Whisper models,
wav2vec2 with CTC, Transducers).

## E3. How much does text normalization change WER?

**Question.** Text normalization decides which differences between a reference and a model output
count as errors. How much does this choice change the WER of a given model, and can it change the
conclusions drawn when comparing two models?

### Normalizers compared

| Normalizer | What it does |
|---|---|
| `none` | no change |
| `basic-v1` | Unicode NFC, unify apostrophes, lowercase, remove punctuation (keeping apostrophes inside words), collapse whitespace |
| `whisper-basic` | reimplementation of OpenAI Whisper's `BasicTextNormalizer`, the usual choice for non-English text in papers and leaderboards: lowercase, delete text in brackets and parentheses, replace every mark, symbol and punctuation character (apostrophes included) with a space |
| `it-v1` | `basic-v1`, then split Italian elisions ("l'uomo" → "l uomo") |
| `it-v2` | `it-v1` plus numbers, symbols and unambiguous units in words: "2.000" → "duemila", "3,5" → "tre virgola cinque", "10:30" → "dieci e trenta", "$5" → "cinque dollari", "km" → "chilometri", "vent'anni" → "venti anni" |

**Method.** The saved transcripts of Whisper-small and Whisper-base (E1) are scored with each
normalizer, without running the models again. Normalizers are compared with the paired bootstrap
on the same Whisper-small transcripts; the two models are compared under each normalizer.

### Results

Whisper-small, the same transcripts scored with each normalizer:

<!-- BEGIN e3-normalizers-small -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Whisper-small | `none` | 865 / 20,605 | 13.87% [12.85%, 14.91%] | 2,455 / 184 / 218 | 3.42% [3.12%, 3.76%] |
| Whisper-small | `basic-v1` | 865 / 20,701 | 7.69% [7.06%, 8.34%] | 1,194 / 196 / 202 | 2.14% [1.90%, 2.40%] |
| Whisper-small | `whisper-basic` | 865 / 20,787 | 9.56% [8.44%, 10.75%] | 1,178 / 159 / 651 | 4.12% [3.31%, 5.07%] |
| Whisper-small | `it-v1` | 865 / 21,255 | 7.46% [6.81%, 8.08%] | 1,223 / 169 / 193 | 2.11% [1.87%, 2.37%] |
| Whisper-small | `it-v2` | 865 / 21,262 | 7.15% [6.49%, 7.78%] | 1,171 / 158 / 191 | 1.88% [1.66%, 2.13%] |
<!-- END e3-normalizers-small -->

Whisper-base, the same comparison:

<!-- BEGIN e3-normalizers-base -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Whisper-base | `none` | 865 / 20,605 | 23.13% [21.95%, 24.30%] | 3,859 / 440 / 467 | 5.81% [5.43%, 6.22%] |
| Whisper-base | `basic-v1` | 865 / 20,701 | 16.43% [15.51%, 17.45%] | 2,513 / 441 / 448 | 4.21% [3.89%, 4.54%] |
| Whisper-base | `whisper-basic` | 865 / 20,787 | 18.20% [16.78%, 19.74%] | 2,529 / 383 / 871 | 6.16% [5.34%, 7.09%] |
| Whisper-base | `it-v1` | 865 / 21,255 | 16.09% [15.13%, 17.12%] | 2,595 / 402 / 422 | 4.16% [3.84%, 4.49%] |
| Whisper-base | `it-v2` | 865 / 21,262 | 15.79% [14.86%, 16.82%] | 2,537 / 396 / 425 | 3.89% [3.59%, 4.21%] |
<!-- END e3-normalizers-base -->

Paired differences between normalizers (Whisper-small; B - A, negative when B gives a lower WER):

<!-- BEGIN e3-normalizer-comparisons -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| none → basic-v1 | 13.87% | 7.69% | -6.18% [-6.89%, -5.48%] | 100.0% | yes |
| basic-v1 → it-v1 | 7.69% | 7.46% | -0.23% [-0.36%, -0.12%] | 100.0% | yes |
| it-v1 → it-v2 | 7.46% | 7.15% | -0.31% [-0.48%, -0.15%] | 100.0% | yes |
| basic-v1 → whisper-basic | 7.69% | 9.56% | +1.87% [+1.03%, +2.79%] | 0.0% | yes |
<!-- END e3-normalizer-comparisons -->

Difference between the two models under each normalizer (A = Whisper-base, B = Whisper-small):

<!-- BEGIN e3-model-gap -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| base → small, `none` | 23.13% | 13.87% | -9.26% [-10.09%, -8.51%] | 100.0% | yes |
| base → small, `basic-v1` | 16.43% | 7.69% | -8.74% [-9.47%, -8.01%] | 100.0% | yes |
| base → small, `whisper-basic` | 18.20% | 9.56% | -8.64% [-9.42%, -7.92%] | 100.0% | yes |
| base → small, `it-v1` | 16.09% | 7.46% | -8.63% [-9.38%, -7.89%] | 100.0% | yes |
| base → small, `it-v2` | 15.79% | 7.15% | -8.64% [-9.39%, -7.91%] | 100.0% | yes |
<!-- END e3-model-gap -->

### Findings

1. **Casing and punctuation account for almost half of the raw errors.** Lowercasing and removing
   punctuation (`none` → `basic-v1`) lowers Whisper-small's WER from 13.87% to 7.69%. Reporting an
   unnormalized WER, or comparing it with a normalized one, misrepresents a model by a factor of
   almost two.
2. **Language-specific rules matter, even when small.** Splitting elisions (`it-v1`) and writing
   numbers, symbols and units in words (`it-v2`) each give a small but significant improvement
   (-0.23 and -0.31 points). With `it-v2`, substitutions involving digits or symbols drop from 56
   to 10 in the error analysis: most of them were not recognition errors.
3. **A standard normalizer can be wrong for a dataset.** `whisper-basic` deletes everything in
   parentheses, as if it were an annotation such as "(laughs)". In FLEURS, parentheses contain
   spoken text, so the words disappear from the reference and the model is penalized for
   transcribing them: insertions jump from 202 to 651 and WER rises by 1.87 points (Whisper-small)
   compared with `basic-v1`. Its CER nearly doubles (2.14% to 4.12%).
4. **The ranking of the two models never changes, but the size of the improvement does.** The
   absolute gap between Whisper-base and Whisper-small stays between 8.6 and 9.3 points. The
   relative reduction, the figure often quoted when announcing a better model, ranges from 40%
   (`none`) to 53-55% (`basic-v1`, `it-v1`, `it-v2`), with `whisper-basic` in between (47%).

**Interpretation.** WER figures computed with different normalizers are not comparable, even on the
same data and model. Comparisons inside one study are robust as long as the normalizer is fixed
and stated; comparisons across studies (or with leaderboards) need the same normalizer, and a
normalizer designed for one dataset can silently penalize another.

### Limitations

- One dataset (read speech) and one language. Spontaneous speech, with disfluencies and
  inconsistent transcription conventions, is likely to make normalization matter even more.
- `whisper-basic` is a reimplementation, tested on known cases but not compared output by output
  with the original package.
- The rules of `it-v2` were chosen partly after looking at errors on this test set. They apply
  identically to every system and favour none, but their effect on new data may be smaller.
  Ordinals, fractions, Roman numerals and alternative spoken forms ("ventun anni", "dieci e
  mezza") are not handled.
- Normalizers also change the number of reference words (20,605 to 21,262 here), for example by
  splitting elisions, so they change the denominator of WER as well as the error count.
- The variation of the model gap across normalizers is reported descriptively; it is not tested
  as a difference of differences.
- No correction for multiple comparisons is applied to the nine comparisons. Every one of them has
  B better (or worse) in all 1,000 replicates, so a Bonferroni-corrected threshold would not change
  any verdict.

**Reproduce:** `bash experiments/E3/run.sh` (or `experiments/E3/run.ps1` on Windows). With the
committed transcripts it only rescores; without them it first transcribes the test set
(about 40 minutes on a laptop CPU).

## Limitations and next steps

- All results so far are on FLEURS Italian read speech with a strong female majority; German and
  English, and noisier or spontaneous speech, come next.
- Only two Whisper sizes have been evaluated; other model families (CTC with an n-gram language
  model, Transducers) are planned.
- Decoding settings (beam size, temperature, voice activity detection, precision) are fixed here;
  their effect is the subject of a later experiment, compared with the same paired statistics.
