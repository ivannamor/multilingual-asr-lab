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

- **Baseline in three languages.** On the FLEURS test sets, Whisper-small (faster-whisper, int8,
  beam size 5) reaches **6.66% WER in English** (95% CI [5.98, 7.36]), **7.46% in Italian**
  ([6.81, 8.08]) and **9.31% in German** ([8.58, 10.11]); Whisper-base reaches 10.03%, 16.09% and
  18.07%.
- **The benefit of the larger model depends on the language.** Whisper-small removes about half of
  Whisper-base's word errors in Italian (54%) and German (48%), but only a third in English (34%),
  where the smaller model is already much better.
- **Language is confounded with speaker gender.** The German test set has only male voices and the
  Italian one is 96% female, so the gaps between languages cannot be attributed to the language
  alone.
- **Normalization is a large hidden variable.** On the same Whisper-small Italian transcripts, WER
  ranges from 13.87% with no normalization to 7.15% with the most complete Italian normalizer.
  Almost half of the raw word errors are casing, punctuation and formatting.
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

FLEURS (Google), test splits for Italian, German and English (US), downloaded from the Hugging Face
Hub and converted to 16 kHz mono 16-bit WAV with a JSONL manifest. FLEURS is built on FLoRes, a
corpus of Wikipedia sentences translated into many languages, and the three test sets read almost
the same sentences: 343 of them are in all three. The content is therefore close to parallel
across languages, while the speakers are different.

| | Italian | German | English |
|---|---|---|---|
| Utterances | 865 | 862 | 647 |
| Hours of audio | 3.52 | 3.15 | 1.77 |
| Distinct sentences | 346 | 347 | 350 |
| Duration per utterance (median) | 5.3-52.9 s (13.9 s) | 3.6-47.9 s (11.4 s) | 3.6-29.3 s (9.5 s) |
| Speaker gender (utterances) | 827 female, 38 male | 862 male | 412 female, 235 male |

Each sentence is read by one to three speakers (two at most in English). FLEURS does not provide
speaker ids, so readings of the same sentence are the only known source of correlation between
utterances (see Statistics). References are the original transcripts with casing and punctuation;
in Italian, 103 contain text in parentheses that is read aloud.

The gender balance differs completely between languages: German is all male, Italian almost all
female. Language and gender are therefore confounded, and no Italian result should be read as
representative of male voices or German result of female voices.

### Models

| | |
|---|---|
| Models | Whisper-small and Whisper-base (OpenAI), through faster-whisper 1.2.1 (CTranslate2) |
| Precision | int8 weights |
| Decoding | beam size 5, language forced to the test set's language, `condition_on_previous_text` disabled, no voice activity detection |
| Hardware | 12-core ARM64 laptop CPU (Snapdragon X Elite), WSL2 Ubuntu, no GPU, 4 threads |
| Speed | real-time factor 0.11-0.20 for small (24 min for 3.5 h of Italian audio) and 0.06-0.11 for base; German was the slowest language for both models |

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
  computed both resampling utterances and resampling sentences; the widest is reported. In
  practice the sentence-level interval is always the widest, 19% to 46% wider than the
  utterance-level one.
- **Comparisons** between two systems, or two normalizers, use the paired bootstrap: each
  replicate draws the same sentences for both sides, and the interval of the difference is
  reported. Two separate intervals can overlap while the paired difference is clearly significant
  (see `it-v1 → it-v2` below): overlap is not a test.
- "B better in" is the fraction of bootstrap replicates in which B has the lower error rate.

## E1. Baseline

**Question.** How accurate are two off-the-shelf multilingual models, Whisper-small and
Whisper-base, on read speech in Italian, German and English, and how much does the larger model
gain in each language?

**Method.** Each model transcribes each test set once, with the settings in [Setup](#models).
Transcripts are scored with the normalizer of their language: `it-v1`, `de-v1` (`basic-v1` plus
"ß" → "ss") and `en-v1` (`basic-v1`); see [E3](#normalizers-compared) for what they do. The two
models are compared with the paired bootstrap within each language. Languages are compared only
descriptively: the utterances differ, so no paired test is possible between them.

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

<!-- BEGIN e1-model-gap -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| Italian: base → small | 16.09% | 7.46% | -8.63% [-9.38%, -7.89%] | 100.0% | yes |
| German: base → small | 18.07% | 9.31% | -8.75% [-9.70%, -7.89%] | 100.0% | yes |
| English: base → small | 10.03% | 6.66% | -3.37% [-3.98%, -2.76%] | 100.0% | yes |
<!-- END e1-model-gap -->

### Larger models on a GPU

Whisper-large-v3-turbo and Whisper-large-v3 were transcribed in float16 on a Colab T4 GPU
([notebook](notebooks/e1_large_colab.ipynb)), together with a *bridge* run of Whisper-small on the
same GPU: comparing it with the CPU int8 run of the same model measures the effect of hardware and
precision alone, so that the model comparisons below are not confounded with it.

<!-- BEGIN e1-gpu -->
<!-- END e1-gpu -->

<!-- BEGIN e1-gpu-gaps -->
<!-- END e1-gpu-gaps -->

### Findings

1. **English is the easiest language for both models, German the hardest.** Whisper-small reaches
   6.66% WER in English, 7.46% in Italian and 9.31% in German. The English and Italian intervals
   overlap; German is clearly behind. The ranking is the same for Whisper-base (10.03%, 16.09%,
   18.07%). CER ranks the languages differently: for Whisper-small English has the highest CER
   (2.75%, against 2.11% for Italian), so even the choice between WER and CER changes which
   language looks easiest.
2. **The larger model is always significantly better, but by how much depends on the language.**
   Whisper-small removes 54% of Whisper-base's word errors in Italian, 48% in German and only 34%
   in English, where Whisper-base is already much better. A plausible explanation is that English
   dominates Whisper's training data, so even the smaller model has seen enough of it; these
   experiments do not test that explanation.
3. **Most errors are substitutions in every language**, spread over hundreds of distinct word pairs
   (885 in Italian, 1,059 in German, 548 in English for Whisper-small): there is no single dominant
   error to fix.

**Speaker gender.** Language and gender are confounded (see [Data](#data)), so the gap between
German and the other languages may be partly a gender effect. Within a language, the scores can be
split by gender only where both are present; the subsets read different sentences, so these
figures are descriptive, not a test:

| Whisper-small, WER [95% CI] | Female | Male |
|---|---|---|
| Italian (827 / 38 utterances) | 7.17% [6.57, 7.78] | 13.67% [10.22, 17.81] |
| English (412 / 235 utterances) | 6.41% [5.64, 7.20] | 7.09% [5.91, 8.38] |

In English the two groups are close and their intervals overlap widely. The Italian male subset is
almost twice as hard, but it has 38 utterances from an unknown, probably very small, number of
speakers, so it says more about those speakers than about male voices in general. These figures
were computed with a one-off script, not with a committed command, which is why they are not in a
generated table.

### Error analysis

The most frequent errors (`asr-lab errors`, Whisper-small) are different in each language, and
many of them are about writing conventions rather than recognition.

**Italian.**
- **Formatting rather than recognition**: abbreviations and numbers ("km" vs "chilometri"), elided
  forms ("delle" vs "dell'"). Experiment E3 measures how much of the WER they account for.
- **"e" (and) and "è" (is)**, which sound almost the same, confused 22 times: a context error that a
  language model could help with.
- **Rare proper nouns** replaced by frequent words that sound alike: *Geospiza* became
  "joe s pizza", *Martelly* became "Martelli".
- **Hallucination**: in 5 of 865 utterances Whisper appends "Sottotitoli (e revisione) a cura di
  QTSS" ("subtitles by QTSS"), text learned from subtitled videos in its training data, which
  never occurs in the references. No such text appears in the German or English transcripts.

**German.**
- **Compounds.** German writes compound nouns as one word, and speakers do not pause between their
  parts, so the model has to decide where a word ends. A simple check (a reference word that the
  hypothesis writes as two consecutive words, or the reverse) finds 86 compounds split by
  Whisper-small and 45 words joined. Each costs two word errors: one substitution plus one
  insertion or deletion. Part of this is a normalization artifact: `de-v1` turns hyphens into
  spaces, so the reference "Schengen-Raum" becomes two words while the model's "Schengenraum" is
  one; "Schengen" is among the most deleted words (8 times).
- **Abbreviations**: "n. Chr." (*nach Christus*, AD) and "usw." (*und so weiter*, etc.) are written
  in full by the model, and "km" as "Kilometer": about 25 substitutions, all formatting.
- **Short function words** that differ by one sound: "einen" vs "ein", "zum" vs "zu", "im" vs "in",
  "von" vs "vom". Whisper-base also confuses "das" (the, that) and "dass" (that, conjunction) 17
  times; they sound the same and differ only in grammar.

**English.**
- **Spelling variants**: the references use British spelling and the model American spelling
  ("travellers" vs "travelers", 7 times). `en-v1` does not map spelling variants (Whisper's English
  normalizer does), so these are counted as errors.
- **Short, unstressed words**: "in" heard as "and" 15 times, "a" and "the" swapped 18 times, and
  "the", "a", "to" are the most frequent deletions and insertions.
- **Numbers**: 61 substitutions involve digits or symbols, more than in Italian (56) although
  English has fewer words: `en-v1`, like `de-v1`, does not write numbers in words.

**Interpretation.** WER differences between languages mix three things: how hard the language is
for the model, who the speakers are, and how well the normalizer fits the language's writing
conventions. The normalizers for German and English are much simpler than `it-v2` (E3), so their
WER is probably inflated by formatting errors that `it-v2` would remove in Italian.

### Limitations

- Language is confounded with speaker gender, and speaker ids are not available.
- `de-v1` and `en-v1` handle neither numbers nor abbreviations, `de-v1` splits hyphenated
  compounds and `en-v1` keeps spelling variants. Comparing languages fairly needs normalizers of
  similar coverage.
- The test sets are small (1.8 to 3.5 hours) and contain read Wikipedia text only.
- No correction for multiple comparisons is applied to the three model comparisons; every one has
  Whisper-small better in all 1,000 replicates, so a corrected threshold would not change any
  verdict.

E1 will be extended to other model families (larger Whisper models, wav2vec2 with CTC,
Transducers).

**Reproduce:** `bash experiments/E1/run.sh` (or `experiments/E1/run.ps1` on Windows). With the
committed transcripts it only rescores; without them it downloads the three test sets and
transcribes them (about two hours on a laptop CPU).

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

- All results so far are on FLEURS read speech, where language is confounded with speaker gender;
  noisier and spontaneous speech come next. The normalization study (E3) covers Italian only.
- Only two Whisper sizes have been evaluated; other model families (CTC with an n-gram language
  model, Transducers) are planned.
- Decoding settings (beam size, temperature, voice activity detection, precision) are fixed here;
  their effect is the subject of a later experiment, compared with the same paired statistics.
