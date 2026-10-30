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
4. [E1b. Three model families on the same test sets](#e1b-three-model-families-on-the-same-test-sets)
5. [E2. Which decoding settings?](#e2-which-decoding-settings)
6. [E3. How much does text normalization change WER?](#e3-how-much-does-text-normalization-change-wer)
7. [E6. Which training transcripts can be trusted?](#e6-which-training-transcripts-can-be-trusted)
8. [E7. Does the choice of training data matter when the amount does not?](#e7-does-the-choice-of-training-data-matter-when-the-amount-does-not)
9. [E10. Usage analytics: where should a product team invest?](#e10-usage-analytics-where-should-a-product-team-invest)
   (simulated traffic, real audio and errors)
10. [Limitations and next steps](#limitations-and-next-steps)

## Executive summary

- **Baseline in three languages.** On the FLEURS test sets, Whisper-small (faster-whisper, int8,
  beam size 5) reaches **6.66% WER in English** (95% CI [5.98, 7.36]), **7.46% in Italian**
  ([6.81, 8.08]) and **9.31% in German** ([8.58, 10.11]); Whisper-base reaches 10.03%, 16.09% and
  18.07%.
- **Large models: 2.26% WER in Italian.** Whisper-large-v3 (float16, GPU) reaches 2.26% in Italian
  (95% CI [1.93, 2.64]), 4.02% in German and 4.62% in English. The language ranking reverses:
  English, easiest for the small models, is now the hardest, partly because the simple English
  normalizer counts British versus American spellings as errors. large-v3-turbo is about twice as
  fast and 0.4 points worse in every language.
- **A Transducer nearly matches large-v3-turbo at a third of the time, with one failure that WER
  alone would misread.** On FLEURS, Parakeet TDT is 0.36 to 0.64 points behind large-v3-turbo
  (2.76% against 2.40% in Italian, 5.09% against 4.45% in German, 5.45% against 5.00% in English)
  at about 2.8 times its speed on the same GPU, and wrote none of the invented closing phrases.
  A wav2vec2 CTC model without a language model is the fastest (5 to 6 times Whisper-small), level
  with Whisper-small in Italian and far behind in English (14.54%). On parliamentary speech, two
  runaway Parakeet transcripts (1,588 words for 12.7 seconds of audio) hold a quarter of its errors
  and widen its interval to [13.96, 30.89]: the decoding path in Transformers has no limit on the
  tokens written at one audio frame. Without those two utterances its WER is 15.35%, though it was
  probably trained on the same sessions.
- **The benefit of a larger model depends on the language.** Whisper-small removes about half of
  Whisper-base's word errors in Italian (54%) and German (48%), but only a third in English (34%),
  where the smaller model is already much better.
- **Language is confounded with speaker gender.** The German test set has only male voices and the
  Italian one is 96% female, so the gaps between languages cannot be attributed to the language
  alone.
- **Decoding settings matter little for WER; voice activity detection matters for
  hallucinations.** In a sweep on the dev sets, beam size, temperature fallback and Whisper's
  hallucination-silence threshold move large-v3-turbo's WER by at most a quarter of a point.
  Voice activity detection lowers German WER by 0.31 points on dev ([-0.59, -0.03]) and removes
  the invented "Vielen Dank" endings (12 to 0 on the German test set), for 6% more processing
  time. The pre-registered selection rule picked the hallucination-silence threshold, which is in
  practice the baseline, after a borderline comparison: an example of why such a rule needs a
  non-inferiority margin.
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
- **In simulated usage, competing speakers are the problem, and confidence helps find it.** On
  3,000 sessions with simulated traffic but real noise, telephone degradations and transcripts,
  large-v3-turbo's WER is 3.56% in quiet rooms and 45.47% with babble at 0 dB on a headset, where
  it transcribes a background voice instead of the speaker in 12.6% of the sessions. Babble
  segments hold 41.5% of all errors as avoidable, the telephone channel alone under 2%. Reviewing
  the least confident 5% of sessions catches 34% of all word errors, while a simulated correction
  rate understates how bad the worst conditions are.
- **A high WER against a transcript is not proof of a wrong transcript.** On 10 hours of Italian
  VoxPopuli training data, a pre-registered rule (WER of large-v3-turbo above 50%) flags 12.4% of
  the utterances. Read after the fact, about half of them (163) are not label noise but the model
  collapsing: it writes "Grazie a tutti." for a whole speech, and Whisper-small transcribes 159 of
  these 163 utterances. Most of the rest are real label noise, shifted segment boundaries that no
  acoustic or length rule detects. Filtering training data on one model's WER would have discarded
  0.56 hours of probably valid, and hard, speech.
- **Fine-tuning did not help, but the choice of data decided how much it hurt.** Whisper-small was
  fine-tuned with LoRA on 5 hours of that data chosen four ways, three seeds each. No strategy beat
  the base model in-domain (20.69% WER), and all of them forgot: FLEURS WER rose from 7.05% to
  8.47%-9.14%, and the models started to open encyclopedic sentences with "Signor Presidente,
  onorevoli colleghi". Filtering the flagged data or balancing the speakers was significantly
  better than random sampling (0.55 to 0.80 points in-domain, 0.43 to 0.67 on FLEURS, after
  Holm's correction), and the two filters did not differ. Between seeds, one strategy's WER moved
  by up to 0.87 points: one training run per strategy would have been misleading.

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
| Models | Whisper-base, Whisper-small, Whisper-large-v3-turbo and Whisper-large-v3 (OpenAI), through faster-whisper 1.2.1 (CTranslate2); from [E1b](#e1b-three-model-families-on-the-same-test-sets), wav2vec2 XLS-R (CTC) and Parakeet TDT 0.6B v3 (Transducer), through Transformers |
| Precision | int8 weights on the CPU, float16 on the GPU |
| Decoding | beam size 5, language forced to the test set's language, `condition_on_previous_text` disabled, no voice activity detection |
| Hardware | base and small: 12-core ARM64 laptop CPU (Snapdragon X Elite), WSL2 Ubuntu, 4 threads; large models and a bridge run of small: NVIDIA T4 GPU on Google Colab |
| Speed (CPU) | real-time factor 0.11-0.20 for small (24 min for 3.5 h of Italian audio) and 0.06-0.11 for base; German was the slowest language for both models. GPU speeds are in [E1](#larger-models-on-a-gpu) |

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

**Question.** How accurate are off-the-shelf multilingual Whisper models, from Whisper-base to
Whisper-large-v3, on read speech in Italian, German and English, and how much does a larger model
gain in each language?

**Method.** Each model transcribes each test set once, with the settings in [Setup](#models):
Whisper-base and Whisper-small on the CPU, the large models on a GPU (see
[below](#larger-models-on-a-gpu)).
Transcripts are scored with the normalizer of their language: `it-v1`, `de-v1` (`basic-v1` plus
"ß" → "ss") and `en-v1` (`basic-v1`); see [E3](#normalizers-compared) for what they do. Models are
compared with the paired bootstrap within each language. Languages are compared only
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

### Findings: Whisper-base and Whisper-small

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

### Error analysis: Whisper-small

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

### Larger models on a GPU

Whisper-large-v3-turbo and Whisper-large-v3 were transcribed in float16 on a Colab T4 GPU
([notebook](notebooks/e1_large_colab.ipynb)), together with a *bridge* run of Whisper-small on the
same GPU: comparing it with the CPU int8 run of the same model measures the effect of hardware and
precision alone, so that the model comparisons below are not confounded with it.

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

<!-- BEGIN e1-gpu-gaps -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| Italian: small, CPU int8 → GPU float16 | 7.46% | 7.43% | -0.03% [-0.26%, +0.21%] | 61.1% | no |
| Italian: small → large-v3-turbo | 7.43% | 2.64% | -4.79% [-5.30%, -4.31%] | 100.0% | yes |
| Italian: large-v3-turbo → large-v3 | 2.64% | 2.26% | -0.38% [-0.59%, -0.15%] | 100.0% | yes |
| German: small, CPU int8 → GPU float16 | 9.31% | 9.13% | -0.18% [-0.37%, +0.01%] | 97.0% | no |
| German: small → large-v3-turbo | 9.13% | 4.45% | -4.68% [-5.26%, -4.15%] | 100.0% | yes |
| German: large-v3-turbo → large-v3 | 4.45% | 4.02% | -0.43% [-0.75%, -0.14%] | 100.0% | yes |
| English: small, CPU int8 → GPU float16 | 6.66% | 6.59% | -0.06% [-0.23%, +0.10%] | 73.8% | no |
| English: small → large-v3-turbo | 6.59% | 5.00% | -1.59% [-1.99%, -1.19%] | 100.0% | yes |
| English: large-v3-turbo → large-v3 | 5.00% | 4.62% | -0.39% [-0.69%, -0.12%] | 99.8% | yes |
<!-- END e1-gpu-gaps -->

**Hardware and precision alone do not change the results.** The bridge comparisons (Whisper-small,
CPU int8 → GPU float16) give -0.03 points in Italian (95% CI [-0.26, +0.21]), -0.18 in German
([-0.37, +0.01]) and -0.06 in English ([-0.23, +0.10]): none is significant. Not significant does
not mean zero, but any effect is at most a few tenths of a point, an order of magnitude below the
model differences, so the GPU comparisons below are comparisons of models and the CPU and GPU
tables can be read side by side.

**The large models are much better, and the gain again depends on the language.** From
Whisper-small to large-v3-turbo, WER drops by 4.79 points in Italian (65% of the errors), 4.68 in
German (51%) and 1.59 in English (24%). Over the whole range, from Whisper-small on the CPU to
large-v3, the reduction is 70%, 57% and 31%.

**The language ranking reverses.** Whisper-large-v3 reaches **2.26% in Italian** (95% CI
[1.93, 2.64]), **4.02% in German** ([3.52, 4.56]) and **4.62% in English** ([4.08, 5.17]): English,
the easiest language for the small models, is now the hardest, level with German. Part of this is
measurement rather than recognition. Substitutions involving digits or symbols stay at about the
same count for every model (Italian 60 for Whisper-small on the GPU and 55 for large-v3, German 50
and 49, English 54 and 53), so as recognition errors disappear, formatting takes a growing share:
in Italian, from 5% to 14% of the substitutions. In English, the most frequent substitutions of
large-v3 are British versus American spellings ("travellers" vs "travelers" 7 times, "travelling"
4, "theatre" 3), which `en-v1` counts as errors. With a simple normalizer, the better the model, the
more its WER measures writing conventions.

**large-v3-turbo is about twice as fast as large-v3 and slightly less accurate.** On the GPU it is
2.0 to 2.3 times faster; large-v3 is better by 0.38 points in Italian, 0.43 in German and 0.39 in
English (8% to 14% fewer errors), a significant difference in every language but a small one in
absolute terms.

| Real-time factor, T4 GPU, float16 | Italian | German | English | GPU minutes, 3 languages |
|---|---|---|---|---|
| Whisper-small | 0.023 | 0.024 | 0.025 | 12 |
| Whisper-large-v3-turbo | 0.034 | 0.037 | 0.044 | 19 |
| Whisper-large-v3 | 0.078 | 0.082 | 0.087 | 41 |

The turbo model has the encoder of large-v3 and 4 decoder layers instead of 32. The encoder does
much of the work, and utterances are transcribed one at a time, which adds a fixed cost per
utterance for every model: that is why it is not eight times faster, and why Whisper-small is only
3.4 times faster than large-v3. These factors were measured on a shared Colab machine and are not
comparable with the CPU ones in [Setup](#models).

**Larger models hallucinate more closing phrases in German.** Utterances whose transcript ends with
a phrase that is not in the reference, typical of subtitled videos in Whisper's training data
("Sottotitoli a cura di QTSS" and "grazie a tutti" in Italian, "Vielen Dank" and "Untertitelung des
ZDF 2020" in German, "thank you" in English):

<!-- BEGIN e1-closing-phrases -->
| Utterances with an invented closing phrase | small (CPU int8) | small | large-v3-turbo | large-v3 |
|---|---|---|---|---|
| Italian | 5 | 4 | 3 | 2 |
| German | 3 | 3 | 12 | 12 |
| English | 0 | 1 | 0 | 1 |
<!-- END e1-closing-phrases -->

In German, 10 of the 12 utterances are the same for large-v3-turbo and large-v3, which share their
encoder: something in those recordings triggers the phrase. Each case costs only two to six
insertions, so the effect on WER is small, but in a product an invented "Vielen Dank" at the end
of a dictated message is a visible error. Voice activity detection, disabled here, removes them
almost entirely ([E2](#e2-which-decoding-settings)). The counts are part of the error report
(`asr-lab errors`), which matches a short list of known phrases at the end of the normalized
transcript when the reference does not end the same way.

**Speaker gender** (Whisper-large-v3, descriptive as above): Italian 2.19% for female voices
([1.86, 2.56]) and 3.88% for the 38 male utterances ([2.03, 6.20]); English 4.40% ([3.77, 5.03])
and 5.01% ([4.15, 6.04]). The intervals overlap in both languages.

### Limitations

- Language is confounded with speaker gender, and speaker ids are not available.
- `de-v1` and `en-v1` handle neither numbers nor abbreviations, `de-v1` splits hyphenated
  compounds and `en-v1` keeps spelling variants. Comparing languages fairly needs normalizers of
  similar coverage.
- The test sets are small (1.8 to 3.5 hours) and contain read Wikipedia text only.
- No correction for multiple comparisons is applied to the twelve comparisons of this chapter.
  The nine model comparisons have the larger model better in at least 99.8% of the 1,000
  replicates; the one closest to a Bonferroni-corrected threshold is English large-v3-turbo →
  large-v3 (99.8%), which more resamples would be needed to settle.
- The large models ran on a different machine (Colab T4 GPU, float16). The bridge run shows that
  this does not change accuracy measurably, but speeds are not comparable across machines.
- Hallucinated closing phrases are counted by matching a short list of known phrases, so other
  invented endings are missed.

The other model families are compared in [E1b](#e1b-three-model-families-on-the-same-test-sets).

**Reproduce:** `bash experiments/E1/run.sh` (or `experiments/E1/run.ps1` on Windows). With the
committed transcripts it only rescores. Without them it downloads the three test sets and
transcribes them with Whisper-small and Whisper-base (about two hours on a laptop CPU); the GPU
runs are transcribed with [`notebooks/e1_large_colab.ipynb`](notebooks/e1_large_colab.ipynb) on a
Colab GPU (about 75 minutes of transcription on a T4, plus the download).

## E1b. Three model families on the same test sets

**Question.** How do the three families of end-to-end models compare on the same audio, normalizers
and statistics, in errors, in the kind of errors, and in speed?

**Method.** Models, revisions, test sets, normalizers and comparisons were fixed in
[`experiments/E1b/models.toml`](experiments/E1b/models.toml) before any model transcribed a test set
([decision 28](docs/design.md#28-other-model-families-through-transformers-with-the-training-data-of-each-model-on-record)).

| Family | Model | How it writes | Language | Trained on |
|---|---|---|---|---|
| Encoder-decoder | Whisper-small, large-v3-turbo (beam 5) | Decoder conditioned on everything written so far | Forced | Not published in detail |
| CTC | wav2vec2 XLS-R, one checkpoint per language (greedy) | One character distribution per frame, no memory of the text and no language model | Fixed by the checkpoint | Common Voice 6.1 |
| Transducer | Parakeet TDT 0.6B v3, one model for 25 languages (greedy) | A small predictor remembers the tokens; each step also predicts how many frames to skip | Detected by the model | Includes Europarl-ASR, Common Voice and MOSEL |

- **Test sets.** FLEURS in three languages is the primary result. VoxPopuli Italian is a second
  domain, and MOSEL is the unlabelled VoxPopuli audio with Whisper's transcripts, so Parakeet has
  probably heard the sessions of that test split: **its VoxPopuli results are reported as probably
  contaminated**, and nothing below concludes that it generalizes better there.
- **Scoring.** `it-v2` writes numbers in words, `de-v1` and `en-v1` do not. The CTC vocabularies have
  no digits, so every comparison is repeated on the utterances whose reference has no digit (81% of
  each FLEURS test set, 89% of VoxPopuli).
- **Statistics.** Corpus WER, the widest of the bootstrap intervals (1,000 resamples), paired
  bootstrap with Holm's correction over the five comparisons of each test set.
- **Runs.** One utterance at a time on a Colab T4, float16. Whisper's FLEURS runs come from E1;
  on VoxPopuli it uses the decoding selected in E2 (hallucination silence 2 s).

### Results

<!-- BEGIN e1b-wer -->
| Test set | Utterances / words | Model | WER [95% CI] | S / D / I | CER |
|---|---|---|---|---|---|
| fleurs-it (`it-v2`) | 865 / 21,262 | Whisper-small | 7.11% [6.46%, 7.75%] | 1,171 / 165 / 176 | 1.84% |
| fleurs-it (`it-v2`) | 865 / 21,262 | Whisper-large-v3-turbo | 2.40% [2.04%, 2.80%] | 404 / 44 / 62 | 0.65% |
| fleurs-it (`it-v2`) | 865 / 21,262 | XLS-R (CTC) | 7.36% [6.66%, 8.02%] | 1,133 / 253 / 178 | 1.67% |
| fleurs-it (`it-v2`) | 865 / 21,262 | Parakeet TDT | 2.76% [2.34%, 3.20%] | 439 / 90 / 58 | 0.83% |
| fleurs-de (`de-v1`) | 862 / 18,703 | Whisper-small | 9.13% [8.43%, 9.90%] | 1,303 / 168 / 237 | 2.54% |
| fleurs-de (`de-v1`) | 862 / 18,703 | Whisper-large-v3-turbo | 4.45% [3.91%, 5.01%] | 612 / 89 / 131 | 1.44% |
| fleurs-de (`de-v1`) | 862 / 18,703 | XLS-R (CTC) | 11.79% [10.84%, 12.85%] | 1,746 / 257 / 202 | 4.90% |
| fleurs-de (`de-v1`) | 862 / 18,703 | Parakeet TDT | 5.09% [4.52%, 5.72%] | 735 / 134 / 83 | 1.50% |
| fleurs-en (`en-v1`) | 647 / 14,467 | Whisper-small | 6.59% [5.93%, 7.27%] | 696 / 104 / 154 | 2.70% |
| fleurs-en (`en-v1`) | 647 / 14,467 | Whisper-large-v3-turbo | 5.00% [4.45%, 5.55%] | 517 / 110 / 97 | 2.13% |
| fleurs-en (`en-v1`) | 647 / 14,467 | XLS-R (CTC) | 14.54% [13.38%, 15.75%] | 1,611 / 252 / 241 | 6.68% |
| fleurs-en (`en-v1`) | 647 / 14,467 | Parakeet TDT | 5.45% [4.86%, 6.07%] | 586 / 143 / 59 | 2.32% |
| voxpopuli-it (`it-v2`) | 1,177 / 37,454 | Whisper-small | 20.63% [18.33%, 23.24%] | 3,060 / 2,732 / 1,933 | 13.96% |
| voxpopuli-it (`it-v2`) | 1,177 / 37,454 | Whisper-large-v3-turbo | 19.11% [16.12%, 22.18%] | 1,782 / 3,964 / 1,412 | 15.59% |
| voxpopuli-it (`it-v2`) | 1,177 / 37,454 | XLS-R (CTC) | 26.19% [24.07%, 28.68%] | 4,502 / 3,149 / 2,160 | 14.91% |
| voxpopuli-it (`it-v2`) | 1,177 / 37,454 | Parakeet TDT | 20.51% [13.96%, 30.89%] | 1,958 / 2,579 / 3,144 | 14.61% |
<!-- END e1b-wer -->

Comparisons, B - A, negative when B has fewer errors:

<!-- BEGIN e1b-comparisons -->
| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| fleurs-it | XLS-R (CTC) - Whisper-small | +0.24% [-0.46%, +0.97%] | 0.494 | 0.494 | no |
| fleurs-it | XLS-R (CTC) - Whisper-large-v3-turbo | +4.96% [+4.43%, +5.49%] | < 0.001 | < 0.005 | yes |
| fleurs-it | Parakeet TDT - Whisper-small | -4.35% [-4.88%, -3.82%] | < 0.001 | < 0.005 | yes |
| fleurs-it | Parakeet TDT - Whisper-large-v3-turbo | +0.36% [+0.09%, +0.66%] | 0.016 | 0.032 | yes |
| fleurs-it | Parakeet TDT - XLS-R (CTC) | -4.60% [-5.12%, -4.08%] | < 0.001 | < 0.005 | yes |
| fleurs-de | XLS-R (CTC) - Whisper-small | +2.66% [+1.80%, +3.48%] | < 0.001 | < 0.005 | yes |
| fleurs-de | XLS-R (CTC) - Whisper-large-v3-turbo | +7.34% [+6.52%, +8.18%] | < 0.001 | < 0.005 | yes |
| fleurs-de | Parakeet TDT - Whisper-small | -4.04% [-4.72%, -3.44%] | < 0.001 | < 0.005 | yes |
| fleurs-de | Parakeet TDT - Whisper-large-v3-turbo | +0.64% [+0.23%, +1.07%] | 0.002 | 0.002 | yes |
| fleurs-de | Parakeet TDT - XLS-R (CTC) | -6.70% [-7.49%, -6.00%] | < 0.001 | < 0.005 | yes |
| fleurs-en | XLS-R (CTC) - Whisper-small | +7.95% [+6.91%, +9.03%] | < 0.001 | < 0.005 | yes |
| fleurs-en | XLS-R (CTC) - Whisper-large-v3-turbo | +9.54% [+8.50%, +10.64%] | < 0.001 | < 0.005 | yes |
| fleurs-en | Parakeet TDT - Whisper-small | -1.15% [-1.63%, -0.64%] | < 0.001 | < 0.005 | yes |
| fleurs-en | Parakeet TDT - Whisper-large-v3-turbo | +0.44% [+0.06%, +0.84%] | 0.032 | 0.032 | yes |
| fleurs-en | Parakeet TDT - XLS-R (CTC) | -9.10% [-10.12%, -8.07%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | XLS-R (CTC) - Whisper-small | +5.57% [+5.02%, +6.17%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | XLS-R (CTC) - Whisper-large-v3-turbo | +7.08% [+5.54%, +8.71%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | Parakeet TDT - Whisper-small | -0.12% [-5.61%, +9.72%] | 0.832 | 1.000 | no |
| voxpopuli-it | Parakeet TDT - Whisper-large-v3-turbo | +1.40% [-4.28%, +11.67%] | 0.822 | 1.000 | no |
| voxpopuli-it | Parakeet TDT - XLS-R (CTC) | -5.69% [-11.08%, +4.22%] | 0.170 | 0.510 | no |
<!-- END e1b-comparisons -->

Speed. RTF is processing time over audio duration, so lower is faster:

<!-- BEGIN e1b-speed -->
| Model | Parameters | RTF fleurs-it | RTF fleurs-de | RTF fleurs-en | RTF voxpopuli-it |
|---|---|---|---|---|---|
| Whisper-small | 242 M | 0.0229 | 0.0241 | 0.0251 | 0.0339 |
| Whisper-large-v3-turbo | 809 M | 0.0336 | 0.0369 | 0.0439 | 0.0415 |
| XLS-R (CTC) | 315 M | 0.0048 | 0.0047 | 0.0042 | 0.0048 |
| Parakeet TDT | 627 M | 0.0122 | 0.0134 | 0.0150 | 0.0150 |
<!-- END e1b-speed -->

### Without the references with digits

<!-- BEGIN e1b-no-digit -->
| Model | fleurs-it | fleurs-de | fleurs-en | voxpopuli-it |
|---|---|---|---|---|
| Whisper-small | 7.05% [6.31%, 7.76%] | 8.44% [7.64%, 9.27%] | 5.95% [5.24%, 6.66%] | 21.64% [18.80%, 24.73%] |
| Whisper-large-v3-turbo | 2.24% [1.84%, 2.66%] | 3.92% [3.34%, 4.55%] | 4.53% [3.95%, 5.13%] | 20.46% [16.99%, 24.25%] |
| XLS-R (CTC) | 6.33% [5.71%, 6.97%] | 9.52% [8.68%, 10.37%] | 11.43% [10.52%, 12.41%] | 26.95% [24.21%, 29.83%] |
| Parakeet TDT | 2.64% [2.22%, 3.13%] | 4.20% [3.64%, 4.80%] | 4.95% [4.26%, 5.59%] | 22.43% [15.14%, 35.12%] |
| *utterances kept* | 703 of 865 | 698 of 862 | 525 of 647 | 1,043 of 1,177 |

| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| fleurs-it | XLS-R (CTC) - Whisper-small | -0.72% [-1.42%, -0.03%] | 0.040 | 0.040 | yes |
| fleurs-it | XLS-R (CTC) - Whisper-large-v3-turbo | +4.09% [+3.56%, +4.65%] | < 0.001 | < 0.005 | yes |
| fleurs-it | Parakeet TDT - Whisper-small | -4.40% [-4.98%, -3.79%] | < 0.001 | < 0.005 | yes |
| fleurs-it | Parakeet TDT - Whisper-large-v3-turbo | +0.40% [+0.11%, +0.71%] | 0.004 | 0.008 | yes |
| fleurs-it | Parakeet TDT - XLS-R (CTC) | -3.68% [-4.21%, -3.18%] | < 0.001 | < 0.005 | yes |
| fleurs-de | XLS-R (CTC) - Whisper-small | +1.08% [+0.35%, +1.81%] | 0.006 | 0.012 | yes |
| fleurs-de | XLS-R (CTC) - Whisper-large-v3-turbo | +5.60% [+4.92%, +6.27%] | < 0.001 | < 0.005 | yes |
| fleurs-de | Parakeet TDT - Whisper-small | -4.24% [-4.94%, -3.54%] | < 0.001 | < 0.005 | yes |
| fleurs-de | Parakeet TDT - Whisper-large-v3-turbo | +0.28% [-0.12%, +0.67%] | 0.142 | 0.142 | no |
| fleurs-de | Parakeet TDT - XLS-R (CTC) | -5.32% [-5.98%, -4.62%] | < 0.001 | < 0.005 | yes |
| fleurs-en | XLS-R (CTC) - Whisper-small | +5.48% [+4.61%, +6.30%] | < 0.001 | < 0.005 | yes |
| fleurs-en | XLS-R (CTC) - Whisper-large-v3-turbo | +6.90% [+6.13%, +7.71%] | < 0.001 | < 0.005 | yes |
| fleurs-en | Parakeet TDT - Whisper-small | -1.00% [-1.53%, -0.49%] | < 0.001 | < 0.005 | yes |
| fleurs-en | Parakeet TDT - Whisper-large-v3-turbo | +0.43% [-0.02%, +0.85%] | 0.058 | 0.058 | no |
| fleurs-en | Parakeet TDT - XLS-R (CTC) | -6.48% [-7.29%, -5.76%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | XLS-R (CTC) - Whisper-small | +5.31% [+4.71%, +5.90%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | XLS-R (CTC) - Whisper-large-v3-turbo | +6.49% [+4.83%, +8.10%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | Parakeet TDT - Whisper-small | +0.79% [-5.50%, +13.05%] | 0.934 | 1.000 | no |
| voxpopuli-it | Parakeet TDT - Whisper-large-v3-turbo | +1.97% [-4.83%, +14.50%] | 0.740 | 1.000 | no |
| voxpopuli-it | Parakeet TDT - XLS-R (CTC) | -4.52% [-10.83%, +7.59%] | 0.336 | 1.000 | no |
<!-- END e1b-no-digit -->

### Two runaway transcripts (post hoc)

Parakeet's interval on VoxPopuli is several times wider than the others', and the reason is in the
longest transcripts: two utterances received 1,588 words for 12.7 seconds of audio ("È una cosa che
è una cosa che...") and 355 for 5.7 seconds ("Il n'y a rien d'un colour dix, non, non, non..."). The
next fastest transcript of any model in any run is at 5.6 words per second; these two are at 125 and
63. No other model, on any test set, produced anything similar.

They reproduce locally, in float32 on a CPU, so neither half precision nor the GPU causes them. The
cause is in the decoding loop of Transformers (versions 5.20.2 on Colab and 5.21.1 locally). A TDT
step moves through the audio by the duration it predicts. The RNN-T path of the same library limits
the tokens written at one frame to `max_symbols_per_step`, but the TDT path skips that count and
only forces an advance after a blank. A model that keeps predicting a word with a duration of zero
therefore stays on the same frame until the length limit, ten tokens per encoder frame.

With that limit put back, in a one-off check on these two utterances only, they become 38 and 21
words: "È una cosa che è una cosa che è una organizzata e non organizzata, che sfrutta e continua a
sfruttare i migranti..." and "Il n'y a rien d'un colour dix, non, non, non può che essere la
posizione espressa in prima lettura dal Parlamento." The model still starts wrongly, in the second
case in French, but it recovers. The fix was not applied to the runs, which stay as pre-registered.

The table excludes these two utterances for every model, so that the comparisons stay paired. This
is an analysis chosen after seeing the data, with its own family of comparisons:

<!-- BEGIN e1b-runaway -->
| Test set | Runaway transcripts: words written / reference words / seconds | Model | WER with them | WER without them [95% CI] |
|---|---|---|---|---|
| voxpopuli-it | 355 / 16 / 5.7; 1,588 / 28 / 12.7 | Whisper-small | 20.63% | 20.61% [18.33%, 23.23%] |
| voxpopuli-it | 355 / 16 / 5.7; 1,588 / 28 / 12.7 | Whisper-large-v3-turbo | 19.11% | 19.10% [16.11%, 22.17%] |
| voxpopuli-it | 355 / 16 / 5.7; 1,588 / 28 / 12.7 | XLS-R (CTC) | 26.19% | 26.18% [24.05%, 28.67%] |
| voxpopuli-it | 355 / 16 / 5.7; 1,588 / 28 / 12.7 | Parakeet TDT | 20.51% | 15.35% [13.26%, 17.76%] |

| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| voxpopuli-it | XLS-R (CTC) - Whisper-small | +5.57% [+5.01%, +6.17%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | XLS-R (CTC) - Whisper-large-v3-turbo | +7.08% [+5.53%, +8.73%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | Parakeet TDT - Whisper-small | -5.26% [-6.07%, -4.53%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | Parakeet TDT - Whisper-large-v3-turbo | -3.75% [-5.10%, -2.36%] | < 0.001 | < 0.005 | yes |
| voxpopuli-it | Parakeet TDT - XLS-R (CTC) | -10.83% [-11.74%, -10.05%] | < 0.001 | < 0.005 | yes |
<!-- END e1b-runaway -->

### What the errors look like (post hoc)

<!-- BEGIN e1b-errors -->
| Test set | Model | Errors | At the segment edges | Invented closing phrases | Empty transcripts |
|---|---|---|---|---|---|
| fleurs-it | Whisper-small | 1,512 | 223 (15%) | 4 | 0 |
| fleurs-it | Whisper-large-v3-turbo | 510 | 68 (13%) | 3 | 0 |
| fleurs-it | XLS-R (CTC) | 1,564 | 166 (11%) | 0 | 0 |
| fleurs-it | Parakeet TDT | 587 | 69 (12%) | 0 | 0 |
| fleurs-de | Whisper-small | 1,708 | 250 (15%) | 3 | 0 |
| fleurs-de | Whisper-large-v3-turbo | 832 | 132 (16%) | 12 | 0 |
| fleurs-de | XLS-R (CTC) | 2,205 | 266 (12%) | 0 | 0 |
| fleurs-de | Parakeet TDT | 952 | 125 (13%) | 0 | 0 |
| fleurs-en | Whisper-small | 954 | 150 (16%) | 1 | 0 |
| fleurs-en | Whisper-large-v3-turbo | 724 | 108 (15%) | 0 | 0 |
| fleurs-en | XLS-R (CTC) | 2,104 | 348 (17%) | 0 | 0 |
| fleurs-en | Parakeet TDT | 788 | 116 (15%) | 0 | 0 |
| voxpopuli-it | Whisper-small | 7,725 | 4,408 (57%) | 22 | 3 |
| voxpopuli-it | Whisper-large-v3-turbo | 7,158 | 5,226 (73%) | 70 | 0 |
| voxpopuli-it | XLS-R (CTC) | 9,811 | 4,903 (50%) | 11 | 0 |
| voxpopuli-it | Parakeet TDT | 7,681 | 5,341 (70%) | 2 | 38 |
<!-- END e1b-errors -->

Parakeet returned an empty transcript for 38 VoxPopuli utterances. The others do not do much better
on them, which suggests hard audio or wrong segment boundaries rather than speech that only Parakeet
misses; these utterances were not listened to:

<!-- BEGIN e1b-silent -->
| Test set | Utterances / words / mean duration | Model | Words right | Errors | Inserted words | Invented closing phrases |
|---|---|---|---|---|---|---|
| voxpopuli-it | 38 / 348 / 3.3 s | Whisper-small | 111 | 291 | 54 | 1 |
| voxpopuli-it | 38 / 348 / 3.3 s | Whisper-large-v3-turbo | 113 | 280 | 45 | 10 |
| voxpopuli-it | 38 / 348 / 3.3 s | XLS-R (CTC) | 108 | 308 | 68 | 1 |
| voxpopuli-it | 38 / 348 / 3.3 s | Parakeet TDT | 0 | 348 | 0 | 0 |
<!-- END e1b-silent -->

### Findings

- **Parakeet is close to large-v3-turbo and much faster.** On FLEURS it is significantly behind in
  all three languages, by 0.36 points in Italian ([+0.09, +0.66]), 0.64 in German ([+0.23, +1.07])
  and 0.44 in English ([+0.06, +0.84]), and significantly ahead of Whisper-small by 1.15 to 4.35
  points. It takes about 2.8 times less time than turbo, with 627 M parameters against 809 M. Two of
  the three gaps lose significance without the references with digits (German +0.28 [-0.12, +0.67],
  English +0.43 [-0.02, +0.85]), while Italian, where the normalizer already handles numbers, stays
  significant (+0.40 [+0.11, +0.71]). These runs compare the implementations as used here, not the
  architectures: batching and other runtimes would change the speeds.
- **The CTC model's weakness depends on the language, and only part of it is notation.** It is the
  fastest by far: 5 to 6 times Whisper-small with a similar number of parameters (315 M against
  242 M). In Italian it does not differ from Whisper-small (+0.24 [-0.46, +0.97]), and without digits
  it is slightly better (-0.72 [-1.42, -0.03]). In English it is far behind (+7.95 [+6.91, +9.03]).
  Removing digits narrows that gap to +5.48, so most of it remains. A model without a memory of the
  text writes what it hears, as in "cabbige", "hostitle" and "protocal". The next experiment adds a
  language model, which addresses this directly.
- **Word error rate hides how near the errors are.** Those misspellings cost a whole word and a few
  characters. In Italian, XLS-R has a higher WER than Whisper-small (7.36% against 7.11%) and a lower
  CER (1.67% against 1.84%).
- **The families fail differently.** In German, large-v3-turbo invents 12 closing phrases on FLEURS
  and the CTC and Transducer models none. On VoxPopuli, the phrases missing from the references are
  70 for turbo, 22 for small, 11 for CTC and 2 for Parakeet. Some of these may be real speech outside
  the transcribed segment, since the CTC count cannot come from a subtitle habit, but the size of the
  gap fits the collapse seen in E6. Parakeet's failures are different: silence on 38 utterances and
  repetition on two.
- **A single WER can be misleading in both directions.** As measured, Parakeet on VoxPopuli is
  indistinguishable from Whisper (20.51% [13.96, 30.89]). Two utterances out of 1,177 hold 25% of its
  errors. Without them it is at 15.35% [13.26, 17.76], ahead of turbo by 3.75 points
  ([-5.10, -2.36]), but that advantage is post hoc and comes on sessions it has probably heard, so it
  shows what the failure hides rather than how good the model is.
- **Segment edges matter on VoxPopuli whichever the family.** 50% to 73% of the errors are at the
  edges of the segments, against 11% to 17% on FLEURS, which extends the finding of E7 to two other
  families: much of the VoxPopuli error rate belongs to the segmentation of the dataset.

For a product, these results point to the safeguard more than to a model: whichever model is chosen,
check transcript length against audio duration. A simple words-per-second limit catches both runaway
transcripts here, with no false alarm on any of the 16 runs.

### Limitations

- The comparison uses each model as distributed. Training data differ between the families, and
  they are part of what is measured.
- Parakeet's VoxPopuli results are probably contaminated, and its FLEURS results may not be free of
  overlap with its training data, which include Wikipedia-derived sentences in some corpora; this
  was not verified.
- Speeds come from one utterance at a time on one T4, in different sessions and runtimes (CTranslate2
  for Whisper, PyTorch for the others). They show orders of magnitude, not what a batched deployment
  would give.
- The XLS-R checkpoints use greedy decoding with no language model, by design of this experiment, so
  this is not the best a CTC system can do.
- The runaway limit and the analyses of errors were chosen after seeing the data. The threshold of
  20 words per second has a wide margin in these runs, but it is untested elsewhere. The check with
  the symbol limit covers the two utterances, not the whole test sets.
- The 38 empty transcripts were not listened to, and the closing phrases are matched against a short
  list, so other invented endings are missed.
- `de-v1` and `en-v1` do not normalize numbers, so the full-set comparisons in those languages include
  differences of notation; the subset without digits is the fairer reading, on 81% of the FLEURS utterances.
- **The notebook was not run as committed at the time.** With the GPU setup for Whisper before the
  PyTorch models, the session crashed during Parakeet on a T4, so the Transducer and CTC runs were
  done without that setup and Whisper afterwards in the same session, with the same commit and
  configurations. The committed notebook now has this order. The CUDA 12 libraries are the likely
  cause, but this was not isolated, and a similar setup ran the fine-tuning of E7 without problems.

**Reproduce:** `bash experiments/E1b/run.sh` (or `experiments/E1b/run.ps1` on Windows). With the
committed transcripts it only rescores, after downloading the manifests if they are missing. The
transcriptions run in [`notebooks/e1b_families_colab.ipynb`](notebooks/e1b_families_colab.ipynb)
(about 35 minutes of transcription on a T4, plus the downloads).

## E2. Which decoding settings?

**Question.** E1 used fixed decoding settings. Do the usual alternatives (a smaller or larger
beam, no temperature fallback, voice activity detection, Whisper's hallucination-silence
threshold) make large-v3-turbo more accurate or faster, and which settings should a product use?

**Method.**
- **One factor at a time.** Each configuration changes one option of the E1 settings (beam size 5,
  temperature fallback on, no voice activity detection), listed in
  [`experiments/E2/sweep.toml`](experiments/E2/sweep.toml).
- **Tuned on dev, confirmed on test.** Every configuration transcribes the three FLEURS dev sets
  (Italian 391 utterances / 9,018 words, German 363 / 7,410, English 394 / 8,241; no sentence is
  shared with the test sets), in one Colab session on a T4 GPU, with the model loaded once and a
  warm-up before timing. The baseline configuration is run a second time at the end of the session
  (`baseline-repeat`): this measures run-to-run variation, both in the transcripts and in the
  timings.
- **One configuration for all languages, chosen by a rule fixed before the runs**
  (`asr-lab select`, [decision 22](docs/design.md#22-decoding-options-are-chosen-on-the-dev-set-by-a-rule-written-in-code)).
  In each language, find the configuration with the lowest WER. A configuration is eligible if a
  paired bootstrap does not find it significantly worse than that best one in any language. Among
  the eligible ones, pick the fastest, by total processing time over the three dev sets. If none
  is eligible, keep the baseline.
- **The test sets are used once**, for the selected configuration only, against the E1 run of
  large-v3-turbo, which has the baseline settings. Voice activity detection was also run on the
  test sets, as an **exploratory** comparison decided after seeing the dev results (see below).

### Results on the dev sets

The selection, as computed by `asr-lab select` (best WER of each language in bold; times are the
total over the three dev sets):

<!-- BEGIN e2-selection -->
| Configuration | Change from the baseline | WER it | WER de | WER en | RTF | Time vs baseline | Eligible |
|---|---|---|---|---|---|---|---|
| baseline | none | 2.37% | 5.16% | **5.33%** | 0.037 | +0.0% | no |
| beam1 | `beam_size = 1` | 2.36% | 5.38% | 5.47% | 0.033 | -12.6% | no |
| beam2 | `beam_size = 2` | 2.34% | 5.30% | 5.36% | 0.034 | -8.8% | no |
| beam8 | `beam_size = 8` | 2.38% | 5.14% | 5.35% | 0.042 | +13.1% | no |
| no-fallback | `temperature = [0.0]` | 2.37% | 5.16% | 5.33% | 0.038 | +0.7% | no |
| vad | `vad_filter = True` | **2.30%** | **4.84%** | 5.41% | 0.040 | +6.2% | yes |
| **hst2** (selected) | `hallucination_silence_threshold = 2.0` | 2.37% | 5.13% | 5.33% | 0.038 | +2.3% | yes |
| baseline-repeat | none | 2.37% | 5.16% | 5.33% | 0.038 | +0.6% | n/a |
<!-- END e2-selection -->

![WER against processing time on the dev sets, one panel per language](reports/e2_pareto.png)

*Each point is a configuration; the dashed line joins the configurations that no other one beats
on both WER and time (the Pareto front); the selected one is blue. The WER axis spans the same
number of points in every panel.*

Effect of each option against the baseline (B - A, negative when the option lowers WER):

<!-- BEGIN e2-dev-effects -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| Italian: beam1 | 2.37% | 2.36% | -0.01% [-0.16%, +0.13%] | 55.0% | no |
| Italian: beam2 | 2.37% | 2.34% | -0.03% [-0.16%, +0.07%] | 67.4% | no |
| Italian: beam8 | 2.37% | 2.38% | +0.01% [+0.00%, +0.04%] | 0.0% | no |
| Italian: no-fallback | 2.37% | 2.37% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| Italian: vad | 2.37% | 2.30% | -0.08% [-0.25%, +0.10%] | 78.6% | no |
| Italian: hst2 | 2.37% | 2.37% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| Italian: baseline-repeat | 2.37% | 2.37% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| German: beam1 | 5.16% | 5.38% | +0.23% [+0.00%, +0.48%] | 2.0% | no |
| German: beam2 | 5.16% | 5.30% | +0.15% [-0.01%, +0.35%] | 3.5% | no |
| German: beam8 | 5.16% | 5.14% | -0.01% [-0.10%, +0.07%] | 56.6% | no |
| German: no-fallback | 5.16% | 5.16% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| German: vad | 5.16% | 4.84% | -0.31% [-0.59%, -0.03%] | 98.3% | yes |
| German: hst2 | 5.16% | 5.13% | -0.03% [-0.07%, +0.00%] | 86.3% | no |
| German: baseline-repeat | 5.16% | 5.16% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| English: beam1 | 5.33% | 5.47% | +0.15% [+0.01%, +0.30%] | 1.8% | yes |
| English: beam2 | 5.33% | 5.36% | +0.04% [-0.09%, +0.17%] | 26.6% | no |
| English: beam8 | 5.33% | 5.35% | +0.02% [-0.06%, +0.12%] | 23.9% | no |
| English: no-fallback | 5.33% | 5.33% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| English: vad | 5.33% | 5.41% | +0.08% [-0.13%, +0.30%] | 21.2% | no |
| English: hst2 | 5.33% | 5.33% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
| English: baseline-repeat | 5.33% | 5.33% | +0.00% [+0.00%, +0.00%] | 0.0% | no |
<!-- END e2-dev-effects -->

Utterances with an invented closing phrase (see [E1](#larger-models-on-a-gpu)):

<!-- BEGIN e2-closing-phrases-dev -->
| Utterances with an invented closing phrase | baseline | beam1 | beam2 | beam8 | no-fallback | vad | hst2 | baseline-repeat |
|---|---|---|---|---|---|---|---|---|
| Italian (dev) | 2 | 2 | 2 | 2 | 2 | 1 | 3 | 2 |
| German (dev) | 2 | 2 | 2 | 3 | 2 | 0 | 2 | 2 |
| English (dev) | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
<!-- END e2-closing-phrases-dev -->

### Results on the test sets

The E1 run of large-v3-turbo (baseline settings) against the selected configuration and against
voice activity detection:

<!-- BEGIN e2-test -->
| Comparison | A | B | B - A [95% CI] | B better in | Significant |
|---|---|---|---|---|---|
| Italian: baseline → hst2 (selected) | 2.64% | 2.64% | +0.00% [-0.03%, +0.03%] | 35.4% | no |
| Italian: baseline → vad (exploratory) | 2.64% | 2.70% | +0.06% [-0.09%, +0.19%] | 21.7% | no |
| German: baseline → hst2 (selected) | 4.45% | 4.46% | +0.01% [-0.01%, +0.04%] | 14.8% | no |
| German: baseline → vad (exploratory) | 4.45% | 4.25% | -0.20% [-0.41%, +0.01%] | 96.6% | no |
| English: baseline → hst2 (selected) | 5.00% | 5.02% | +0.01% [+0.00%, +0.04%] | 0.0% | no |
| English: baseline → vad (exploratory) | 5.00% | 4.94% | -0.07% [-0.24%, +0.11%] | 77.0% | no |
<!-- END e2-test -->

<!-- BEGIN e2-closing-phrases-test -->
| Utterances with an invented closing phrase | baseline (E1) | hst2 (selected) | vad (exploratory) |
|---|---|---|---|
| Italian (test) | 3 | 2 | 2 |
| German (test) | 12 | 13 | 0 |
| English (test) | 0 | 0 | 0 |
<!-- END e2-closing-phrases-test -->

### Findings

1. **Decoding is deterministic, and the temperature fallback never triggers on this data.**
   `baseline-repeat` produces the same transcripts as `baseline` in all 1,148 dev utterances, and
   so does `no-fallback`: the fallback re-decodes a segment at a higher temperature only when the
   output looks wrong (too repetitive or too unlikely), and on short read sentences it never did.
   Timings vary by about 1% between identical runs, so smaller speed differences are noise.
2. **The beam size changes speed more than accuracy.** Greedy decoding (`beam1`) is 12.6% faster
   than beam 5 and worse in German (+0.23 points) and English (+0.15, the only beam effect
   significant before correction). A beam of 8 is 13.1% slower and changes nothing. Beam 5, the
   default, is a reasonable setting for this model.
3. **Voice activity detection is the only option with a clear effect on accuracy.** It lowers WER
   in German by 0.31 points on dev (95% CI [-0.59, -0.03]), mostly by removing insertions and
   substitutions (S/D/I from 300/50/32 to 285/48/26), and removes almost every invented closing
   phrase: on dev, 5 to 1 over the three languages; on the test sets, the 12 German ones of E1
   disappear. It changes about a quarter of the transcripts (100, 86 and 79 dev utterances) and
   costs 6.2% more processing time.
4. **The selected configuration, `hst2`, is in practice the baseline.** The hallucination-silence
   threshold acts only on silences longer than 2 seconds around a suspicious segment, which short
   read utterances rarely contain: it changes 2 Italian and 2 German dev transcripts and none in
   English. On the test sets it is indistinguishable from E1 (+0.00, +0.01 and +0.01 points, none
   significant) and keeps the German closing phrases (13 against 12).
5. **The selection was decided by a borderline comparison.** Only `vad` and `hst2` are eligible.
   The baseline is excluded because it is significantly worse than `vad` in German (+0.31,
   [+0.03, +0.59]), while `hst2`, which differs from it in two German transcripts, is not
   (+0.28, [-0.01, +0.56]). The rule then picks `hst2` as the faster of the two (+2.3% against
   +6.2%). The rule works as written, but "not significantly worse" is not "equivalent": a
   configuration whose interval reaches +0.56 points is not shown to be as good. A rule with a
   non-inferiority margin (eligible only if the upper bound of the difference stays below a fixed
   margin) would make this explicit: with a margin of half a point it would have excluded `hst2`
   and the baseline (+0.56 and +0.59 in German) and kept `vad` (+0.30, in English), while a margin
   of 0.2 points would have excluded every configuration. Choosing the margin now, after seeing
   these numbers, would choose the result, so it is left for the next sweep, fixed in advance.
6. **On the test sets, voice activity detection confirms its direction in German but not
   significantly.** As an exploratory comparison, it gives -0.20 points in German ([-0.41, +0.01],
   lower in 96.6% of the replicates), -0.07 in English and +0.06 in Italian, none significant.
   It again removes insertions (German 131 to 92) but adds deletions (Italian 52 to 65, German 89
   to 101); among the deleted words are numbers (German, 5 to 14 deleted words with digits), as if
   short segments were cut. This pattern does not appear on dev and is based on few words, so it
   is an observation to check, not a finding.

**Interpretation.** For large-v3-turbo on read speech, the decoding settings matter little for WER:
apart from voice activity detection, no option moves it by more than a quarter of a point. The
practical choice is between the baseline and voice activity detection, and it is about
hallucinations rather than WER: VAD removes the invented closing phrases, which are rare but very
visible to a user, at the cost of about 6% of processing time and possibly a few deleted words.
For a dictation product we would adopt VAD and monitor deletions; the pre-registered rule, as
written, selected `hst2`, and that is the result reported here.

### Limitations

- **Multiple comparisons.** The dev table has 21 comparisons (18 options and 3 repeats). The two
  significant ones (`vad` in German and `beam1` in English) have B better in 98.3% and 1.8% of the
  replicates, which a Bonferroni correction for 21 comparisons would not accept. The selection
  rule itself compares every configuration with the best one of each language without
  correction; a correction would make more configurations eligible, not fewer.
- **Winner's curse.** The best configuration of each language on dev is partly best by chance, so
  its dev WER is optimistic; the test runs, used once, are the unbiased estimate.
- **The VAD test run is exploratory.** It was decided after seeing the dev results and is not
  covered by the selection rule. Its speed was measured in a different Colab session from the E1
  run it is compared with (about 5-8% slower, in line with the +6.2% on dev), so test speeds are
  only indicative.
- **Read speech only.** FLEURS utterances are short (test medians 9.5 to 13.9 seconds), with
  little silence or noise. Voice activity detection and the hallucination-silence threshold are
  designed for long recordings with pauses, where their effect is likely to be larger.
- Options were varied one at a time; their interactions (for example VAD with a smaller beam) were
  not tested.

**Reproduce:** `bash experiments/E2/run.sh` (or `experiments/E2/run.ps1` on Windows). With the
committed transcripts it only rescores, compares, selects and redraws the figure (the figure needs
`uv sync --extra plots`). The transcripts are produced with
[`notebooks/e2_decoding_colab.ipynb`](notebooks/e2_decoding_colab.ipynb) on a Colab GPU (about two
hours on a T4 for the dev sweep, plus the download).

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

## E6. Which training transcripts can be trusted?

**Question.** Before a sample of training data is used for fine-tuning, which utterances are
broken: transcripts that do not match the audio, empty transcripts, distorted audio, duplicates?
And can cheap, interpretable features find them without listening to everything?

**Method** ([decision 26](docs/design.md#26-training-data-quality-cheap-features-pre-registered-rules-clusters-to-read)).
- **Data.** Italian VoxPopuli: European Parliament speeches with the parliament's own
  transcripts (CC0), the planned domain of the fine-tuning experiment (E7). The sample is the
  first row group of three training shards, read from the Hub's parquet files at a pinned revision
  ([`experiments/E6/quality.toml`](experiments/E6/quality.toml)): 3,000 utterances, 10.21 hours,
  128 speakers, 840 plenary sessions. Nothing is filtered at ingestion.
- **Features** (`asr_lab.data.quality`): duration; speaking rate (letters of the normalized
  transcript per second); a blind SNR estimate (energy of the loudest against the quietest 25 ms
  frames); clipping; repeated transcripts or audio; and the WER of a strong model against each
  transcript, with the length of the model's transcript relative to it. The normalizer is
  `it-v2`, so that "2" against "due" is not counted as label noise.
- **Model.** Whisper-large-v3-turbo with the decoding settings selected in E2 (`hst2`), float16
  on a Colab T4 GPU (27 minutes, real-time factor 0.045).
- **Rules.** Thresholds written in the spec before any feature was computed: duration 1 to 30
  seconds, speaking rate 5 to 25 letters per second, estimated SNR at least 10 dB, at most 0.1%
  clipped samples, and WER of the strong model at most 50%. Each rule is counted on its own and as
  a union.
- **Clusters** (`asr_lab.data.clustering`). k-means on five standardized features (duration and
  WER on a log scale, speaking rate, SNR, length ratio), the number of clusters chosen by the
  silhouette score among 2 to 8, each cluster described by the features it stands out on.
- **Post hoc** (added after reading the high-WER transcripts, see below): the utterances flagged
  by the WER rule are split by the length of the model's transcript, ten of them were listened to,
  and Whisper-small transcribed the collapsed ones on a CPU as a second opinion. These steps
  describe the flagged utterances; they change no rule and no threshold.

<!-- BEGIN e6-wer -->
| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |
|---|---|---|---|---|---|
| Italian VoxPopuli sample · Whisper-large-v3-turbo | `it-v2` | 3,000 / 90,999 | 17.58% [15.95%, 19.18%] | 3,314 / 9,613 / 3,075 | 15.06% [13.47%, 16.63%] |
<!-- END e6-wer -->

The corpus WER is high for a model that reaches 2.64% on FLEURS Italian read speech (E1, with
`it-v1`), and it is driven by
deletions: 9,613 of 16,002 errors. The median utterance has 7.1% WER; a minority of utterances
with almost every word missing carries the total.

### What the rules flag

<!-- BEGIN e6-flags -->
| Rule | Utterances flagged | Hours flagged |
|---|---|---|
| `empty_text` | 23 (0.8%) | 0.07 (0.7%) |
| `too_short` | 10 (0.3%) | 0.00 (0.0%) |
| `too_long` | 48 (1.6%) | 0.48 (4.7%) |
| `too_slow` | 6 (0.2%) | 0.02 (0.2%) |
| `too_fast` | 4 (0.1%) | 0.00 (0.0%) |
| `low_snr` | 2 (0.1%) | 0.00 (0.0%) |
| `clipped` | 38 (1.3%) | 0.11 (1.1%) |
| `high_wer` | 372 (12.4%) | 0.97 (9.5%) |
| `duplicate_text` | 0 (0.0%) | 0.00 (0.0%) |
| `duplicate_audio` | 0 (0.0%) | 0.00 (0.0%) |
| **any rule** | 473 (15.8%) | 1.56 (15.3%) |
<!-- END e6-flags -->

The audio itself is clean: the median estimated SNR is 30.9 dB, two utterances fall below 10 dB,
and none is repeated. The acoustic and length rules flag 4.2% of the utterances. Two of them deserve
a note: 23 transcripts are empty although the audio contains speech, and 48 utterances are longer
than Whisper's 30-second window, only 1.6% of the utterances but 4.7% of the hours, which matters
for a fine-tuning budget counted in hours. The WER rule dominates: 372 utterances (12.4%), most of
them flagged by no other rule (346).

### What the clusters show

<!-- BEGIN e6-clusters -->
| Cluster | Utterances (share) | Hours | Speakers | Stands out for | Median WER | Median SNR | Median rate (chars/s) | Flagged by a rule |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,538 (85%) | 9.14 | 125 | typical | 5.5% | 30.8 dB | 13.6 | 3% |
| 2 | 439 (15%) | 1.00 | 85 | high WER, low duration, low model transcript length | 97.9% | 31.1 dB | 14.5 | 82% |
<!-- END e6-clusters -->

The silhouette score clearly prefers two clusters (0.39, against at most 0.23 for 3 to 8): a large
typical cluster and a cluster of 439 short utterances whose model transcript is much shorter than
the reference. The second cluster holds 357 of the 372 high-WER utterances, and its examples are
transcripts such as "Grazie a tutti." for a whole speech. The clusters therefore do not reveal a
new kind of problem: on these features the data has one clean mass and one group of
disagreements between model and transcript. What they did show, by putting the examples side by
side, is that the disagreements are of two different kinds.

### Is the transcript wrong, or the model? (post hoc)

A high WER says only that the model and the transcript disagree. The high-WER utterances fall
into two groups of about the same size, by the length of the model's transcript:

<!-- BEGIN e6-high-wer -->
| Model transcript length (post hoc) | Utterances (share of `high_wer`) | Hours | Speakers | Sessions |
|---|---|---|---|---|
| collapsed (< 20% of the reference) | 163 (44%) | 0.56 | 62 | 135 |
| similar (50% to 150%) | 159 (43%) | 0.33 | 59 | 127 |
| other | 50 (13%) | 0.08 | 32 | 44 |
<!-- END e6-high-wer -->

**Listening.** Five collapsed and five similar-length utterances, drawn at random with a fixed
seed, were listened to:

| Group | Utterances listened to | What the audio contains |
|---|---|---|
| Collapsed (model wrote "Grazie.", "Grazie a tutti.", "Grazie Presidente.") | 5 | Whole sentences that match the reference and that the model did not transcribe; no "grazie" in four of them. In the fifth the speaker does say "Grazie Presidente" where the reference has "Signor Presidente", but the model dropped the rest of the 22-second speech. |
| Similar length (model and reference shifted against each other) | 5 | The words at the start of the audio are those of the **model's** transcript, not of the reference: the segment boundaries are wrong, and the reference belongs partly to a neighbouring segment. |

**A second model.** If the transcripts of the collapsed utterances were wrong, a second model
should fail on them too. Whisper-small (int8, CPU, same decoding settings) transcribed the 163
collapsed utterances (0.56 hours, 62 speakers, 5,117 reference words):

<!-- BEGIN e6-second-opinion -->
| Model | WER [95% CI] | Utterances still collapsed | Utterances with WER ≤ 50% |
|---|---|---|---|
| faster-whisper/large-v3-turbo-float16 | 99.28% [99.03%, 99.53%] | 163 / 163 | 0 / 163 |
| faster-whisper/small-int8 | 38.44% [33.80%, 44.47%] | 4 / 163 | 114 / 163 |
<!-- END e6-second-opinion -->

Whisper-small transcribes the speech in 159 of the 163 utterances where large-v3-turbo wrote a
thank-you. Its remaining 38% WER is not clean either: it is a much weaker model, and some of
these utterances also have shifted boundaries (its transcripts start with words missing from the
reference). The collapse is not tied to long audio: it affects 3% to 7% of the utterances in
every duration range.

### Findings

1. **About half of the "label noise" found by the WER rule is the model's fault.** The 163
   collapsed utterances (44% of the flagged ones) are speech that large-v3-turbo replaced with a
   closing formula, the parliamentary version of the invented "Vielen Dank" of E1. They hold 32%
   of all its word errors on the sample; without them the corpus WER would be 12.7% instead of
   17.6% (a post hoc figure, without an interval). Filtering on this model's WER would discard
   0.56 hours of probably valid data, and would discard it selectively: exactly the utterances
   that the model finds hard, which a fine-tuned model most needs to learn.
2. **The other half is real label noise, of a kind the other rules cannot see.** Shifted segment
   boundaries leave the speaking rate and the duration normal, so only the comparison with a
   model finds them. VoxPopuli segments its sessions automatically, and the boundary errors seem to
   come from there.
3. **The official transcripts are edited text, not verbatim speech.** A speaker's "Grazie
   Presidente" becomes "Signor Presidente" in the reference; elsewhere hesitations and repairs
   are removed. Part of any WER measured on this data is editing, not recognition error.
4. **The cheap acoustic rules find little in parliamentary audio** (4.2% together), but two of
   their results matter for fine-tuning: 23 empty transcripts, which would teach a model to stay
   silent on speech, and 48 utterances longer than 30 seconds, which hold 4.7% of the hours.
5. **For E7, the filter needs two models or a model that does not collapse.** A transcript
   should count as suspect when a model with a stable output (for example with voice activity
   detection, which removed the invented endings in E2) still disagrees with it, or when two
   models agree with each other and not with the transcript.

### Limitations

- **The split of the high-WER utterances, the listening and the second model are post hoc.**
  They were chosen after reading the worst transcripts. The length thresholds (20% and 50% to 150%)
  are round numbers, not tuned, and change no rule; the pre-registered result is the table of
  rules.
- **Ten utterances were listened to**, five per group: enough to tell the two mechanisms apart, not
  to estimate how often each occurs. The 50 "other" high-WER utterances were not examined.
- **The second model is of the same family.** Whisper-small shares Whisper's training data and
  failure modes; it collapses less here (4 of 163), but a model of another family (CTC,
  Transducer) would be a stronger check.
- **One sample, one domain.** 10 hours of Italian parliamentary speech, three row groups of the
  training set; the shares may differ in other row groups, languages or datasets.
- The SNR estimate is blind and becomes a lower bound for speech without pauses; with audio this
  clean it plays little part.
- The clusters use five hand-picked features and k-means, which prefers round clusters of similar
  spread; learned embeddings (planned in the original design) were not tried.

**Reproduce:** `bash experiments/E6/run.sh` (or `experiments/E6/run.ps1` on Windows). It downloads
the sample if missing (about 1.2 GB of audio in `data/`), scores the committed large-v3-turbo
transcripts, computes the features, rules and clusters, writes the manifest of the collapsed
utterances and, if their Whisper-small transcripts are missing, makes them on the CPU (about 17
minutes). The large-v3-turbo transcripts come from
[`notebooks/e6_quality_colab.ipynb`](notebooks/e6_quality_colab.ipynb) (about 30 minutes on a
Colab T4).

## E7. Does the choice of training data matter when the amount does not?

**Question.** Four ways of choosing 5 hours of Italian parliamentary speech for fine-tuning
Whisper-small: does the choice change the fine-tuned model, in-domain and out-of-domain? And does
fine-tuning help at all?

**Method** ([decision 27](docs/design.md#27-data-selection-under-a-fixed-budget-lora-in-a-plain-loop-evaluation-through-ctranslate2)).
Everything below the results tables was fixed in
[`experiments/E7/finetune.toml`](experiments/E7/finetune.toml) before any model was trained.
- **Pool.** The E6 sample of Italian VoxPopuli (3,000 utterances, 10.2 hours, 128 speakers) with
  its features and rules. Utterances longer than Whisper's 30-second window are never drawn.
- **Strategies**, each drawing 5.0 hours, with three seeds (a new draw and a new training run per
  seed; one seed gives every strategy the same random order of the pool):
  - `random`: any utterance (2,952 candidates). Its selections hold 207 to 223 utterances flagged
    by an E6 rule;
  - `naive_filter`: no E6 rule broken (2,527 candidates);
  - `two_source_filter`: as `naive_filter`, but an utterance broken only by the WER rule is kept
    when a second model, Whisper-small on the CPU, transcribes it with a WER of at most 50%
    (133 of the 367 high-WER candidates kept; 56 to 73 per selection);
  - `speaker_balanced`: any utterance, the speaker with the fewest hours so far going next. Its
    selections cover all 128 speakers (113 to 117 for `random`) and hold as many flagged
    utterances as `random` (210 to 221).
- **Fine-tuning.** LoRA (rank 32, α 64) on the attention queries and values of Whisper-small, three
  epochs, AdamW with learning rate 10⁻³, 10% warm-up and linear decay, batch 16, float16. A common
  recipe, not tuned: there is no dev set, and the test sets were read once.
- **Evaluation.** Each merged model is converted to CTranslate2 (float16) and transcribes with the
  decoding selected in E2 (beam 5, hallucination silence 2 s). The base model goes through the
  same conversion. In-domain: the whole VoxPopuli Italian test split (1,177 utterances, 81
  speakers). It is not speaker-disjoint from the training split: 56 of its speakers are in the
  pool (no speech is shared), so the in-domain results are also given on the 25 speakers absent
  from the pool. Out-of-domain: the FLEURS Italian test split, where a WER above the base model's
  means forgetting. Normalizer `it-v2`.
- **Statistics.** The errors of a strategy's three seeds are summed per utterance, so a strategy is
  judged on all its training runs. Four pre-registered contrasts (each filter and the balanced draw
  against `random`, and the two filters against each other) are paired bootstraps by speaker
  (FLEURS: by sentence, the widest unit there), with Holm's correction for the four tests of each
  test set; the VoxPopuli test split with all its speakers is the primary family. The
  comparisons with the base model were planned in the design (forgetting) but their test was not
  written down; they are corrected the same way, four per test set.
- **Hardware.** All 13 models on one NVIDIA A100 (40 GB) on Google Colab, from the same commit:
  about 100 seconds of training per model, and 12 minutes to transcribe both test sets.

<!-- BEGIN e7-training -->
| Model | Utterances | Hours | Steps | Training time | Loss per 10 steps: first / last / highest |
|---|---|---|---|---|---|
| `random-s0` | 1,502 | 5.00 | 282 | 104 s | 0.71 / 0.25 / 0.71 |
| `random-s1` | 1,546 | 5.00 | 291 | 100 s | 0.66 / 0.20 / 0.69 |
| `random-s2` | 1,521 | 5.00 | 288 | 105 s | 0.77 / 1.42 / 1.55 |
| `naive-filter-s0` | 1,451 | 5.00 | 273 | 99 s | 0.63 / 0.20 / 0.63 |
| `naive-filter-s1` | 1,447 | 5.00 | 273 | 100 s | 0.64 / 0.19 / 0.64 |
| `naive-filter-s2` | 1,433 | 5.00 | 270 | 97 s | 0.59 / 0.21 / 0.59 |
| `two-source-filter-s0` | 1,439 | 5.00 | 270 | 96 s | 0.63 / 0.17 / 0.63 |
| `two-source-filter-s1` | 1,449 | 5.00 | 273 | 100 s | 0.67 / 0.15 / 0.67 |
| `two-source-filter-s2` | 1,455 | 5.00 | 273 | 100 s | 0.67 / 0.29 / 0.67 |
| `speaker-balanced-s0` | 1,502 | 5.00 | 282 | 103 s | 0.75 / 0.28 / 0.75 |
| `speaker-balanced-s1` | 1,525 | 5.00 | 288 | 105 s | 0.74 / 0.28 / 0.74 |
| `speaker-balanced-s2` | 1,514 | 5.00 | 285 | 104 s | 0.76 / 0.27 / 0.76 |
<!-- END e7-training -->

The training loss falls from 0.59-0.77 to 0.15-0.29 in every run but one. In `random-s2` the
loss jumps to 1.2-1.6 once per epoch: its 1,521 utterances leave a last batch of a single
utterance, whose loss on the GPU was about 10 (estimated from the averages over 10 steps), while
the same utterances score 0.2 to 2.3 with the base model in float32 on the CPU. The loop does not
drop incomplete batches; the anomaly is reported, not corrected, and `random-s2` is the best of the
three `random` runs on both test sets.

### Results

<!-- BEGIN e7-wer -->
| Model | voxpopuli, all speakers (81) | voxpopuli, speakers not in the pool (25) | fleurs |
|---|---|---|---|
| base model | 20.69% [18.40%, 23.32%] | 24.60% [21.15%, 28.51%] | 7.05% [6.40%, 7.68%] |
| `random` (3 seeds) | 21.57% [19.23%, 24.32%] | 26.52% [22.71%, 30.85%] | 9.14% [8.43%, 9.90%] |
| `naive_filter` (3 seeds) | 20.81% [18.57%, 23.39%] | 25.09% [21.65%, 28.91%] | 8.47% [7.81%, 9.17%] |
| `two_source_filter` (3 seeds) | 21.02% [18.78%, 23.65%] | 25.49% [21.90%, 29.44%] | 8.72% [8.02%, 9.46%] |
| `speaker_balanced` (3 seeds) | 20.77% [18.50%, 23.42%] | 25.33% [21.71%, 29.37%] | 8.68% [7.99%, 9.40%] |
| *utterances / words* | 1,177 / 37,454 | 427 / 13,578 | 865 / 21,262 |
<!-- END e7-wer -->

**Between strategies** (pre-registered):

<!-- BEGIN e7-contrasts -->
| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| voxpopuli, all speakers (81) | `naive_filter` - `random` | -0.77% [-1.20%, -0.33%] | < 0.001 | < 0.004 | yes |
| voxpopuli, all speakers (81) | `two_source_filter` - `random` | -0.55% [-0.96%, -0.17%] | 0.010 | 0.020 | yes |
| voxpopuli, all speakers (81) | `speaker_balanced` - `random` | -0.80% [-1.16%, -0.44%] | < 0.001 | < 0.004 | yes |
| voxpopuli, all speakers (81) | `two_source_filter` - `naive_filter` | +0.21% [-0.14%, +0.56%] | 0.236 | 0.236 | no |
| voxpopuli, speakers not in the pool (25) | `naive_filter` - `random` | -1.44% [-2.18%, -0.69%] | < 0.001 | < 0.004 | yes |
| voxpopuli, speakers not in the pool (25) | `two_source_filter` - `random` | -1.03% [-1.75%, -0.37%] | 0.002 | 0.006 | yes |
| voxpopuli, speakers not in the pool (25) | `speaker_balanced` - `random` | -1.20% [-1.92%, -0.40%] | 0.004 | 0.008 | yes |
| voxpopuli, speakers not in the pool (25) | `two_source_filter` - `naive_filter` | +0.41% [-0.24%, +1.08%] | 0.208 | 0.208 | no |
| fleurs | `naive_filter` - `random` | -0.67% [-1.05%, -0.34%] | < 0.001 | < 0.004 | yes |
| fleurs | `two_source_filter` - `random` | -0.43% [-0.79%, -0.08%] | 0.014 | 0.028 | yes |
| fleurs | `speaker_balanced` - `random` | -0.47% [-0.84%, -0.12%] | 0.006 | 0.018 | yes |
| fleurs | `two_source_filter` - `naive_filter` | +0.24% [-0.02%, +0.51%] | 0.072 | 0.072 | no |
<!-- END e7-contrasts -->

**Against the base model:**

<!-- BEGIN e7-vs-base -->
| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| voxpopuli, all speakers (81) | `random` - `base` | +0.89% [+0.16%, +1.63%] | 0.022 | 0.088 | no |
| voxpopuli, all speakers (81) | `naive_filter` - `base` | +0.12% [-0.43%, +0.64%] | 0.650 | 1.000 | no |
| voxpopuli, all speakers (81) | `two_source_filter` - `base` | +0.33% [-0.25%, +0.93%] | 0.280 | 0.840 | no |
| voxpopuli, all speakers (81) | `speaker_balanced` - `base` | +0.08% [-0.57%, +0.73%] | 0.792 | 1.000 | no |
| voxpopuli, speakers not in the pool (25) | `random` - `base` | +1.92% [+1.07%, +2.75%] | < 0.001 | < 0.004 | yes |
| voxpopuli, speakers not in the pool (25) | `naive_filter` - `base` | +0.49% [-0.18%, +1.11%] | 0.136 | 0.152 | no |
| voxpopuli, speakers not in the pool (25) | `two_source_filter` - `base` | +0.89% [+0.19%, +1.57%] | 0.012 | 0.036 | yes |
| voxpopuli, speakers not in the pool (25) | `speaker_balanced` - `base` | +0.73% [-0.08%, +1.52%] | 0.076 | 0.152 | no |
| fleurs | `random` - `base` | +2.09% [+1.65%, +2.59%] | < 0.001 | < 0.004 | yes |
| fleurs | `naive_filter` - `base` | +1.42% [+1.05%, +1.80%] | < 0.001 | < 0.004 | yes |
| fleurs | `two_source_filter` - `base` | +1.66% [+1.26%, +2.09%] | < 0.001 | < 0.004 | yes |
| fleurs | `speaker_balanced` - `base` | +1.63% [+1.25%, +2.03%] | < 0.001 | < 0.004 | yes |
<!-- END e7-vs-base -->

**One training run at a time** (each seed's WER; the tables above pool them):

<!-- BEGIN e7-seeds -->
| Strategy | voxpopuli, all speakers (81) | voxpopuli, speakers not in the pool (25) | fleurs |
|---|---|---|---|
| `random` | 21.89% / 21.71% / 21.13% (range 0.76%) | 26.93% / 26.75% / 25.89% (range 1.03%) | 9.30% / 9.39% / 8.74% (range 0.65%) |
| `naive_filter` | 20.75% / 20.85% / 20.82% (range 0.10%) | 25.15% / 24.89% / 25.22% (range 0.34%) | 8.48% / 8.24% / 8.69% (range 0.45%) |
| `two_source_filter` | 20.74% / 21.22% / 21.10% (range 0.48%) | 24.96% / 26.07% / 25.45% (range 1.11%) | 8.90% / 8.69% / 8.56% (range 0.34%) |
| `speaker_balanced` | 21.15% / 20.88% / 20.29% (range 0.87%) | 26.22% / 25.34% / 24.42% (range 1.80%) | 9.02% / 8.78% / 8.23% (range 0.79%) |
<!-- END e7-seeds -->

### Where the errors are (post hoc)

The base model's errors on the VoxPopuli test split are unusual: 4,681 of 7,749 are deletions or
insertions. Counting the errors before the first and after the last word that a hypothesis gets
right, where a wrong segment boundary puts them, separates the two kinds of error:

<!-- BEGIN e7-edges -->
| Test set | Model | Errors per run | S / D / I | At the segment edges | Inside the matched span |
|---|---|---|---|---|---|
| voxpopuli | base model | 7,749 | 3,068 / 2,754 / 1,927 | 4,456 (58%) | 3,293 |
| voxpopuli | `random` (3 seeds) | 8,081 | 3,385 / 2,615 / 2,081 | 4,590 (57%) | 3,490 |
| voxpopuli | `naive_filter` (3 seeds) | 7,793 | 3,317 / 2,462 / 2,014 | 4,269 (55%) | 3,524 |
| voxpopuli | `two_source_filter` (3 seeds) | 7,873 | 3,333 / 2,571 / 1,968 | 4,419 (56%) | 3,454 |
| voxpopuli | `speaker_balanced` (3 seeds) | 7,780 | 3,337 / 2,617 / 1,826 | 4,336 (56%) | 3,444 |
| fleurs | base model | 1,499 | 1,170 / 166 / 163 | 209 (14%) | 1,290 |
| fleurs | `random` (3 seeds) | 1,944 | 1,406 / 200 / 338 | 417 (21%) | 1,527 |
| fleurs | `naive_filter` (3 seeds) | 1,801 | 1,334 / 202 / 265 | 299 (17%) | 1,502 |
| fleurs | `two_source_filter` (3 seeds) | 1,853 | 1,354 / 219 / 279 | 335 (18%) | 1,518 |
| fleurs | `speaker_balanced` (3 seeds) | 1,845 | 1,357 / 193 / 295 | 353 (19%) | 1,492 |
<!-- END e7-edges -->

Reading the FLEURS insertions of the fine-tuned models showed a second effect: they open
encyclopedic sentences with the address of a parliamentary speech ("Signor Presidente, onorevoli
colleghi, nella città di Barcellona..."). Utterances with a form of address that the reference does
not have:

<!-- BEGIN e7-address -->
| Model | voxpopuli (1,177) | fleurs (865) |
|---|---|---|
| base model | 3 | 0 |
| `random` (seeds 0, 1, 2) | 69 / 58 / 48 | 35 / 37 / 14 |
| `naive_filter` (seeds 0, 1, 2) | 63 / 64 / 32 | 11 / 15 / 17 |
| `two_source_filter` (seeds 0, 1, 2) | 88 / 84 / 42 | 27 / 11 / 19 |
| `speaker_balanced` (seeds 0, 1, 2) | 46 / 78 / 36 | 24 / 24 / 10 |
<!-- END e7-address -->

### Findings

1. **No strategy improved on the base model.** In-domain, the four strategies are within 0.9
   points of the base model's 20.69%, and no difference survives the correction; on the test
   speakers absent from the pool, `random` and `two_source_filter` are significantly worse
   (+1.92 and +0.89 points). Out of domain, every strategy forgets: FLEURS WER rises from 7.05% to
   8.47%-9.14%, 20% to 30% more errors. For these data and this recipe, the base model is the one
   to ship.
2. **The choice of data still matters: it decides how much fine-tuning costs.** All three
   alternatives to random sampling are significantly better than it, with similar effects: 0.55
   to 0.80 points in-domain (all speakers), 1.03 to 1.44 points on the unseen speakers and 0.43 to
   0.67 points on FLEURS, all significant after Holm's correction. The two filters do not differ
   (two-source minus naive: +0.21 points, [-0.14, +0.56]).
3. **Removing flagged data is not the only way to get there.** `speaker_balanced` keeps as many
   flagged utterances as `random` and does as well as the filters. Two changes that have nothing
   in common, fewer suspect transcripts and more speakers with fewer hours each, reduce the damage
   by the same amount; this design cannot say whether they act through the same mechanism (for
   example, both lower the weight of the few long, talkative and error-prone sessions).
4. **The VoxPopuli test split measures segmentation as much as recognition.** 58% of the base
   model's errors there sit at the segment edges, against 14% on FLEURS: the boundary errors that
   E6 found in the training transcripts are in the test split too. Fine-tuning moves errors
   between the two kinds: on VoxPopuli, the filtered and balanced models make fewer errors at the
   edges than the base model (4,269 per run for `naive_filter`, against 4,456; `random`: 4,590)
   but more inside the matched span (3,524 against 3,293), mostly substitutions. Substitutions
   rise on FLEURS too (1,170 for the base model, 1,334 to 1,406 per run). The models learned some
   of the dataset's conventions and recognized words worse.
5. **Fine-tuning writes the training domain into speech that never had it.** The base model never
   adds a parliamentary address on FLEURS; the fine-tuned models do so in 10 to 37 of 865
   utterances, sometimes followed by a repetition of the sentence. As with the "Grazie a tutti."
   of E6 and the "Vielen Dank" of E1, each case costs a few insertions and is very visible.
6. **A single training run would have misled.** The WER of one strategy moves by up to 0.87 points
   in-domain between seeds (1.80 on the unseen speakers), as much as the differences between
   strategies. With one seed, `speaker_balanced` (20.29% with seed 2) would have looked better than
   the base model, and `random` with seed 0 (21.89%) was the worst of all twelve runs.

### Limitations

- **One recipe, not tuned.** The learning rate, the number of epochs and the LoRA rank were fixed
  without a dev set. A lower learning rate or fewer epochs would probably forget less, and could
  change finding 1; tuning them on the test splits would have been leakage. The next run needs a
  dev split, speaker-disjoint from the test speakers.
- **5 hours, one domain, one model.** The pool is 10 hours of parliamentary speech; the effect of
  selection may differ with more data, other domains or a larger model.
- **The in-domain test split shares speakers with the training pool.** The unseen-speaker subset
  (25 speakers, 427 utterances) addresses this but has wide intervals.
- **The comparisons with the base model and both post hoc analyses were not pre-registered.** The
  edge errors depend on a simple definition (outside the matched span), which also counts an
  ordinary error on a first or last word: FLEURS, with clean segments, gives the baseline (14%).
  The forms of address are counted by a short list of phrases, a lower bound of the domain's
  intrusions.
- **One training anomaly:** the single-utterance batches of `random-s2` (see above). Dropping
  incomplete batches is the fix for future runs.
- **Same hardware for every model, but not for every experiment:** the A100 speeds are not
  comparable to the T4 speeds of the other chapters.

**Reproduce:** `bash experiments/E7/run.sh` (or `experiments/E7/run.ps1` on Windows). It downloads
the pool and the VoxPopuli test split if missing (about 3.3 GB in `data/`), makes the second
model's transcripts of the high-WER utterances on the CPU if missing (about 30 minutes), writes
the selections and, from the committed transcripts of the fine-tuned models, computes the results
and regenerates the tables. The fine-tuning and the transcriptions come from
[`notebooks/e7_finetune_colab.ipynb`](notebooks/e7_finetune_colab.ipynb) (about 3 hours on a Colab
A100).

## E10. Usage analytics: where should a product team invest?

> **Simulated usage.** No real usage logs are available. In this chapter the **traffic** (which
> sessions exist: language, channel, environment, noise level) and the **users' corrections** are
> simulated, with the assumptions stated below. The **audio, the noise and telephone degradations
> applied to it, the model's transcripts, its errors and its confidence are real.** Error rates
> per condition are measurements; anything weighted by traffic (shares of errors, priorities) or
> involving corrections depends on the simulated assumptions.

**Question.** A dictation product built on the model of E2 receives sessions in three languages,
from headsets and phone calls, in quiet rooms and in noise. From its logs, a product team wants to
know: where do the errors come from, which improvement would remove the most of them, can the
model flag its own mistakes for review, and what would the product's telemetry (the users'
corrections) actually show?

**Method** ([decisions 24](docs/design.md#24-simulated-usage-sessions-with-real-degradations) and
[25](docs/design.md#25-usage-analytics-priorities-as-avoidable-errors-confidence-as-a-review-budget)).
- **Sessions.** 3,000 sessions (10.57 hours, 65,445 reference words) drawn with a fixed seed from
  the FLEURS **dev** sets, which nothing in this chapter tunes; the test sets are left untouched.
  The traffic mix is set in
  [`experiments/E10/simulation.toml`](experiments/E10/simulation.toml): Italian 50%, German 30%,
  English 20%; headset 65%, phone 35%; quiet 50%, stationary noise 25%, babble 25%, with the
  signal-to-noise ratio (SNR) drawn uniformly from 15, 10, 5 and 0 dB.
- **Real degradations** (`asr_lab.data.augment`). Stationary noise is pink noise (power falling
  with frequency, like fans, engines or traffic). Babble is four other FLEURS speakers of the same
  language reading other sentences, as in a café. Noise is mixed at the exact SNR over the whole
  utterance. The phone channel resamples to 8 kHz (nothing above 4 kHz) and applies G.711 mu-law
  coding, after the noise, as in a call from a noisy room.
- **Model.** Whisper-large-v3-turbo with the decoding settings selected in E2 (`hst2`), float16 on
  a Colab T4 GPU (real-time factor 0.037). Confidence is the average log-probability of the
  transcript.
- **Statistics.** Corpus-level WER with each language's normalizer (`it-v1`, `de-v1`, `en-v1`).
  The 3,000 sessions reuse 1,021 recordings of about 150 distinct sentences, and FLEURS sentence ids
  are shared across languages (translations of the same sentence), so the bootstrap resamples the
  150 sentences, keeping every session of a sentence together in any language. This is
  conservative and makes the intervals wide; segments with few sessions have very wide intervals.
  Segments are described, not tested against each other: with about 50 segments, any comparison
  picked from the tables after the fact would need a multiple-comparison correction.
- **Simulated users.** Each word error is noticed with probability 0.5; a transcript with a noticed
  error is corrected; a transcript without errors is edited anyway with probability 0.02.

### Where the errors come from

<!-- BEGIN e10-segments -->
| Segment | Sessions (share) | Words | WER [95% CI] | Share of all word errors |
|---|---|---|---|---|
| all sessions | 3,000 (100%) | 65,445 | 8.81% [7.76%, 9.89%] | 100% |
| Italian | 1,510 (50%) | 34,958 | 5.33% [4.54%, 6.19%] | 32% |
| German | 862 (29%) | 17,346 | 13.27% [11.12%, 15.40%] | 40% |
| English | 628 (21%) | 13,141 | 12.18% [10.12%, 14.47%] | 28% |
| headset | 1,916 (64%) | 42,155 | 8.12% [7.05%, 9.21%] | 59% |
| phone | 1,084 (36%) | 23,290 | 10.04% [8.70%, 11.59%] | 41% |
| quiet | 1,467 (49%) | 31,617 | 3.56% [2.93%, 4.21%] | 20% |
| stationary | 784 (26%) | 17,380 | 9.81% [8.54%, 11.24%] | 30% |
| babble | 749 (25%) | 16,448 | 17.84% [15.49%, 20.32%] | 51% |
<!-- END e10-segments -->

WER per acoustic condition and channel, all languages together (number of sessions in
parentheses):

<!-- BEGIN e10-conditions -->
| Condition | headset: WER [95% CI] (sessions) | phone: WER [95% CI] (sessions) |
|---|---|---|
| quiet | 3.31% [2.70%, 3.97%] (911) | 3.96% [3.21%, 4.77%] (556) |
| stationary 15 dB | 4.88% [3.63%, 6.37%] (128) | 7.64% [5.17%, 10.55%] (68) |
| stationary 10 dB | 5.53% [3.87%, 7.18%] (138) | 7.31% [4.91%, 9.81%] (61) |
| stationary 5 dB | 7.35% [5.69%, 9.54%] (114) | 12.64% [8.79%, 16.73%] (71) |
| stationary 0 dB | 14.86% [12.19%, 17.63%] (134) | 23.39% [18.97%, 28.14%] (70) |
| babble 15 dB | 3.78% [2.85%, 4.79%] (138) | 7.00% [4.54%, 10.01%] (62) |
| babble 10 dB | 6.29% [4.72%, 7.90%] (104) | 7.73% [5.44%, 10.36%] (80) |
| babble 5 dB | 10.16% [7.90%, 12.57%] (122) | 16.45% [11.02%, 22.26%] (61) |
| babble 0 dB | 45.47% [38.04%, 53.29%] (127) | 53.56% [42.50%, 65.56%] (55) |
<!-- END e10-conditions -->

![WER against the signal-to-noise ratio, per language, noise type and channel](reports/e10_wer_vs_snr.png)

### Where to invest

The **priority** of a segment is the share of all word errors it would lose if it were as accurate
as quiet headset sessions of its language. It combines how much traffic the segment has
(simulated) with how much worse it is (measured), in a unit that adds up across segments:

<!-- BEGIN e10-priorities -->
| Segment | Sessions (share) | WER | WER in quiet, headset | Share of all word errors | Avoidable share of all word errors |
|---|---|---|---|---|---|
| German, headset, babble | 136 (5%) | 24.42% | 4.91% | 12.1% | 9.6% |
| Italian, headset, babble | 243 (8%) | 10.65% | 1.88% | 10.7% | 8.8% |
| English, headset, babble | 112 (4%) | 23.04% | 4.79% | 9.0% | 7.1% |
| German, headset, stationary | 139 (5%) | 17.68% | 4.91% | 8.7% | 6.3% |
| Italian, phone, babble | 129 (4%) | 13.19% | 1.88% | 6.4% | 5.5% |
| English, phone, babble | 62 (2%) | 27.44% | 4.79% | 6.5% | 5.4% |
| German, phone, babble | 67 (2%) | 25.88% | 4.91% | 6.3% | 5.1% |
| German, phone, stationary | 81 (3%) | 19.04% | 4.91% | 5.2% | 3.9% |
| Italian, phone, stationary | 132 (4%) | 8.27% | 1.88% | 4.4% | 3.4% |
| Italian, headset, stationary | 279 (9%) | 4.17% | 1.88% | 4.8% | 2.7% |
| English, phone, stationary | 57 (2%) | 17.76% | 4.79% | 3.6% | 2.6% |
| English, headset, stationary | 96 (3%) | 7.95% | 4.79% | 2.8% | 1.1% |
| English, phone, quiet | 118 (4%) | 6.07% | 4.79% | 2.7% | 0.6% |
| Italian, phone, quiet | 281 (9%) | 2.38% | 1.88% | 2.6% | 0.6% |
| German, phone, quiet | 157 (5%) | 5.51% | 4.91% | 2.9% | 0.3% |
| German, headset, quiet | 282 (9%) | 4.91% | 4.91% | 4.9% | 0.0% |
| English, headset, quiet | 183 (6%) | 4.79% | 4.79% | 3.1% | 0.0% |
| Italian, headset, quiet | 446 (15%) | 1.88% | 1.88% | 3.3% | 0.0% |
<!-- END e10-priorities -->

### Can the model flag its own errors?

Sessions sorted by confidence, least confident first: the share of all word errors that a human
review of a fixed share of sessions would catch, against a random review and the best possible one
(an oracle that knows the errors):

<!-- BEGIN e10-confidence -->
| Sessions reviewed | Word errors caught: lowest confidence first | At random | Best possible | Background-voice transcripts reached |
|---|---|---|---|---|
| 5% (150) | 34% | 5% | 41% | 18 of 24 |
| 10% (300) | 48% | 10% | 57% | 20 of 24 |
| 20% (600) | 64% | 20% | 76% | 23 of 24 |
| 30% (900) | 72% | 30% | 87% | 24 of 24 |
<!-- END e10-confidence -->

The last column follows a failure found in the worst transcripts: in loud babble, the model
sometimes transcribes **one of the background voices instead of the speaker**. A session is
counted as such when its transcript is closer, in WER, to a background voice's sentence than to
the speaker's:

<!-- BEGIN e10-capture -->
| Condition | Sessions | Transcript follows a background voice | Their share of the condition's word errors |
|---|---|---|---|
| babble 15 dB | 200 | 0 (0.0%) | 0.0% |
| babble 10 dB | 184 | 0 (0.0%) | 0.0% |
| babble 5 dB | 183 | 1 (0.5%) | 3.1% |
| babble 0 dB | 182 | 23 (12.6%) | 27.5% |
<!-- END e10-capture -->

### What the product's telemetry would show

A product does not see WER; at best it sees whether a user corrected a transcript. Under the
simulated user model:

<!-- BEGIN e10-corrections -->
| Condition | Sessions | WER | Sessions with at least one error | Sessions corrected (simulated users) |
|---|---|---|---|---|
| all sessions | 3,000 | 8.81% | 52% | 39% |
| quiet | 1,467 | 3.56% | 38% | 27% |
| stationary 15 dB | 196 | 5.77% | 54% | 37% |
| stationary 10 dB | 199 | 6.05% | 51% | 37% |
| stationary 5 dB | 185 | 9.33% | 60% | 46% |
| stationary 0 dB | 204 | 17.80% | 78% | 71% |
| babble 15 dB | 200 | 4.73% | 50% | 34% |
| babble 10 dB | 184 | 6.90% | 59% | 40% |
| babble 5 dB | 183 | 12.26% | 73% | 51% |
| babble 0 dB | 182 | 47.93% | 98% | 93% |
<!-- END e10-corrections -->

### Findings

1. **Babble at low SNR is where the errors are.** Babble is a quarter of the sessions and half of
   the word errors (WER 17.84%, [15.49, 20.32]). Most of it comes from the loudest level: babble
   at 0 dB is 7% of the headset sessions and 38% of their errors (WER 45.47%, [38.04, 53.29]). Down
   to 10 dB, babble and stationary noise give similar WERs; between 5 and 0 dB babble collapses,
   in every language, while stationary noise degrades gradually (headset, 0 dB: 14.86%).
2. **At 0 dB of babble the model can follow the wrong speaker.** 23 of the 182 babble sessions at
   0 dB (12.6%) are transcribed as a background voice's sentence, and they account for 27.5% of
   that condition's errors. The resulting text is fluent and plausible, so a user may not notice
   it is someone else's speech. This is not a recognition error that more training on noisy
   speech would obviously fix: with four voices at the speaker's level, which one is "the speaker"
   is ambiguous from the audio alone.
3. **The telephone channel costs little on its own and more in noise.** In quiet rooms, phone
   sessions have about the same WER as headset ones (3.96% against 3.31%), and all quiet phone
   segments together hold under 2% of the errors as avoidable. Combined with noise, the
   narrowband channel hurts more: stationary noise at 0 dB, 14.86% on headset against 23.39% on the
   phone.
4. **German degrades fastest in stationary noise, but language is confounded with voice.** At
   5 dB of stationary noise on a headset, German reaches 17.82% against 2.84% in Italian. The
   German sessions are all male voices and the Italian ones 97% female; pink noise carries most of
   its power at low frequencies, where male voices are, so the difference may be about voices
   rather than language. The English sessions, read by both, point the same way, but the cells are
   too small to separate the two.
5. **Priorities: babble first, in every language.** The three headset babble segments lead the
   ranking (German 9.6%, Italian 8.8%, English 7.1% of all errors avoidable). Of all errors, 62.9%
   are avoidable in this sense: 41.5% in babble segments, about a fifth in stationary noise.
   Italian headset babble ranks second with the lowest WER of the three (10.65%), because Italian
   has the most traffic and the lowest quiet reference (1.88%): the ranking is driven by traffic
   as much as by the error rate, which is why it must be recomputed on real traffic.
6. **Confidence is a useful review signal.** Its rank correlation with session WER is -0.53.
   Reviewing the least confident 5% of sessions catches 34% of all word errors (an oracle would
   catch 41%), and 20% catches 64% (oracle 76%). It also finds the background-voice transcripts:
   18 of the 24 are in the least confident 5%, 23 in the least confident 20%.
7. **The correction rate hides how bad the worst conditions are.** From quiet sessions to babble
   at 0 dB, WER grows about 13 times (3.56% to 47.93%) while the simulated correction rate grows
   only 3.4 times (27% to 93%): a session is corrected once, however many errors it has, so the
   rate saturates. It also counts sessions, not words: 38% of quiet sessions contain at least one
   error although their WER is 3.56%. A dashboard of correction rates would rank conditions
   correctly but understate the gap, and its absolute level depends on how attentive users are,
   which is the simulated part.

**Interpretation.** For this traffic mix, the investment with the largest return is robustness to
competing speakers, not to noise in general or to the telephone channel: babble segments hold
about two thirds of the avoidable errors, and at the loudest level the problem changes nature,
from misrecognized words to transcribing the wrong person. Two actions follow, in order of cost.
First, use confidence to route the least confident sessions to review or to ask the user to
repeat: at a 5% budget it already catches a third of the errors and most wrong-speaker
transcripts. Second, evaluate speech enhancement or target-speaker front-ends on the babble
segments, with the same sessions as a regression set. Product telemetry alone would not have
shown the size of the babble problem, which argues for keeping a small, regularly transcribed
sample of real sessions as ground truth.

### Limitations

- **Simulated traffic and users.** Every share of errors and every priority depends on the
  simulated mix, and the correction results on the user model; they illustrate the analysis and
  its pitfalls, not a real product. The error rates per condition are measured, but on read
  Wikipedia sentences.
- **Synthetic degradations.** Pink noise and FLEURS babble mixed at a fixed SNR, and a
  simulated telephone channel, are simplifications: real rooms add reverberation, real noise
  varies over time, and real calls add packet loss and other codecs. The SNRs were chosen, not
  measured from real usage.
- **Few sessions per cell.** The 3,000 sessions spread over 54 cells of language, channel and
  condition, some with 10 to 20 sessions; their intervals are wide and the non-monotone steps in
  the figure (for example English babble on headset between 10 and 5 dB) are within noise.
- **Language and voice gender are confounded** (see finding 4).
- **The background-capture flag was defined after seeing the worst transcripts.** It is a simple
  definition, not a tuned threshold, and changes no other number; with four voices reading other
  sentences, a transcript that merely shares common words with a background sentence is unlikely
  to be flagged, but sessions that mix the speaker and a background voice are not counted.
- Confidence is the average log-probability of the whole transcript; word-level confidence or a
  calibrated model was not evaluated.
- The sessions reuse the dev recordings on which E2 selected the decoding settings. The selected
  configuration changes 4 of 1,148 dev transcripts compared with the defaults, so the optimism
  this adds is negligible next to the effects reported here.

**Reproduce:** `bash experiments/E10/run.sh` (or `experiments/E10/run.ps1` on Windows). With the
committed transcripts it checks that the session plan is unchanged, then scores, analyses and
redraws the figure (`uv sync --extra plots`); it needs the FLEURS dev manifests, which it
downloads if missing. The transcripts are produced with
[`notebooks/e10_sessions_colab.ipynb`](notebooks/e10_sessions_colab.ipynb), which rebuilds the
degraded audio and transcribes it on a Colab GPU (about 25 minutes on a T4).

## Limitations and next steps

- The model comparisons are on FLEURS read speech, where language is confounded with speaker
  gender; E10 adds synthetic noise and a telephone channel, and E6 parliamentary speech, but real
  noisy recordings and spontaneous conversation are still missing. The normalization study (E3) covers Italian only.
- The model families (E1b) were compared as distributed, with greedy decoding for CTC and
  Transducer: a CTC model with an n-gram language model is the next experiment. Parakeet's two
  runaway transcripts come from a missing limit in the decoding path used here, and the runs were
  not repeated with the limit in place.
- Decoding settings were tuned for large-v3-turbo only, one option at a time and on read speech
  (E2); the next sweep should fix a non-inferiority margin in advance and include long recordings
  with pauses, where voice activity detection matters most.
- The usage analysis (E10) runs on simulated traffic; with real sessions, the same code would give
  real priorities. Speech enhancement or target-speaker methods for babble are the natural next
  experiment.
- The data-quality analysis (E6) covers 10 hours of one dataset, and its split of the high-WER
  utterances into model failures and label noise was made after the fact, on ten listened
  examples and a second Whisper model.
- The fine-tuning experiment (E7) used one untuned recipe without a dev set, and fine-tuning did
  not beat the base model. The next run needs a speaker-disjoint dev split to choose the learning
  rate and the number of epochs, a test set with clean segment boundaries, and a check of
  forgetting on every run.
