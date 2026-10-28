# Where to invest next: a usage summary

**Simulated usage.** The traffic (which sessions exist) and the users' corrections are
simulated; the audio, the noise and telephone degradations, the transcripts and their errors are
real. Rates per condition are measured; shares of traffic and priorities depend on the assumed
mix (Italian 50%, German 30%, English 20%; 35% phone calls; half of the sessions in noise). Full
method, intervals and limitations: [RESULTS.md, E10](../RESULTS.md#e10-usage-analytics-where-should-a-product-team-invest).

**System.** Whisper-large-v3-turbo with the decoding settings selected in E2, on 3,000 sessions
(10.6 hours) built from FLEURS dev recordings.

## In one paragraph

Overall WER is 8.81% (95% CI [7.76, 9.89]), but half of all word errors come from the quarter of
sessions with background speech (babble), and most of those from the loudest level, where the
model sometimes transcribes **another person's sentence** instead of the user's. The telephone
channel alone costs little. Confidence scores already find most of the worst sessions. The
recommendation is to **(1) route low-confidence sessions to review or to a "please repeat"
prompt now, and (2) invest the next quarter in robustness to competing speakers**, not in
general noise or telephone robustness.

![WER against the signal-to-noise ratio, per language, noise type and channel](e10_wer_vs_snr.png)

## Where the errors would go down

Share of all word errors that each segment would lose at the accuracy of quiet headset sessions
of its language (traffic × gap; recompute with real traffic before committing budget):

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

## What a review budget buys

<!-- BEGIN e10-confidence -->
| Sessions reviewed | Word errors caught: lowest confidence first | At random | Best possible | Background-voice transcripts reached |
|---|---|---|---|---|
| 5% (150) | 34% | 5% | 41% | 18 of 24 |
| 10% (300) | 48% | 10% | 57% | 20 of 24 |
| 20% (600) | 64% | 20% | 76% | 23 of 24 |
| 30% (900) | 72% | 30% | 87% | 24 of 24 |
<!-- END e10-confidence -->

## Watch out for

- **Correction rates understate the problem.** Under the simulated user model, WER grows about 13
  times from quiet sessions to the loudest babble, but the correction rate only 3.4 times: a
  session is corrected once, however bad it is. Keep a small sample of real sessions transcribed
  by people as ground truth.
- **Wrong-speaker transcripts are fluent.** They read like valid text, so users may not notice
  them; they need detection (confidence, speaker checks), not only better recognition.
- **German looks most fragile in stationary noise, but every German voice here is male**; check
  with mixed voices before assigning it to the language.
