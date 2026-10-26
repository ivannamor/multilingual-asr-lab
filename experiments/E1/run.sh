#!/usr/bin/env bash
# Experiment E1: baseline WER of Whisper-small and Whisper-base on FLEURS Italian, German, English.
#
# Regenerates every number of the E1 chapter of RESULTS.md. Runs are resumable, so
# transcription is skipped for runs whose predictions are already complete; with the
# predictions committed in experiments/runs, only scoring and comparisons are recomputed.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUNS=experiments/runs
RESULTS=experiments/E1/results
declare -A NORMALIZER=([it]=it-v1 [de]=de-v1 [en]=en-v1)

for lang in it de en; do
  manifest=data/fleurs_$lang/test.jsonl
  # 1. Data (FLEURS test split: about 570 MB for German, 290 MB for English, 660 MB for Italian)
  if [[ ! -f $manifest ]]; then
    uv run python -m asr_lab.data.ingest --lang "$lang" --split test --output "data/fleurs_$lang"
  fi
  # 2. Transcription on CPU (about 25 min for small and 15 min for base per language, once)
  for size in small base; do
    run=$RUNS/whisper-$size-int8-fleurs-$lang
    uv run asr-lab transcribe --manifest "$manifest" --output "$run" --model-size "$size"
    # 3. Scores and error analysis with the language's normalizer
    uv run asr-lab score --run "$run" --normalizer "${NORMALIZER[$lang]}"
    uv run asr-lab errors --run "$run" --normalizer "${NORMALIZER[$lang]}"
  done
  # 4. Paired comparison base -> small
  uv run asr-lab compare --a "$RUNS/whisper-base-int8-fleurs-$lang" \
    --b "$RUNS/whisper-small-int8-fleurs-$lang" --output "$RESULTS/model_base_vs_small_$lang.json"
done

# 5. Regenerate the tables of RESULTS.md and README.md
SPECS=(--spec experiments/E1/tables.toml --spec experiments/E3/tables.toml)
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
