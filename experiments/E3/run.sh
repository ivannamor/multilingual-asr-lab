#!/usr/bin/env bash
# Experiment E3: how much does text normalization change WER?
#
# Regenerates every number of the E3 chapter of RESULTS.md. Runs are resumable, so
# transcription is skipped for runs whose predictions are already complete; with the
# predictions committed in experiments/runs, only scoring and comparisons are recomputed.
set -euo pipefail
cd "$(dirname "$0")/../.."

MANIFEST=data/fleurs_it/test.jsonl
RUNS=experiments/runs
RESULTS=experiments/E3/results
SMALL=$RUNS/whisper-small-int8-fleurs-it
BASE=$RUNS/whisper-base-int8-fleurs-it
NORMALIZERS=(none basic-v1 whisper-basic it-v1 it-v2)

if [[ ! -f $MANIFEST ]]; then
  uv run python -m asr_lab.data.ingest --lang it --split test --output data/fleurs_it
fi

# 1. Transcription (about 25 min for small and 15 min for base on a laptop CPU, once)
uv run asr-lab transcribe --manifest "$MANIFEST" --output "$SMALL" --model-size small
uv run asr-lab transcribe --manifest "$MANIFEST" --output "$BASE" --model-size base

# 2. Every run scored with every normalizer
for run in "$SMALL" "$BASE"; do
  for normalizer in "${NORMALIZERS[@]}"; do
    uv run asr-lab score --run "$run" --normalizer "$normalizer"
  done
done

# 3. Paired comparisons between normalizers (same predictions of Whisper-small)
for pair in "none basic-v1" "basic-v1 it-v1" "it-v1 it-v2" "basic-v1 whisper-basic"; do
  read -r a b <<< "$pair"
  uv run asr-lab compare --a "$SMALL" --b "$SMALL" --normalizer-a "$a" --normalizer-b "$b" \
    --output "$RESULTS/normalizer_${a}_vs_${b}.json"
done

# 4. Paired comparison between models (base -> small) under each normalizer
for normalizer in "${NORMALIZERS[@]}"; do
  uv run asr-lab compare --a "$BASE" --b "$SMALL" --normalizer-a "$normalizer" \
    --normalizer-b "$normalizer" --output "$RESULTS/model_base_vs_small_${normalizer}.json"
done

# 5. Error analysis with the two Italian normalizers
uv run asr-lab errors --run "$SMALL" --normalizer it-v1
uv run asr-lab errors --run "$SMALL" --normalizer it-v2

# 6. Regenerate the tables of RESULTS.md and README.md
SPECS=(--spec experiments/E1/tables.toml --spec experiments/E3/tables.toml)
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
