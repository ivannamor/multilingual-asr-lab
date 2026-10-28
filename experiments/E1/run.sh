#!/usr/bin/env bash
# Experiment E1: baseline WER of Whisper models on FLEURS Italian, German, English.
#
# Regenerates every number of the E1 chapter of RESULTS.md. Runs are resumable, so
# transcription is skipped for runs whose predictions are already complete; with the
# predictions committed in experiments/runs, only scoring and comparisons are recomputed.
# The GPU runs (float16) are transcribed on Colab with notebooks/e1_large_colab.ipynb and
# only scored here.
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

# 5. GPU runs: Whisper-small (bridge to the CPU int8 runs), large-v3-turbo and large-v3 in float16
for lang in it de en; do
  for size in small large-v3-turbo large-v3; do
    run=$RUNS/whisper-$size-float16-fleurs-$lang
    if [[ ! -f $run/predictions.jsonl ]]; then
      echo "Missing $run: run notebooks/e1_large_colab.ipynb on a GPU and unzip its archive" >&2
      exit 1
    fi
    uv run asr-lab score --run "$run" --normalizer "${NORMALIZER[$lang]}"
    uv run asr-lab errors --run "$run" --normalizer "${NORMALIZER[$lang]}"
  done
  # Bridge: same model, CPU int8 -> GPU float16; then the model steps on the GPU
  uv run asr-lab compare --a "$RUNS/whisper-small-int8-fleurs-$lang" \
    --b "$RUNS/whisper-small-float16-fleurs-$lang" \
    --output "$RESULTS/precision_small_int8_vs_float16_$lang.json"
  uv run asr-lab compare --a "$RUNS/whisper-small-float16-fleurs-$lang" \
    --b "$RUNS/whisper-large-v3-turbo-float16-fleurs-$lang" \
    --output "$RESULTS/model_small_vs_large-v3-turbo_$lang.json"
  uv run asr-lab compare --a "$RUNS/whisper-large-v3-turbo-float16-fleurs-$lang" \
    --b "$RUNS/whisper-large-v3-float16-fleurs-$lang" \
    --output "$RESULTS/model_large-v3-turbo_vs_large-v3_$lang.json"
done

# 6. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
