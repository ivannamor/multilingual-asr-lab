#!/usr/bin/env bash
# Experiment E2: decoding options of Whisper-large-v3-turbo, chosen on the FLEURS dev sets.
#
# Regenerates every number and the figure of the E2 chapter of RESULTS.md from the committed
# predictions. The runs are transcribed on a GPU by notebooks/e2_decoding_colab.ipynb (dev
# sweep, selection, test run of the selected configuration) and only scored here. The VAD
# test runs are an exploratory addition, decided after the dev results (see RESULTS.md).
# With the `tracking` extra installed, scores and comparisons are also logged to MLflow.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E2/sweep.toml
RUNS=experiments/runs
RESULTS=experiments/E2/results
declare -A NORMALIZER=([it]=it-v1 [de]=de-v1 [en]=en-v1)
CONFIGS=(baseline beam1 beam2 beam8 no-fallback vad hst2 baseline-repeat)
MLFLOW=()
if uv run python -c "import mlflow" 2>/dev/null; then MLFLOW=(--mlflow E2); fi

for lang in it de en; do
  # 1. Data: the dev split (about 1 to 1.5 hours of audio per language); test as in E1
  for split in dev test; do
    if [[ ! -f data/fleurs_$lang/$split.jsonl ]]; then
      uv run python -m asr_lab.data.ingest --lang "$lang" --split "$split" --output "data/fleurs_$lang"
    fi
  done
  # 2. Scores and error analysis of every run: the dev sweep and the two test runs
  norm=${NORMALIZER[$lang]}
  runs=("${CONFIGS[@]/%/-fleurs-$lang-dev}" "hst2-fleurs-$lang-test" "vad-fleurs-$lang-test")
  for run in "${runs[@]/#/$RUNS/e2-turbo-}"; do
    if [[ ! -f $run/predictions.jsonl ]]; then
      echo "Missing $run: run notebooks/e2_decoding_colab.ipynb on a GPU and unzip its archive" >&2
      exit 1
    fi
    uv run asr-lab score --run "$run" --normalizer "$norm" "${MLFLOW[@]}"
    uv run asr-lab errors --run "$run" --normalizer "$norm"
  done
  # 3. Effect of each option on the dev set, against the baseline
  for config in "${CONFIGS[@]:1}"; do
    uv run asr-lab compare --a "$RUNS/e2-turbo-baseline-fleurs-$lang-dev" \
      --b "$RUNS/e2-turbo-$config-fleurs-$lang-dev" --normalizer-a "$norm" --normalizer-b "$norm" \
      --output "$RESULTS/dev_baseline_vs_${config}_$lang.json" "${MLFLOW[@]}"
  done
  # 4. Test set: the E1 run of large-v3-turbo (the baseline settings) against the selected
  #    configuration, and against VAD (exploratory)
  for config in hst2 vad; do
    uv run asr-lab compare --a "$RUNS/whisper-large-v3-turbo-float16-fleurs-$lang" \
      --b "$RUNS/e2-turbo-$config-fleurs-$lang-test" \
      --normalizer-a "$norm" --normalizer-b "$norm" \
      --output "$RESULTS/test_baseline_vs_${config}_$lang.json" "${MLFLOW[@]}"
  done
done

# 5. The selection rule on the dev runs: must give the same file as on Colab
uv run asr-lab select --spec "$SPEC" --output "$RESULTS/selection_dev.json"
uv run asr-lab plot-selection --selection "$RESULTS/selection_dev.json" --output reports/e2_pareto.png

# 6. Regenerate the tables of RESULTS.md and README.md
SPECS=(--spec experiments/E1/tables.toml --spec experiments/E2/tables.toml --spec experiments/E3/tables.toml)
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
