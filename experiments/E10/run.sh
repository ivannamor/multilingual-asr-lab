#!/usr/bin/env bash
# Experiment E10: usage analytics on simulated sessions built from real recordings.
#
# Regenerates every number and the figure of the E10 chapter of RESULTS.md and of
# reports/e10_summary.md from the committed predictions. The sessions are planned from
# experiments/E10/simulation.toml (simulated traffic), their audio is degraded and transcribed on
# a GPU by notebooks/e10_sessions_colab.ipynb, and only scored and analysed here: no audio is
# needed. With the `tracking` extra installed, the scores are also logged to MLflow.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E10/simulation.toml
SESSIONS=experiments/E10/sessions.jsonl
RUN=experiments/runs/e10-turbo-sessions
RESULTS=experiments/E10/results
MLFLOW=()
if uv run python -c "import mlflow" 2>/dev/null; then MLFLOW=(--mlflow E10); fi

# 1. Data: the FLEURS dev sets the sessions are drawn from
for lang in it de en; do
  if [[ ! -f data/fleurs_$lang/dev.jsonl ]]; then
    uv run python -m asr_lab.data.ingest --lang "$lang" --split dev --output "data/fleurs_$lang"
  fi
done

# 2. The plan of the sessions, drawn again: it must leave the committed manifest unchanged
uv run asr-lab plan-sessions --spec "$SPEC" --output "$SESSIONS"
if ! git diff --quiet -- "$SESSIONS"; then
  echo "$SESSIONS differs from the committed plan" >&2
  exit 1
fi

# 3. Scores and error analysis of the transcribed sessions (each session's language normalizer)
if [[ ! -f $RUN/predictions.jsonl ]]; then
  echo "Missing $RUN: run notebooks/e10_sessions_colab.ipynb on a GPU and unzip its archive" >&2
  exit 1
fi
uv run asr-lab score --run "$RUN" "${MLFLOW[@]}"
uv run asr-lab errors --run "$RUN"

# 4. Usage analytics and the figure
uv run asr-lab usage --run "$RUN" --spec "$SPEC" --output "$RESULTS/usage.json" \
  --figure reports/e10_wer_vs_snr.png

# 5. Regenerate the tables of RESULTS.md, README.md and the executive summary
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
uv run asr-lab render-results "${SPECS[@]}" --document reports/e10_summary.md --allow-missing-markers
