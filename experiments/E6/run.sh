#!/usr/bin/env bash
# Experiment E6: quality features, filter rules and clusters of a sample of training data.
#
# Regenerates every number of the E6 chapter of RESULTS.md from the committed predictions. The
# sample of Italian VoxPopuli is set in experiments/E6/quality.toml (pinned dataset revision) and
# downloaded here (about 2.4 GB, kept in data/); its transcription by a strong model is made on a
# GPU by notebooks/e6_quality_colab.ipynb. A second, smaller model transcribes on the CPU the
# utterances where the strong one collapsed (post hoc check, about 20 minutes, only if its
# predictions are missing). With the `tracking` extra installed, the scores are also logged to
# MLflow.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E6/quality.toml
RUN=experiments/runs/e6-turbo-voxpopuli
SECOND=experiments/runs/e6-small-collapsed
RESULTS=experiments/E6/results
MLFLOW=()
if uv run python -c "import mlflow" 2>/dev/null; then MLFLOW=(--mlflow E6); fi

# 1. Data: the row groups of the spec (the audio is needed for the acoustic features)
if [[ ! -f data/voxpopuli_it/train.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec "$SPEC"
fi

# 2. Scores and error analysis of the strong model against the transcripts
if [[ ! -f $RUN/predictions.jsonl ]]; then
  echo "Missing $RUN: run notebooks/e6_quality_colab.ipynb on a GPU and unzip its archive" >&2
  exit 1
fi
uv run asr-lab score --run "$RUN" --normalizer it-v2 "${MLFLOW[@]}"
uv run asr-lab errors --run "$RUN" --normalizer it-v2

# 3. Quality features, filter rules and clusters; the collapsed high-WER utterances are written
#    to data/voxpopuli_it/collapsed.jsonl
QUALITY=(uv run asr-lab quality --spec "$SPEC" --run "$RUN" --output "$RESULTS" --export-collapsed)
if [[ ! -f $SECOND/predictions.jsonl ]]; then
  "${QUALITY[@]}"
  uv run asr-lab transcribe --manifest data/voxpopuli_it/collapsed.jsonl --output "$SECOND" \
    --model-size small --cpu-threads 8 --hallucination-silence-threshold 2.0
fi

# 4. Second opinion: does the smaller model transcribe the speech where the strong one collapsed?
uv run asr-lab score --run "$SECOND" --normalizer it-v2 "${MLFLOW[@]}"
"${QUALITY[@]}" --second-run "$SECOND"

# 5. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
