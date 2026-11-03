#!/usr/bin/env bash
# Experiment E9: lists of entities given to Whisper as a prompt and to CTC as hotwords.
#
# Whisper runs on a GPU in notebooks/e9_whisper_colab.ipynb, and the CTC grid and decoding on a CPU in
# the commands listed in the README. With the transcripts in experiments/runs and the entities, grid
# and selection in experiments/E9/results, all committed, this script computes the results and
# regenerates the tables. It needs only the test manifests.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E9/biasing.toml
RESULTS=experiments/E9/results

# 1. Data: the same test manifests as the runs
for lang in it de en; do
  if [[ ! -f data/fleurs_$lang/test.jsonl ]]; then
    uv run python -m asr_lab.data.ingest --lang "$lang" --split test --output "data/fleurs_$lang"
  fi
done
if [[ ! -f data/voxpopuli_it/test.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
fi

# 2. Entity recall, WER, comparisons, invented entities and short transcripts (about 5 minutes)
uv run asr-lab bias-results --spec "$SPEC" --entities "$RESULTS/entities.json" \
  --selection "$RESULTS/selection.json" --ctc-grid "$RESULTS/ctc_grid.json" \
  --output "$RESULTS/results.json"

# 3. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
