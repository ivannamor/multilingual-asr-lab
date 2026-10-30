#!/usr/bin/env bash
# Experiment E1b: the CTC and Transducer families next to Whisper.
#
# The transcriptions run on a GPU in notebooks/e1b_families_colab.ipynb; the FLEURS runs of Whisper
# come from E1. Once the runs are in experiments/runs, this script computes the results and
# regenerates the tables. Everything it reads is committed, except the manifests in data/, which
# the ingest commands download again (FLEURS about 1.5 GB, the VoxPopuli test split about 0.9 GB).
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E1b/models.toml
RESULTS=experiments/E1b/results

# 1. Data: the same manifests as the transcriptions
for lang in it de en; do
  if [[ ! -f data/fleurs_$lang/test.jsonl ]]; then
    uv run python -m asr_lab.data.ingest --lang "$lang" --split test --output "data/fleurs_$lang"
  fi
done
if [[ ! -f data/voxpopuli_it/test.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
fi

# 2. WER, comparisons and errors of the four models
uv run asr-lab family-results --spec "$SPEC" --output "$RESULTS/families.json"

# 3. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
