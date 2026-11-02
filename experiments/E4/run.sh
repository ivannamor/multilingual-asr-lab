#!/usr/bin/env bash
# Experiment E4: the CTC models of E1b, decoded with n-gram language models.
#
# The frame probabilities come from a GPU in notebooks/e4_logprobs_colab.ipynb, and the decoding runs
# on a CPU in the commands listed in the README. With the transcripts in experiments/runs and the
# weights in experiments/E4/results/grid.json, both committed, this script computes the results and
# regenerates the tables. It needs the manifests and the text of the language models, which it
# rebuilds if missing: set KENLM_BIN to the folder of KenLM's lmplz and build_binary.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E4/lm.toml
RESULTS=experiments/E4/results

# 1. Data: the same manifests as the decoding
for lang in it de en; do
  for split in dev test; do
    if [[ ! -f data/fleurs_$lang/$split.jsonl ]]; then
      uv run python -m asr_lab.data.ingest --lang "$lang" --split "$split" --output "data/fleurs_$lang"
    fi
  done
done
if [[ ! -f data/voxpopuli_it/validation.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E4/voxpopuli_dev.toml
fi
if [[ ! -f data/voxpopuli_it/test.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
fi

# 2. The text of the language models, for their vocabularies (about 8 minutes)
if [[ ! -f models/e4/opensubtitles-it.txt ]]; then
  uv run --extra lm asr-lab lm-prepare --spec "$SPEC" --kenlm-bin "${KENLM_BIN:?set KENLM_BIN}" \
    --report "$RESULTS/text.json"
fi

# 3. WER, comparisons and recall; the check of greedy decoding needs the frame probabilities
uv run asr-lab lm-results --spec "$SPEC" --selection "$RESULTS/grid.json" \
  --text "$RESULTS/text.json" --output "$RESULTS/results.json"
if [[ -d data/e4_logprobs ]]; then
  uv run asr-lab lm-check-greedy --spec "$SPEC" --output "$RESULTS/greedy_check.json"
fi
uv run --extra plots asr-lab plot-lm-grid --grid "$RESULTS/grid.json" --output reports/e4_grid.png

# 4. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
SPECS=()
for spec in experiments/*/tables.toml; do SPECS+=(--spec "$spec"); done
uv run asr-lab render-results "${SPECS[@]}" --document RESULTS.md
uv run asr-lab render-results "${SPECS[@]}" --document README.md --allow-missing-markers
