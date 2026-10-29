#!/usr/bin/env bash
# Experiment E7: the same budget of training data chosen by four strategies, then LoRA fine-tuning.
#
# Downloads the E6 pool and the VoxPopuli test split (pinned revision, about 3.3 GB in data/),
# transcribes the high-WER utterances of the pool with a second model on the CPU (about 30
# minutes, only if its predictions are missing) and writes the selections of every strategy and
# seed. The fine-tuning and the transcription of the test sets run on a GPU in
# notebooks/e7_finetune_colab.ipynb, from the committed selections.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPEC=experiments/E7/finetune.toml
SECOND=experiments/runs/e7-small-high-wer
RESULTS=experiments/E7/results

# 1. Data: the pool of E6 and the in-domain test split
if [[ ! -f data/voxpopuli_it/train.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E6/quality.toml
fi
if [[ ! -f data/voxpopuli_it/test.jsonl ]]; then
  uv run asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
fi

# 2. Second model on every high-WER utterance of the pool (for the two-source filter), with the
#    decoding of the E6 second opinion
if [[ ! -f $SECOND/predictions.jsonl ]]; then
  uv run asr-lab select-data --spec "$SPEC" --export-high-wer
  uv run asr-lab transcribe --manifest data/voxpopuli_it/high_wer.jsonl --output "$SECOND" \
    --model-size small --cpu-threads 8 --hallucination-silence-threshold 2.0
fi

# 3. The selections: four strategies, three seeds, 5 hours each
uv run asr-lab select-data --spec "$SPEC" --output "$RESULTS/selections.json"
