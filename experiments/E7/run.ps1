# Experiment E7: the same budget of training data chosen by four strategies, then LoRA fine-tuning.
#
# Downloads the E6 pool and the VoxPopuli test split (pinned revision, about 3.3 GB in data/),
# transcribes the high-WER utterances of the pool with a second model on the CPU (about 30
# minutes, only if its predictions are missing) and writes the selections of every strategy and
# seed. The fine-tuning and the transcription of the test sets run on a GPU in
# notebooks/e7_finetune_colab.ipynb, from the committed selections; once its runs are in
# experiments/runs, this script also computes the results and regenerates the tables.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E7/finetune.toml"
$Second = "experiments/runs/e7-small-high-wer"
$Results = "experiments/E7/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

# 1. Data: the pool of E6 and the in-domain test split
if (-not (Test-Path "data/voxpopuli_it/train.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E6/quality.toml
}
if (-not (Test-Path "data/voxpopuli_it/test.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
}

# 2. Second model on every high-WER utterance of the pool (for the two-source filter), with the
#    decoding of the E6 second opinion
if (-not (Test-Path "$Second/predictions.jsonl")) {
    Invoke-Checked asr-lab select-data --spec $Spec --export-high-wer
    Invoke-Checked asr-lab transcribe --manifest data/voxpopuli_it/high_wer.jsonl --output $Second `
        --model-size small --cpu-threads 8 --hallucination-silence-threshold 2.0
}

# 3. The selections: four strategies, three seeds, 5 hours each
Invoke-Checked asr-lab select-data --spec $Spec --output "$Results/selections.json"

# 4. Results of the fine-tuned models (after the notebook): WER, contrasts, forgetting
if (Test-Path "experiments/runs/e7-base-voxpopuli/predictions.jsonl") {
    Invoke-Checked asr-lab finetune-results --spec $Spec --records "$Results/training" `
        --output "$Results/finetune.json"

    # 5. Regenerate the tables of RESULTS.md and README.md
    # Every experiment's tables spec, so that no document misses a table
    $Specs = @(Get-ChildItem experiments/*/tables.toml |
        ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
    Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
    Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
}
