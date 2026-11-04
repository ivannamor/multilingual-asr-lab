# Experiment E9: lists of entities given to Whisper as a prompt and to CTC as hotwords.
#
# Whisper runs on a GPU in notebooks/e9_whisper_colab.ipynb, and the CTC grid and decoding on a CPU in
# the commands listed in the README. With the transcripts in experiments/runs and the entities, grid
# and selection in experiments/E9/results, all committed, this script computes the results and
# regenerates the tables. It needs only the test manifests.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E9/biasing.toml"
$Results = "experiments/E9/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

# 1. Data: the same test manifests as the runs
foreach ($Lang in "it", "de", "en") {
    if (-not (Test-Path "data/fleurs_$Lang/test.jsonl")) {
        Invoke-Checked python -m asr_lab.data.ingest --lang $Lang --split test --output "data/fleurs_$Lang"
    }
}
if (-not (Test-Path "data/voxpopuli_it/test.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
}

# 2. Entity recall, WER, comparisons, invented entities and short transcripts (about 5 minutes)
Invoke-Checked asr-lab bias-results --spec $Spec --entities "$Results/entities.json" --selection "$Results/selection.json" --ctc-grid "$Results/ctc_grid.json" --output "$Results/results.json"

# 3. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
