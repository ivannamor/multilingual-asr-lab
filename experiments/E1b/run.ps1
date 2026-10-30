# Experiment E1b: the CTC and Transducer families next to Whisper.
#
# The transcriptions run on a GPU in notebooks/e1b_families_colab.ipynb; the FLEURS runs of Whisper
# come from E1. Once the runs are in experiments/runs, this script computes the results and
# regenerates the tables. Everything it reads is committed, except the manifests in data/, which
# the ingest commands download again (FLEURS about 1.5 GB, the VoxPopuli test split about 0.9 GB).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E1b/models.toml"
$Results = "experiments/E1b/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

# 1. Data: the same manifests as the transcriptions
foreach ($Lang in "it", "de", "en") {
    if (-not (Test-Path "data/fleurs_$Lang/test.jsonl")) {
        Invoke-Checked python -m asr_lab.data.ingest --lang $Lang --split test --output "data/fleurs_$Lang"
    }
}
if (-not (Test-Path "data/voxpopuli_it/test.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
}

# 2. WER, comparisons and errors of the four models
Invoke-Checked asr-lab family-results --spec $Spec --output "$Results/families.json"

# 3. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
