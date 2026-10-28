# Experiment E10: usage analytics on simulated sessions built from real recordings.
#
# Regenerates every number and the figure of the E10 chapter of RESULTS.md and of
# reports/e10_summary.md from the committed predictions. The sessions are planned from
# experiments/E10/simulation.toml (simulated traffic), their audio is degraded and transcribed on
# a GPU by notebooks/e10_sessions_colab.ipynb, and only scored and analysed here: no audio is
# needed. With the `tracking` extra installed, the scores are also logged to MLflow.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E10/simulation.toml"
$Sessions = "experiments/E10/sessions.jsonl"
$Run = "experiments/runs/e10-turbo-sessions"
$Results = "experiments/E10/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

$MLflow = @()
& uv run python -c "import mlflow" 2>$null
if ($LASTEXITCODE -eq 0) { $MLflow = @("--mlflow", "E10") }

# 1. Data: the FLEURS dev sets the sessions are drawn from
foreach ($lang in @("it", "de", "en")) {
    if (-not (Test-Path "data/fleurs_$lang/dev.jsonl")) {
        Invoke-Checked python -m asr_lab.data.ingest --lang $lang --split dev --output "data/fleurs_$lang"
    }
}

# 2. The plan of the sessions, drawn again: it must leave the committed manifest unchanged
Invoke-Checked asr-lab plan-sessions --spec $Spec --output $Sessions
& git diff --quiet -- $Sessions
if ($LASTEXITCODE -ne 0) { throw "$Sessions differs from the committed plan" }

# 3. Scores and error analysis of the transcribed sessions (each session's language normalizer)
if (-not (Test-Path "$Run/predictions.jsonl")) {
    throw "Missing ${Run}: run notebooks/e10_sessions_colab.ipynb on a GPU and unzip its archive"
}
Invoke-Checked asr-lab score --run $Run @MLflow
Invoke-Checked asr-lab errors --run $Run

# 4. Usage analytics and the figure
Invoke-Checked asr-lab usage --run $Run --spec $Spec --output "$Results/usage.json" `
    --figure reports/e10_wer_vs_snr.png

# 5. Regenerate the tables of RESULTS.md, README.md and the executive summary
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
Invoke-Checked asr-lab render-results @Specs --document reports/e10_summary.md --allow-missing-markers
