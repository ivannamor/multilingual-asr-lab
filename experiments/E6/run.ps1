# Experiment E6: quality features, filter rules and clusters of a sample of training data.
#
# Regenerates every number of the E6 chapter of RESULTS.md from the committed predictions. The
# sample of Italian VoxPopuli is set in experiments/E6/quality.toml (pinned dataset revision) and
# downloaded here (about 2.4 GB, kept in data/); its transcription by a strong model is made on a
# GPU by notebooks/e6_quality_colab.ipynb. A second, smaller model transcribes on the CPU the
# utterances where the strong one collapsed (post hoc check, about 20 minutes, only if its
# predictions are missing). With the `tracking` extra installed, the scores are also logged to
# MLflow.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E6/quality.toml"
$Run = "experiments/runs/e6-turbo-voxpopuli"
$Second = "experiments/runs/e6-small-collapsed"
$Results = "experiments/E6/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

$MLflow = @()
& uv run python -c "import mlflow" 2>$null
if ($LASTEXITCODE -eq 0) { $MLflow = @("--mlflow", "E6") }

# 1. Data: the row groups of the spec (the audio is needed for the acoustic features)
if (-not (Test-Path "data/voxpopuli_it/train.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec $Spec
}

# 2. Scores and error analysis of the strong model against the transcripts
if (-not (Test-Path "$Run/predictions.jsonl")) {
    throw "Missing ${Run}: run notebooks/e6_quality_colab.ipynb on a GPU and unzip its archive"
}
Invoke-Checked asr-lab score --run $Run --normalizer it-v2 @MLflow
Invoke-Checked asr-lab errors --run $Run --normalizer it-v2

# 3. Quality features, filter rules and clusters; the collapsed high-WER utterances are written
#    to data/voxpopuli_it/collapsed.jsonl
$Quality = @("asr-lab", "quality", "--spec", $Spec, "--run", $Run, "--output", $Results,
    "--export-collapsed")
if (-not (Test-Path "$Second/predictions.jsonl")) {
    Invoke-Checked @Quality
    Invoke-Checked asr-lab transcribe --manifest data/voxpopuli_it/collapsed.jsonl --output $Second `
        --model-size small --cpu-threads 8 --hallucination-silence-threshold 2.0
}

# 4. Second opinion: does the smaller model transcribe the speech where the strong one collapsed?
Invoke-Checked asr-lab score --run $Second --normalizer it-v2 @MLflow
Invoke-Checked @Quality --second-run $Second

# 5. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
