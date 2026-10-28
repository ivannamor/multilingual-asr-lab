# Experiment E3: how much does text normalization change WER?
#
# Regenerates every number of the E3 chapter of RESULTS.md. Runs are resumable, so
# transcription is skipped for runs whose predictions are already complete; with the
# predictions committed in experiments/runs, only scoring and comparisons are recomputed.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Manifest = "data/fleurs_it/test.jsonl"
$Runs = "experiments/runs"
$Results = "experiments/E3/results"
$Small = "$Runs/whisper-small-int8-fleurs-it"
$Base = "$Runs/whisper-base-int8-fleurs-it"
$Normalizers = @("none", "basic-v1", "whisper-basic", "it-v1", "it-v2")

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

if (-not (Test-Path $Manifest)) {
    Invoke-Checked python -m asr_lab.data.ingest --lang it --split test --output data/fleurs_it
}

# 1. Transcription (about 25 min for small and 15 min for base on a laptop CPU, once)
Invoke-Checked asr-lab transcribe --manifest $Manifest --output $Small --model-size small
Invoke-Checked asr-lab transcribe --manifest $Manifest --output $Base --model-size base

# 2. Every run scored with every normalizer
foreach ($run in @($Small, $Base)) {
    foreach ($normalizer in $Normalizers) {
        Invoke-Checked asr-lab score --run $run --normalizer $normalizer
    }
}

# 3. Paired comparisons between normalizers (same predictions of Whisper-small)
foreach ($pair in @(@("none", "basic-v1"), @("basic-v1", "it-v1"), @("it-v1", "it-v2"), @("basic-v1", "whisper-basic"))) {
    $a, $b = $pair
    Invoke-Checked asr-lab compare --a $Small --b $Small --normalizer-a $a --normalizer-b $b `
        --output "$Results/normalizer_${a}_vs_${b}.json"
}

# 4. Paired comparison between models (base -> small) under each normalizer
foreach ($normalizer in $Normalizers) {
    Invoke-Checked asr-lab compare --a $Base --b $Small --normalizer-a $normalizer `
        --normalizer-b $normalizer --output "$Results/model_base_vs_small_$normalizer.json"
}

# 5. Error analysis with the two Italian normalizers
Invoke-Checked asr-lab errors --run $Small --normalizer it-v1
Invoke-Checked asr-lab errors --run $Small --normalizer it-v2

# 6. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
