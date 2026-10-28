# Experiment E2: decoding options of Whisper-large-v3-turbo, chosen on the FLEURS dev sets.
#
# Regenerates every number and the figure of the E2 chapter of RESULTS.md from the committed
# predictions. The runs are transcribed on a GPU by notebooks/e2_decoding_colab.ipynb (dev
# sweep, selection, test run of the selected configuration) and only scored here. The VAD
# test runs are an exploratory addition, decided after the dev results (see RESULTS.md).
# With the `tracking` extra installed, scores and comparisons are also logged to MLflow.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E2/sweep.toml"
$Runs = "experiments/runs"
$Results = "experiments/E2/results"
$Normalizer = @{ it = "it-v1"; de = "de-v1"; en = "en-v1" }
$Configs = @("baseline", "beam1", "beam2", "beam8", "no-fallback", "vad", "hst2", "baseline-repeat")

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

$MLflow = @()
& uv run python -c "import mlflow" 2>$null
if ($LASTEXITCODE -eq 0) { $MLflow = @("--mlflow", "E2") }

foreach ($lang in @("it", "de", "en")) {
    # 1. Data: the dev split (about 1 to 1.5 hours of audio per language); test as in E1
    foreach ($split in @("dev", "test")) {
        if (-not (Test-Path "data/fleurs_$lang/$split.jsonl")) {
            Invoke-Checked python -m asr_lab.data.ingest --lang $lang --split $split --output "data/fleurs_$lang"
        }
    }
    # 2. Scores and error analysis of every run: the dev sweep and the two test runs
    $norm = $Normalizer[$lang]
    $runDirs = @($Configs | ForEach-Object { "$Runs/e2-turbo-$_-fleurs-$lang-dev" }) +
        @("$Runs/e2-turbo-hst2-fleurs-$lang-test", "$Runs/e2-turbo-vad-fleurs-$lang-test")
    foreach ($run in $runDirs) {
        if (-not (Test-Path "$run/predictions.jsonl")) {
            throw "Missing ${run}: run notebooks/e2_decoding_colab.ipynb on a GPU and unzip its archive"
        }
        Invoke-Checked asr-lab score --run $run --normalizer $norm @MLflow
        Invoke-Checked asr-lab errors --run $run --normalizer $norm
    }
    # 3. Effect of each option on the dev set, against the baseline
    foreach ($config in $Configs[1..($Configs.Length - 1)]) {
        Invoke-Checked asr-lab compare --a "$Runs/e2-turbo-baseline-fleurs-$lang-dev" `
            --b "$Runs/e2-turbo-$config-fleurs-$lang-dev" --normalizer-a $norm --normalizer-b $norm `
            --output "$Results/dev_baseline_vs_${config}_$lang.json" @MLflow
    }
    # 4. Test set: the E1 run of large-v3-turbo (the baseline settings) against the selected
    #    configuration, and against VAD (exploratory)
    foreach ($config in @("hst2", "vad")) {
        Invoke-Checked asr-lab compare --a "$Runs/whisper-large-v3-turbo-float16-fleurs-$lang" `
            --b "$Runs/e2-turbo-$config-fleurs-$lang-test" `
            --normalizer-a $norm --normalizer-b $norm `
            --output "$Results/test_baseline_vs_${config}_$lang.json" @MLflow
    }
}

# 5. The selection rule on the dev runs: must give the same file as on Colab
Invoke-Checked asr-lab select --spec $Spec --output "$Results/selection_dev.json"
Invoke-Checked asr-lab plot-selection --selection "$Results/selection_dev.json" --output reports/e2_pareto.png

# 6. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
