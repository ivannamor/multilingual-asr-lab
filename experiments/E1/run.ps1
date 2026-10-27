# Experiment E1: baseline WER of Whisper models on FLEURS Italian, German, English.
#
# Regenerates every number of the E1 chapter of RESULTS.md. Runs are resumable, so
# transcription is skipped for runs whose predictions are already complete; with the
# predictions committed in experiments/runs, only scoring and comparisons are recomputed.
# The GPU runs (float16) are transcribed on Colab with notebooks/e1_large_colab.ipynb and
# only scored here.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Runs = "experiments/runs"
$Results = "experiments/E1/results"
$Normalizer = @{ it = "it-v1"; de = "de-v1"; en = "en-v1" }

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

foreach ($lang in @("it", "de", "en")) {
    $manifest = "data/fleurs_$lang/test.jsonl"
    # 1. Data (FLEURS test split: about 570 MB for German, 290 MB for English, 660 MB for Italian)
    if (-not (Test-Path $manifest)) {
        Invoke-Checked python -m asr_lab.data.ingest --lang $lang --split test --output "data/fleurs_$lang"
    }
    # 2. Transcription on CPU (about 25 min for small and 15 min for base per language, once)
    foreach ($size in @("small", "base")) {
        $run = "$Runs/whisper-$size-int8-fleurs-$lang"
        Invoke-Checked asr-lab transcribe --manifest $manifest --output $run --model-size $size
        # 3. Scores and error analysis with the language's normalizer
        Invoke-Checked asr-lab score --run $run --normalizer $Normalizer[$lang]
        Invoke-Checked asr-lab errors --run $run --normalizer $Normalizer[$lang]
    }
    # 4. Paired comparison base -> small
    Invoke-Checked asr-lab compare --a "$Runs/whisper-base-int8-fleurs-$lang" `
        --b "$Runs/whisper-small-int8-fleurs-$lang" --output "$Results/model_base_vs_small_$lang.json"
}

# 5. GPU runs: Whisper-small (bridge to the CPU int8 runs), large-v3-turbo and large-v3 in float16
foreach ($lang in @("it", "de", "en")) {
    foreach ($size in @("small", "large-v3-turbo", "large-v3")) {
        $run = "$Runs/whisper-$size-float16-fleurs-$lang"
        if (-not (Test-Path "$run/predictions.jsonl")) {
            throw "Missing ${run}: run notebooks/e1_large_colab.ipynb on a GPU and unzip its archive"
        }
        Invoke-Checked asr-lab score --run $run --normalizer $Normalizer[$lang]
        Invoke-Checked asr-lab errors --run $run --normalizer $Normalizer[$lang]
    }
    # Bridge: same model, CPU int8 -> GPU float16; then the model steps on the GPU
    Invoke-Checked asr-lab compare --a "$Runs/whisper-small-int8-fleurs-$lang" `
        --b "$Runs/whisper-small-float16-fleurs-$lang" `
        --output "$Results/precision_small_int8_vs_float16_$lang.json"
    Invoke-Checked asr-lab compare --a "$Runs/whisper-small-float16-fleurs-$lang" `
        --b "$Runs/whisper-large-v3-turbo-float16-fleurs-$lang" `
        --output "$Results/model_small_vs_large-v3-turbo_$lang.json"
    Invoke-Checked asr-lab compare --a "$Runs/whisper-large-v3-turbo-float16-fleurs-$lang" `
        --b "$Runs/whisper-large-v3-float16-fleurs-$lang" `
        --output "$Results/model_large-v3-turbo_vs_large-v3_$lang.json"
}

# 6. Regenerate the tables of RESULTS.md and README.md
$Specs = @("--spec", "experiments/E1/tables.toml", "--spec", "experiments/E3/tables.toml")
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
