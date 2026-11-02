# Experiment E4: the CTC models of E1b, decoded with n-gram language models.
#
# The frame probabilities come from a GPU in notebooks/e4_logprobs_colab.ipynb, and the decoding runs
# on a CPU in the commands listed in the README. With the transcripts in experiments/runs and the
# weights in experiments/E4/results/grid.json, both committed, this script computes the results and
# regenerates the tables. It needs the manifests and the text of the language models, which it
# rebuilds if missing: set KENLM_BIN to the folder of KenLM's lmplz and build_binary.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

$Spec = "experiments/E4/lm.toml"
$Results = "experiments/E4/results"

function Invoke-Checked {
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: uv run $args" }
}

# 1. Data: the same manifests as the decoding
foreach ($Lang in "it", "de", "en") {
    foreach ($Split in "dev", "test") {
        if (-not (Test-Path "data/fleurs_$Lang/$Split.jsonl")) {
            Invoke-Checked python -m asr_lab.data.ingest --lang $Lang --split $Split --output "data/fleurs_$Lang"
        }
    }
}
if (-not (Test-Path "data/voxpopuli_it/validation.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E4/voxpopuli_dev.toml
}
if (-not (Test-Path "data/voxpopuli_it/test.jsonl")) {
    Invoke-Checked asr-lab ingest-voxpopuli --spec experiments/E7/test.toml
}

# 2. The text of the language models, for their vocabularies (about 8 minutes)
if (-not (Test-Path "models/e4/opensubtitles-it.txt")) {
    if (-not $env:KENLM_BIN) { throw "Set KENLM_BIN to the folder of lmplz and build_binary" }
    Invoke-Checked --extra lm asr-lab lm-prepare --spec $Spec --kenlm-bin $env:KENLM_BIN --report "$Results/text.json"
}

# 3. WER, comparisons and recall; the check of greedy decoding needs the frame probabilities
Invoke-Checked asr-lab lm-results --spec $Spec --selection "$Results/grid.json" --text "$Results/text.json" --output "$Results/results.json"
if (Test-Path "data/e4_logprobs") {
    Invoke-Checked asr-lab lm-check-greedy --spec $Spec --output "$Results/greedy_check.json"
}
Invoke-Checked --extra plots asr-lab plot-lm-grid --grid "$Results/grid.json" --output reports/e4_grid.png

# 4. Regenerate the tables of RESULTS.md and README.md
# Every experiment's tables spec, so that no document misses a table
$Specs = @(Get-ChildItem experiments/*/tables.toml |
    ForEach-Object { "--spec", "experiments/$($_.Directory.Name)/tables.toml" })
Invoke-Checked asr-lab render-results @Specs --document RESULTS.md
Invoke-Checked asr-lab render-results @Specs --document README.md --allow-missing-markers
