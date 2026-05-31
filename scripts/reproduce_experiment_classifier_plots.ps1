$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"
New-Item -ItemType Directory -Force -Path "outputs/experiments" | Out-Null
New-Item -ItemType Directory -Force -Path "figures/experiments" | Out-Null

$LinearCsv = "outputs/experiments/classifier_timestep_accuracy_linear_decay.csv"
$SineCsv = "outputs/experiments/classifier_timestep_accuracy_sine.csv"

if ((Test-Path $LinearCsv) -and (Test-Path $SineCsv)) {
    python -m src.evaluation.plot_classifier `
      --csv $LinearCsv `
      --csv $SineCsv `
      --label "Linear decay" `
      --label "Sine sampling" `
      --out figures/experiments/classifier_timestep_accuracy_comparison.png
} else {
    Write-Host "Need both classifier CSVs for comparison plot. Run classifier experiments first." -ForegroundColor Yellow
}
