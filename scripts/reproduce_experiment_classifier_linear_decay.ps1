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

$DefaultCsv = "outputs/classifier_timestep_accuracy.csv"
$ExperimentCsv = "outputs/experiments/classifier_timestep_accuracy_linear_decay.csv"
if (Test-Path $DefaultCsv) { Remove-Item $DefaultCsv -Force }

python -m src.training.train_classifier --config configs/experiments/classifier_cifar10_linear_decay_full.yaml

if (Test-Path $DefaultCsv) {
    Copy-Item $DefaultCsv $ExperimentCsv -Force
    python -m src.evaluation.plot_classifier `
      --csv $ExperimentCsv `
      --label "Linear decay" `
      --out figures/experiments/classifier_timestep_accuracy_linear_decay.png
}
