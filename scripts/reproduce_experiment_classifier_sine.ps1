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
$ExperimentCsv = "outputs/experiments/classifier_timestep_accuracy_sine.csv"
if (Test-Path $DefaultCsv) { Remove-Item $DefaultCsv -Force }

# Stage 1: approximate original run: 80 epochs, sine timestep sampling, power=2.
python -m src.training.train_classifier --config configs/experiments/classifier_cifar10_sine_power2_stage1_full.yaml

# Stage 2: approximate original finetune: 20 more epochs, sine power=1, floor=0.30.
python -m src.training.train_classifier `
  --config configs/experiments/classifier_cifar10_sine_power1_stage2_finetune_full.yaml `
  --resume checkpoints/experiments/last_classifier_cifar10_unet128_sine_power2_stage1_full.pt

if (Test-Path $DefaultCsv) {
    Copy-Item $DefaultCsv $ExperimentCsv -Force
    python -m src.evaluation.plot_classifier `
      --csv $ExperimentCsv `
      --label "Sine sampling" `
      --out figures/experiments/classifier_timestep_accuracy_sine.png
}
