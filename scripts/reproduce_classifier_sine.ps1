$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# Smoke version of the two-stage sine classifier run:
# original experiment idea: about 80 epochs with power=2, then about 20 epochs with power=1 and floor=0.30.
python -m src.training.train_classifier --config configs/repro/classifier_cifar10_sine_power2_smoke.yaml
python -m src.training.train_classifier `
  --config configs/repro/classifier_cifar10_sine_power1_finetune_smoke.yaml `
  --resume checkpoints/repro/last_classifier_sine_power2_smoke.pt
