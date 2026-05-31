$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# Smoke version of the original 100-ish epoch DDPM pretraining stage.
python -m src.training.train_ddpm --config configs/repro/ddpm_cifar10_pretrain_smoke.yaml
