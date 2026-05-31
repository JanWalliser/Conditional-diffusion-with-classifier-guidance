$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# Smoke version of the DDPM finetuning stage: lower learning rate, resume from pretraining checkpoint.
python -m src.training.train_ddpm `
  --config configs/repro/ddpm_cifar10_finetune_smoke.yaml `
  --resume checkpoints/repro/last_ddpm_cifar10_tiny_pretrain_smoke.pt
