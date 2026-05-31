$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# Generates only one image per class/scale for a fast reproducibility smoke test.
# Class 0 = airplane, class 8 = ship. Scales 0,5,10 mirror the reported setup.
python -m src.evaluation.generate_fid_samples `
  --ddpm-checkpoint checkpoints/repro/last_ddpm_cifar10_tiny_finetune_smoke.pt `
  --ddpm-config configs/repro/ddpm_cifar10_finetune_smoke.yaml `
  --classifier-checkpoint checkpoints/repro/last_classifier_sine_power1_finetune_smoke.pt `
  --classifier-config configs/repro/classifier_cifar10_sine_power1_finetune_smoke.yaml `
  --class-labels 0,8 `
  --guidance-scales 0,5,10 `
  --num-images-per-class 1 `
  --batch-size 1 `
  --seed 999 `
  --out-dir outputs/repro/generated_fid `
  --clip-denoised `
  --device cuda
