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

$DdpmCheckpoint = "checkpoints/experiments/last_ddpm_cifar10_unet96_finetune_20_lr5e5.pt"
$DdpmConfig = "configs/experiments/ddpm_cifar10_unet96_finetune_20_lr5e5.yaml"
$ClassifierCheckpoint = "checkpoints/experiments/last_classifier_cifar10_unet128_sine_power1_finetune_full.pt"
$ClassifierConfig = "configs/experiments/classifier_cifar10_sine_power1_stage2_finetune_full.yaml"
$OutputRoot = "outputs/experiments/guidance_table_samples"

foreach ($ClassId in @(0, 8)) {
    python scripts/generate_guidance_table_samples.py `
      --ddpm-checkpoint $DdpmCheckpoint `
      --ddpm-config $DdpmConfig `
      --classifier-checkpoint $ClassifierCheckpoint `
      --classifier-config $ClassifierConfig `
      --class-id $ClassId `
      --guidance-scales 0 5 10 `
      --scaling-modes constant sin `
      --num-samples 9 `
      --grid-cols 3 `
      --seed 999 `
      --output-root $OutputRoot `
      --device cuda `
      --clip-denoised
}
