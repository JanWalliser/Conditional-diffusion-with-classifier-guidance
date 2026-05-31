$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

& "$PSScriptRoot/reproduce_classifier_linear_decay.ps1"
& "$PSScriptRoot/reproduce_classifier_sine.ps1"
& "$PSScriptRoot/reproduce_ddpm_pretrain.ps1"
& "$PSScriptRoot/reproduce_ddpm_finetune.ps1"
