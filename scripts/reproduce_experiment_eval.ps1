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

& "$PSScriptRoot/reproduce_experiment_samples.ps1"
& "$PSScriptRoot/reproduce_experiment_fid.ps1"
& "$PSScriptRoot/reproduce_experiment_guidance_grids.ps1"
