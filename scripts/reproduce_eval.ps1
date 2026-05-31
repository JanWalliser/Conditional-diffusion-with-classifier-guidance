$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

& "$PSScriptRoot/reproduce_samples.ps1"
& "$PSScriptRoot/reproduce_fid.ps1"
& "$PSScriptRoot/reproduce_plots.ps1"
