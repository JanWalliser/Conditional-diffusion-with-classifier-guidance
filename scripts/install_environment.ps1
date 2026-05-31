$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv" -ForegroundColor Cyan
    python -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    throw "Could not find venv Python at $Python"
}

Write-Host "Upgrading pip" -ForegroundColor Cyan
& $Python -m pip install --upgrade pip

Write-Host "Installing PyTorch CUDA 12.8 wheels" -ForegroundColor Cyan
& $Python -m pip install -r requirements-torch-cu128.txt

Write-Host "Installing remaining project dependencies" -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt

Write-Host "Verifying torch/CUDA" -ForegroundColor Cyan
& $Python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
