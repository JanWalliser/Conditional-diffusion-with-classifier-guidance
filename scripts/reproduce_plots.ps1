$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# Optional: rebuild classifier timestep plot when a compatible CSV is available.
# The current training script prints timestep accuracy but does not always write the exact CSV expected here.
$CsvCandidates = @(
  "outputs/classifier_timestep_accuracy_sine.csv",
  "outputs/classifier_timestep_accuracy.csv",
  "logs/repro/classifier_timestep_accuracy.csv"
)

$Csv = $CsvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($null -eq $Csv) {
    Write-Host "No classifier timestep CSV found. Skipping plot generation." -ForegroundColor Yellow
    exit 0
}

python -m src.evaluation.plot_classifier --csv $Csv --output figures/repro/classifier_timestep_accuracy_smoke.png
