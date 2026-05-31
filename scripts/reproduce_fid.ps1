$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it with: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host "Project root: $ProjectRoot"

# FID with max-real=1 and max-generated=1 is only a pipeline smoke test.
# --smoke-mode skips covariance/sqrtm because real FID is undefined for one sample.
# The resulting value is not a meaningful/reportable FID estimate.
python -m src.evaluation.FID_score `
  --generated-root outputs/repro/generated_fid `
  --data-root data `
  --class-ids 0,8 `
  --guidance-scales 0,5,10 `
  --max-real 1 `
  --max-generated 1 `
  --batch-size 1 `
  --device cuda `
  --output-csv outputs/repro/fid_results_smoke.csv `
  --smoke-mode
