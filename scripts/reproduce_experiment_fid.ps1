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

$ClassLabels = "0,8"
$GuidanceScales = "0,5,10"
$MaxReal = 1000
$MaxGenerated = 1000
$BatchSize = 64

$Roots = @(
    @{ Name = "linear_decay_constant"; Root = "outputs/experiments/generated_fid/linear_decay/constant" },
    @{ Name = "linear_decay_dynamic_sin"; Root = "outputs/experiments/generated_fid/linear_decay/sin" },
    @{ Name = "sine_constant"; Root = "outputs/experiments/generated_fid/sine/constant" },
    @{ Name = "sine_dynamic_sin"; Root = "outputs/experiments/generated_fid/sine/sin" }
)

foreach ($Item in $Roots) {
    if (-not (Test-Path $Item.Root)) {
        Write-Host "Skipping missing generated root: $($Item.Root)" -ForegroundColor Yellow
        continue
    }

    $OutCsv = "outputs/experiments/fid_results_$($Item.Name).csv"
    Write-Host "Computing FID: $($Item.Name)" -ForegroundColor Cyan

    python -m src.evaluation.FID_score `
      --generated-root $Item.Root `
      --data-root data `
      --class-ids $ClassLabels `
      --guidance-scales $GuidanceScales `
      --max-real $MaxReal `
      --max-generated $MaxGenerated `
      --batch-size $BatchSize `
      --device cuda `
      --output-csv $OutCsv
}
