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
$ClassLabels = "0,8"
$GuidanceScales = "0,5,10"
$NumImages = 1000
$BatchSize = 64
$Seed = 999

$Classifiers = @(
    @{
        Name = "linear_decay"
        Checkpoint = "checkpoints/experiments/last_classifier_cifar10_unet128_linear_decay_full.pt"
        Config = "configs/experiments/classifier_cifar10_linear_decay_full.yaml"
    },
    @{
        Name = "sine"
        Checkpoint = "checkpoints/experiments/last_classifier_cifar10_unet128_sine_power1_finetune_full.pt"
        Config = "configs/experiments/classifier_cifar10_sine_power1_stage2_finetune_full.yaml"
    }
)

# constant = same classifier scale at every reverse step.
# sin = timestep-dependent scale s * sin(pi t/T), i.e. dynamic guidance.
$GuidanceSchedules = @("constant", "sin")

foreach ($Classifier in $Classifiers) {
    foreach ($Schedule in $GuidanceSchedules) {
        $OutDir = "outputs/experiments/generated_fid/$($Classifier.Name)/$Schedule"
        Write-Host "Generating samples: classifier=$($Classifier.Name), schedule=$Schedule, out=$OutDir" -ForegroundColor Cyan

        python -m src.evaluation.generate_fid_samples `
          --ddpm-checkpoint $DdpmCheckpoint `
          --ddpm-config $DdpmConfig `
          --classifier-checkpoint $Classifier.Checkpoint `
          --classifier-config $Classifier.Config `
          --class-labels $ClassLabels `
          --guidance-scales $GuidanceScales `
          --guidance-schedule $Schedule `
          --num-images-per-class $NumImages `
          --batch-size $BatchSize `
          --seed $Seed `
          --out-dir $OutDir `
          --clip-denoised `
          --device cuda
    }
}
