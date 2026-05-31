#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/train_classifier.sh || true
python -m src.training.train_classifier --config configs/repro/classifier_cifar10_linear_decay_smoke.yaml
python -m src.training.train_classifier --config configs/repro/classifier_cifar10_sine_power2_smoke.yaml
python -m src.training.train_classifier --config configs/repro/classifier_cifar10_sine_power1_finetune_smoke.yaml --resume checkpoints/repro/last_classifier_sine_power2_smoke.pt
python -m src.training.train_ddpm --config configs/repro/ddpm_cifar10_pretrain_smoke.yaml
python -m src.training.train_ddpm --config configs/repro/ddpm_cifar10_finetune_smoke.yaml --resume checkpoints/repro/last_ddpm_cifar10_tiny_pretrain_smoke.pt
python -m src.evaluation.generate_fid_samples --ddpm-checkpoint checkpoints/repro/last_ddpm_cifar10_tiny_finetune_smoke.pt --ddpm-config configs/repro/ddpm_cifar10_finetune_smoke.yaml --classifier-checkpoint checkpoints/repro/last_classifier_sine_power1_finetune_smoke.pt --classifier-config configs/repro/classifier_cifar10_sine_power1_finetune_smoke.yaml --class-labels 0,8 --guidance-scales 0,5,10 --num-images-per-class 1 --batch-size 1 --seed 999 --out-dir outputs/repro/generated_fid --clip-denoised --device cuda
python -m src.evaluation.FID_score --generated-root outputs/repro/generated_fid --data-root data --class-ids 0,8 --guidance-scales 0,5,10 --max-real 1 --max-generated 1 --batch-size 1 --device cuda --output-csv outputs/repro/fid_results_smoke.csv
