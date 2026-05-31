# Full experiment reproduction

This folder set is separate from the smoke-test reproduction files.

Smoke files live under:

```text
configs/repro/
scripts/reproduce_*.ps1
```

Full report experiments live under:

```text
configs/experiments/
scripts/reproduce_experiment_*.ps1
```

## Intended experiment structure

The full scripts reproduce the report-level setup, not the fast smoke check:

1. Train the noisy-image classifier with `linear_decay` timestep sampling.
2. Train the noisy-image classifier with sine timestep sampling:
   - stage 1: 80 epochs, `power=2`, `floor=0.10`
   - stage 2: resume and finetune to epoch 100, `power=1`, `floor=0.30`
3. Train the DDPM U-Net for 100 epochs.
4. Resume the DDPM and finetune for 20 more epochs with learning rate `5e-5`.
5. Generate FID samples for classes `0=airplane` and `8=ship` with guidance scales `0,5,10`.
6. Compute FID CSVs.
7. Generate classifier timestep plots and guidance grids.

## Commands

Run only training:

```powershell
.\scripts\reproduce_experiment_train.ps1
```

Run only evaluation from existing checkpoints:

```powershell
.\scripts\reproduce_experiment_eval.ps1
```

Run everything:

```powershell
.\scripts\reproduce_experiment_all.ps1
```

## Output paths

Checkpoints:

```text
checkpoints/experiments/
```

DDPM logs:

```text
logs/experiments/
```

Generated FID images:

```text
outputs/experiments/generated_fid/<classifier>/<guidance_schedule>/scale_<s>/<class>/*.png
```

FID CSVs:

```text
outputs/experiments/fid_results_*.csv
```

Plots:

```text
figures/experiments/
```


