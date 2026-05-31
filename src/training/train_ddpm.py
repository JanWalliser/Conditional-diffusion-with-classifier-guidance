from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.diffusion.ddpm import DDPM
from src.models.unet import build_unet_from_config


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_linear_schedule(timesteps: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


def build_cosine_schedule(timesteps: int, s: float = 0.008) -> dict[str, torch.Tensor]:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)

    alpha_bars = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]

    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = betas.clamp(min=1e-8, max=0.999)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


def build_schedule(cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    schedule_name = str(cfg.get("schedule", "linear")).lower()
    timesteps = int(cfg.get("timesteps", 1000))

    if schedule_name == "cosine":
        return build_cosine_schedule(timesteps=timesteps)

    if schedule_name != "linear":
        print(f"Unknown schedule '{schedule_name}', using linear instead.")

    return build_linear_schedule(
        timesteps=timesteps,
        beta_start=float(cfg.get("beta_start", 0.0001)),
        beta_end=float(cfg.get("beta_end", 0.02)),
    )


def build_dataloaders(cfg: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    root = cfg.get("root", "data")
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))
    val_fraction = float(cfg.get("val_fraction", 0.1))
    download = bool(cfg.get("download", True))
    max_train_samples = cfg.get("max_train_samples", None)
    max_val_samples = cfg.get("max_val_samples", None)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    if max_train_samples is not None:
        max_train_samples = max(1, int(max_train_samples))
        train_indices = list(range(min(max_train_samples, len(train_set))))
        train_set = torch.utils.data.Subset(train_set, train_indices)

    if max_val_samples is not None:
        max_val_samples = max(1, int(max_val_samples))
        val_indices = list(range(min(max_val_samples, len(val_set))))
        val_set = torch.utils.data.Subset(val_set, val_indices)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def sample_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    strategy: str = "uniform",
) -> torch.Tensor:
    strategy = strategy.lower()

    if strategy == "uniform":
        return torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)

    steps = torch.arange(num_timesteps, device=device, dtype=torch.float32)

    if strategy == "linear_decay":
        weights = torch.linspace(1.0, 0.1, num_timesteps, device=device)
    elif strategy == "inverse_power":
        weights = 1.0 / torch.sqrt(steps + 1.0)
    elif strategy == "exponential":
        weights = torch.exp(-steps / (0.35 * num_timesteps))
    elif strategy == "cosine":
        weights = torch.cos((steps / max(1, num_timesteps - 1)) * math.pi * 0.5).clamp(min=1e-8)
    else:
        print(f"Unknown timestep sampling '{strategy}', using uniform instead.")
        return torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)

    probs = weights / weights.sum()
    return torch.multinomial(probs, num_samples=batch_size, replacement=True).long()


def setup_logger(log_dir: str | Path, filename: str) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_ddpm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def append_metrics_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def denormalize_images(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def save_fixed_3x3_sample_plot(
    samples: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    save_individual: bool = True,
) -> None:
    out_dir = Path(out_dir)
    grid_dir = out_dir / "grids"
    single_dir = out_dir / "individual" / f"epoch_{epoch:03d}"

    grid_dir.mkdir(parents=True, exist_ok=True)

    samples = denormalize_images(samples.detach().cpu())[:9]
    n, _, h, w = samples.shape

    if save_individual:
        single_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            save_image(samples[i], single_dir / f"sample_{i:02d}_{w}x{h}px.png")

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue

        img = samples[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"sample {i:02d}\n{w}x{h}px", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"DDPM fixed samples | epoch {epoch:03d} | {w}x{h}px", fontsize=13)
    fig.tight_layout()
    fig.savefig(grid_dir / f"epoch_{epoch:03d}_fixed_3x3.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def p_sample(
    ddpm: DDPM,
    x_t: torch.Tensor,
    timestep: int,
    clip_denoised: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size = x_t.shape[0]
    device = x_t.device

    t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
    out = ddpm.p_mean_variance(x_t=x_t, t=t, clip_denoised=clip_denoised)

    mean = out["mean"]
    log_variance = out["log_variance"]

    if timestep == 0:
        noise = torch.zeros_like(x_t)
    else:
        noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)

    return mean + torch.exp(0.5 * log_variance) * noise


@torch.no_grad()
def sample_images(
    ddpm: DDPM,
    shape: tuple[int, int, int, int],
    device: torch.device,
    clip_denoised: bool = True,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    ddpm.eval()

    if initial_noise is None or tuple(initial_noise.shape) != tuple(shape):
        if initial_noise is not None:
            print("Fixed noise has the wrong shape, using fresh noise instead.")
        x_t = torch.randn(shape, device=device, generator=generator)
    else:
        x_t = initial_noise.clone().to(device=device)

    for timestep in reversed(range(ddpm.num_timesteps)):
        x_t = p_sample(
            ddpm=ddpm,
            x_t=x_t,
            timestep=timestep,
            clip_denoised=clip_denoised,
            generator=generator,
        )

    return x_t


@torch.no_grad()
def evaluate(
    ddpm: DDPM,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    ddpm.eval()
    losses: list[float] = []

    for batch_idx, (images, _) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        loss = ddpm.training_loss(images)["loss"]
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: str | Path,
    ddpm: DDPM,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    cfg: dict[str, Any],
    train_loss: float,
    val_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": ddpm.model.state_dict(),
            "ddpm_state_dict": ddpm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str | None,
    ddpm: DDPM,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    if not checkpoint_path:
        return 1, 0

    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Resume checkpoint not found: {path}")
        return 1, 0

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "ddpm_state_dict" in ckpt:
        ddpm.load_state_dict(ckpt["ddpm_state_dict"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ddpm.model.load_state_dict(ckpt["model_state_dict"])
    else:
        ddpm.model.load_state_dict(ckpt)

    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if isinstance(ckpt, dict) and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0

    print(f"Resumed from {path}")
    print(f"Starting at epoch {start_epoch}, global_step {global_step}")

    return start_epoch, global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to DDPM config YAML.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    data_cfg = cfg.get("data", {})
    diffusion_cfg = cfg.get("diffusion", {})
    training_cfg = cfg.get("training", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    log_cfg = cfg.get("logging", {})
    sample_cfg = cfg.get("sampling", {})

    logger = setup_logger(
        log_cfg.get("dir", "logs"),
        log_cfg.get("filename", "train_ddpm.log"),
    )

    metrics_csv_path = Path(log_cfg.get("dir", "logs")) / log_cfg.get(
        "csv_filename", "train_ddpm_metrics.csv"
    )

    sample_every_epochs = int(sample_cfg.get("every_epochs", training_cfg.get("sample_every", 1)))
    sample_num = int(sample_cfg.get("num_samples", 9))
    sample_seed = int(sample_cfg.get("seed", 999))
    sample_out_dir = sample_cfg.get("out_dir", "outputs/ddpm_samples")
    save_individual = bool(sample_cfg.get("save_individual", True))

    image_size = int(data_cfg.get("image_size", 32))
    channels = int(data_cfg.get("channels", 3))

    fixed_gen = torch.Generator(device=device)
    fixed_gen.manual_seed(sample_seed)
    fixed_noise = torch.randn(
        sample_num,
        channels,
        image_size,
        image_size,
        device=device,
        generator=fixed_gen,
    )

    logger.info("DDPM training started")
    logger.info("Device: %s", device)
    logger.info("Fixed sampling seed: %s", sample_seed)
    logger.info("Fixed sample noise shape: %s", tuple(fixed_noise.shape))

    train_loader, val_loader = build_dataloaders(data_cfg, seed=seed)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    schedule = build_schedule(diffusion_cfg)
    model = build_unet_from_config(cfg.get("model", {}))
    ddpm = DDPM(model=model, schedule=schedule).to(device)

    num_params = sum(p.numel() for p in ddpm.model.parameters())
    trainable_params = sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad)

    print(f"Model parameters:     {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Timesteps:            {ddpm.num_timesteps}")

    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 2e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip = training_cfg.get("grad_clip", None)
    log_every = int(training_cfg.get("log_every", 50))
    save_every = int(training_cfg.get("save_every", 1))
    val_max_batches = training_cfg.get("val_max_batches", None)
    train_max_batches = training_cfg.get("train_max_batches", None)
    timestep_sampling = str(diffusion_cfg.get("sampling_strategy", training_cfg.get("timestep_sampling", "uniform")))

    if val_max_batches is not None:
        val_max_batches = int(val_max_batches)

    if train_max_batches is not None:
        train_max_batches = int(train_max_batches)

    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(device.type, enabled=amp_enabled)

    start_epoch, global_step = load_checkpoint(
        checkpoint_path=args.resume,
        ddpm=ddpm,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print(f"Optimizer LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"Timestep sampling: {timestep_sampling}")

    checkpoint_dir = Path(checkpoint_cfg.get("dir", "checkpoints"))
    checkpoint_filename = checkpoint_cfg.get("filename", "ddpm_cifar10.pt")
    checkpoint_path = checkpoint_dir / checkpoint_filename
    last_checkpoint_path = checkpoint_dir / f"last_{checkpoint_filename}"
    best_checkpoint_path = checkpoint_dir / f"best_{checkpoint_filename}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_info_path = checkpoint_dir / f"{Path(checkpoint_filename).stem}_config_snapshot.json"

    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val_loss = float("inf")

    print("Starting DDPM training")

    for epoch in range(start_epoch, epochs + 1):
        ddpm.train()

        epoch_start = time.time()
        running_loss = 0.0
        seen_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            if train_max_batches is not None and batch_idx > train_max_batches:
                break

            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            t = sample_timesteps(
                batch_size=batch_size,
                num_timesteps=ddpm.num_timesteps,
                device=device,
                strategy=timestep_sampling,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                loss = ddpm.training_loss(images, t=t)["loss"]

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), max_norm=float(grad_clip))

            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            seen_batches += 1
            global_step += 1

            if batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, seen_batches)
                logger.info(
                    "Epoch %03d/%03d | batch %04d/%04d | loss %.6f | avg %.6f",
                    epoch,
                    epochs,
                    batch_idx,
                    len(train_loader),
                    loss_value,
                    avg_loss,
                )

        train_loss = running_loss / max(1, seen_batches)
        val_loss = evaluate(ddpm, val_loader, device=device, max_batches=val_max_batches)
        epoch_time = time.time() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        logger.info("-" * 80)
        logger.info("Epoch %03d/%03d", epoch, epochs)
        logger.info("  train loss: %.6f", train_loss)
        logger.info("  val   loss: %.6f", val_loss)
        logger.info("  lr:        %.8f", current_lr)
        logger.info("  time:      %.1fs", epoch_time)
        logger.info("-" * 80)

        append_metrics_csv(
            metrics_csv_path,
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": current_lr,
                "epoch_time_sec": float(epoch_time),
            },
        )

        if epoch == 1 or epoch % sample_every_epochs == 0 or epoch == epochs:
            logger.info("Generating fixed 3x3 sample grid for epoch %03d", epoch)

            sample_gen = torch.Generator(device=device)
            sample_gen.manual_seed(sample_seed)

            samples = sample_images(
                ddpm=ddpm,
                shape=tuple(fixed_noise.shape),
                device=device,
                clip_denoised=True,
                initial_noise=fixed_noise,
                generator=sample_gen,
            )

            save_fixed_3x3_sample_plot(
                samples=samples,
                epoch=epoch,
                out_dir=sample_out_dir,
                save_individual=save_individual,
            )

            logger.info(
                "Saved fixed samples: %s",
                Path(sample_out_dir) / "grids" / f"epoch_{epoch:03d}_fixed_3x3.png",
            )

        if epoch % save_every == 0:
            save_checkpoint(
                path=last_checkpoint_path,
                ddpm=ddpm,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            logger.info("Saved last checkpoint: %s", last_checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=best_checkpoint_path,
                ddpm=ddpm,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            save_checkpoint(
                path=checkpoint_path,
                ddpm=ddpm,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            logger.info("Saved best checkpoint: %s", best_checkpoint_path)
            logger.info("Updated main checkpoint: %s", checkpoint_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
