from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.data.cifar10 import get_cifar10_loaders
from src.diffusion.schedule import build_schedule_from_config
from src.models.classifier import build_classifier_from_config


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # For reproducibility in convolution operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def make_checkpoints(
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_acc: float,
        cfg: dict[str, Any],
)-> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "config": cfg,
    }


def save_checkpoint(
        checkpoint: dict[str, Any],
        checkpoint_dir: str | Path,
        filename: str,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, float]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_acc = checkpoint["best_val_acc"]
    return epoch, best_val_acc

def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    schedule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    use_amp: bool,
    grad_clip: float | None,
    log_every: int,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    progress = tqdm(
        train_loader,
        desc=f"Train epoch {epoch}/{num_epochs}",
        leave=False,
    )

    for step, (x_0, labels) in enumerate(progress, start=1):
        x_0 = x_0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = x_0.shape[0]

        t = schedule.sample_timesteps(batch_size)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0=x_0, t=t, noise=noise)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(x_t, t)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_acc = accuracy(logits.detach(), labels)

        running_loss += loss.item()
        running_acc += batch_acc
        num_batches += 1

        if step % log_every == 0:
            progress.set_postfix(
                {
                    "loss": f"{running_loss / num_batches:.4f}",
                    "acc": f"{running_acc / num_batches:.4f}",
                }
            )

    avg_loss = running_loss / max(num_batches, 1)
    avg_acc = running_acc / max(num_batches, 1)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    *,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    schedule,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    num_batches = 0

    # Timestep-bin accuracy. Useful because the classifier should be strong at
    # low/mid noise and will naturally become weaker near pure noise.
    bin_edges = [
        0,
        int(0.1 * schedule.timesteps),
        int(0.3 * schedule.timesteps),
        int(0.6 * schedule.timesteps),
        int(0.9 * schedule.timesteps),
        schedule.timesteps,
    ]

    bin_correct = {i: 0 for i in range(len(bin_edges) - 1)}
    bin_total = {i: 0 for i in range(len(bin_edges) - 1)}

    progress = tqdm(val_loader, desc="Validate", leave=False)

    for x_0, labels in progress:
        x_0 = x_0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = x_0.shape[0]

        t = schedule.sample_timesteps(batch_size)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0=x_0, t=t, noise=noise)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(x_t, t)
            loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)
        correct = preds == labels

        total_loss += loss.item()
        total_correct += correct.sum().item()
        total_examples += labels.numel()
        num_batches += 1

        for i in range(len(bin_edges) - 1):
            low = bin_edges[i]
            high = bin_edges[i + 1]

            mask = (t >= low) & (t < high)

            if mask.any():
                bin_correct[i] += correct[mask].sum().item()
                bin_total[i] += mask.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_correct / max(total_examples, 1)

    bin_acc = {}
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        key = f"t[{low},{high})"

        if bin_total[i] > 0:
            bin_acc[key] = bin_correct[i] / bin_total[i]
        else:
            bin_acc[key] = float("nan")

    return avg_loss, avg_acc, bin_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train noisy-image classifier for classifier-guided diffusion."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("C1 Noisy-Image Classifier Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA build: {torch.version.cuda}")

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    checkpoint_cfg = cfg.get("checkpoint", {})

    loaders = get_cifar10_loaders(
        root=data_cfg.get("root", "data"),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        download=bool(data_cfg.get("download", True)),
        seed=seed,
    )

    schedule = build_schedule_from_config(cfg, device=device)

    model = build_classifier_from_config(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    grad_clip = train_cfg.get("grad_clip", 1.0)
    grad_clip = None if grad_clip is None else float(grad_clip)

    epochs = int(train_cfg.get("epochs", 5))
    log_every = int(train_cfg.get("log_every", 50))

    checkpoint_dir = checkpoint_cfg.get("dir", "checkpoints")
    checkpoint_filename = checkpoint_cfg.get("filename", "classifier_cifar10.pt")
    last_checkpoint_filename = "last_" + checkpoint_filename

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume is not None:
        loaded_epoch, best_val_acc = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = loaded_epoch + 1
        print(f"Resumed from {args.resume}")
        print(f"Starting at epoch {start_epoch}")
        print(f"Best val acc so far: {best_val_acc:.4f}")

    num_params = sum(p.numel() for p in model.parameters())

    print("-" * 80)
    print(f"Model parameters: {num_params:,}")
    print(f"Train batches: {len(loaders.train)}")
    print(f"Val batches: {len(loaders.val)}")
    print(f"Timesteps: {schedule.timesteps}")
    print(f"AMP enabled: {use_amp}")
    print("-" * 80)

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=loaders.train,
            schedule=schedule,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            num_epochs=epochs,
            use_amp=use_amp,
            grad_clip=grad_clip,
            log_every=log_every,
        )

        val_loss, val_acc, bin_acc = validate(
            model=model,
            val_loader=loaders.val,
            schedule=schedule,
            device=device,
            use_amp=use_amp,
        )

        print(f"Epoch {epoch:03d}/{epochs:03d}")
        print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.4f}")
        print(f"  val   loss: {val_loss:.4f} | val   acc: {val_acc:.4f}")

        print("  val acc by timestep bin:")
        for key, value in bin_acc.items():
            print(f"    {key:14s}: {value:.4f}")

        last_checkpoint = make_checkpoints(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            cfg=cfg,
        )

        last_path = save_checkpoint(
            checkpoint=last_checkpoint,
            checkpoint_dir=checkpoint_dir,
            filename=last_checkpoint_filename,
        )

        print(f"  saved last checkpoint: {last_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            best_checkpoint = make_checkpoints(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                cfg=cfg,
            )

            best_path = save_checkpoint(
                checkpoint=best_checkpoint,
                checkpoint_dir=checkpoint_dir,
                filename=checkpoint_filename,
            )

            print(f"  saved best checkpoint: {best_path}")
            print(f"  new best val acc: {best_val_acc:.4f}")

        print("-" * 80)

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {Path(checkpoint_dir) / checkpoint_filename}")


if __name__ == "__main__":
    main()