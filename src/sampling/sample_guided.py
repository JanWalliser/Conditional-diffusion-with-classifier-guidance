import argparse
import math
from pathlib import Path

import torch
import yaml
from torchvision.utils import make_grid, save_image

from src.diffusion.ddpm import DDPM
from src.diffusion.sampler import DDPMSampler
from src.models.unet import build_unet_from_config


CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_linear_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }


def build_cosine_schedule(timesteps, s=0.008):
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


def build_schedule(cfg):
    name = str(cfg.get("schedule", "linear")).lower()
    timesteps = int(cfg.get("timesteps", 1000))

    if name == "cosine":
        return build_cosine_schedule(timesteps)

    return build_linear_schedule(
        timesteps=timesteps,
        beta_start=float(cfg.get("beta_start", 0.0001)),
        beta_end=float(cfg.get("beta_end", 0.02)),
    )


def denormalize_images(x):
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def make_generator(device, seed):
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_int_list(value):
    value = value.strip().lower()

    if value == "all":
        return list(range(10))

    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("ddpm_state_dict", "model_state_dict", "classifier_state_dict", "state_dict", "model", "net"):
            state = ckpt.get(key)

            if isinstance(state, dict):
                return state

        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    return ckpt


def load_ddpm(checkpoint_path, device, config_path=None):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = None

    if config_path is not None:
        cfg = load_yaml(config_path)
    elif isinstance(ckpt, dict):
        cfg = ckpt.get("config")

    if cfg is None:
        print("DDPM config fehlt. Starte das Script mit --ddpm-config.")
        return None

    schedule = build_schedule(cfg["diffusion"])
    model = build_unet_from_config(cfg["model"])
    ddpm = DDPM(model=model, schedule=schedule).to(device)

    if isinstance(ckpt, dict) and "ddpm_state_dict" in ckpt:
        ddpm.load_state_dict(ckpt["ddpm_state_dict"], strict=True)
    else:
        ddpm.model.load_state_dict(get_state_dict(ckpt), strict=True)

    ddpm.eval()
    return ddpm


def build_classifier_from_config_fallback(cfg):
    from src.models.classifier import NoisyUNetClassifier

    return NoisyUNetClassifier(
        in_channels=cfg.get("in_channels", 3),
        num_classes=cfg.get("num_classes", 10),
        input_size=cfg.get("input_size", 32),
        base_channels=cfg.get("base_channels", 96),
        time_embedding_dim=cfg.get("time_embedding_dim", 384),
        channel_mults=tuple(cfg.get("channel_mults", (1, 2, 4))),
        blocks_per_stage=cfg.get("blocks_per_stage", 2),
        dropout=cfg.get("dropout", 0.1),
        attention_resolutions=tuple(cfg.get("attention_resolutions", (8,))),
        channels_per_head=cfg.get("channels_per_head", 64),
    )


def build_classifier_from_cfg(cfg):
    try:
        from src.models.classifier import build_classifier_from_config
        return build_classifier_from_config(cfg)
    except Exception:
        return build_classifier_from_config_fallback(cfg)


def looks_like_noisy_unet_classifier(state_dict):
    keys = (
        "input_conv.weight",
        "time_mlp.0.weight",
        "pool.positional_embedding",
        "head.weight",
    )

    return isinstance(state_dict, dict) and all(key in state_dict for key in keys)


def build_noisy_unet_classifier_from_state_dict(state_dict, cfg=None):
    from src.models.classifier import NoisyUNetClassifier

    in_channels = int(state_dict["input_conv.weight"].shape[1])
    base_channels = int(state_dict["input_conv.weight"].shape[0])
    time_embedding_dim = int(state_dict["time_mlp.0.weight"].shape[0])
    num_classes = int(state_dict["head.weight"].shape[0])

    input_size = int(cfg.get("input_size", 32)) if cfg else 32
    dropout = float(cfg.get("dropout", 0.0)) if cfg else 0.0
    channels_per_head = int(cfg.get("channels_per_head", 64)) if cfg else 64

    down_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("down.") and len(key.split(".")) > 2
        }
    )

    resolution = input_size
    stage_mults = []
    stage_counts = []
    attention_resolutions = set()

    stage_out_channels = None
    stage_blocks = 0

    for idx in down_indices:
        prefix = f"down.{idx}"

        if f"{prefix}.conv2.weight" in state_dict:
            stage_out_channels = int(state_dict[f"{prefix}.conv2.weight"].shape[0])
            stage_blocks += 1

        elif f"{prefix}.qkv.weight" in state_dict:
            attention_resolutions.add(resolution)

        elif f"{prefix}.op.weight" in state_dict:
            if stage_out_channels is not None:
                stage_mults.append(stage_out_channels // base_channels)
                stage_counts.append(stage_blocks)

            stage_out_channels = None
            stage_blocks = 0
            resolution //= 2

    if stage_out_channels is not None:
        stage_mults.append(stage_out_channels // base_channels)
        stage_counts.append(stage_blocks)

    if not stage_mults and cfg:
        stage_mults = list(cfg.get("channel_mults", (1, 2, 4)))

    if not stage_counts and cfg:
        stage_counts = [int(cfg.get("blocks_per_stage", 2))]

    channel_mults = tuple(stage_mults)
    blocks_per_stage = stage_counts[0] if stage_counts else 2

    if not attention_resolutions and cfg:
        attention_resolutions = set(cfg.get("attention_resolutions", (8,)))

    attention_resolutions = tuple(sorted(attention_resolutions, reverse=True))

    print("Detected NoisyUNetClassifier checkpoint:")
    print(f"  in_channels:           {in_channels}")
    print(f"  num_classes:           {num_classes}")
    print(f"  input_size:            {input_size}")
    print(f"  base_channels:         {base_channels}")
    print(f"  time_embedding_dim:    {time_embedding_dim}")
    print(f"  channel_mults:         {channel_mults}")
    print(f"  blocks_per_stage:      {blocks_per_stage}")
    print(f"  attention_resolutions: {attention_resolutions}")
    print(f"  channels_per_head:     {channels_per_head}")

    return NoisyUNetClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=input_size,
        base_channels=base_channels,
        time_embedding_dim=time_embedding_dim,
        channel_mults=channel_mults,
        blocks_per_stage=blocks_per_stage,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        channels_per_head=channels_per_head,
    )


def load_classifier(checkpoint_path, device, config_path=None):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = None

    if config_path is not None:
        model_cfg = load_yaml(config_path).get("model")
    elif isinstance(ckpt, dict) and "config" in ckpt:
        model_cfg = ckpt["config"].get("model")

    state_dict = get_state_dict(ckpt)

    if looks_like_noisy_unet_classifier(state_dict):
        classifier = build_noisy_unet_classifier_from_state_dict(state_dict, model_cfg)
    elif model_cfg is not None:
        classifier = build_classifier_from_cfg(model_cfg)
    else:
        print("Classifier config fehlt. Sampling läuft ohne Classifier Guidance.")
        return None

    classifier = classifier.to(device)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    for param in classifier.parameters():
        param.requires_grad_(False)

    return classifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ddpm-checkpoint", type=str, required=True)
    parser.add_argument("--ddpm-config", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-config", type=str, default=None)

    parser.add_argument("--class-labels", type=str, default="0,1,3,5,8")
    parser.add_argument("--guidance-scales", type=str, default="0,1,2,4")
    parser.add_argument("--num-images", type=int, default=9)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--out-dir", type=str, default="outputs/guided_samples")
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddpm = load_ddpm(
        checkpoint_path=args.ddpm_checkpoint,
        device=device,
        config_path=args.ddpm_config,
    )

    if ddpm is None:
        return

    classifier = None

    if args.classifier_checkpoint:
        classifier = load_classifier(
            checkpoint_path=args.classifier_checkpoint,
            device=device,
            config_path=args.classifier_config,
        )

    if classifier is not None:
        print("Loaded classifier for guidance.")
    else:
        print("No classifier loaded. Only unguided sampling is available.")

    sampler = DDPMSampler(
        ddpm=ddpm,
        classifier=classifier,
        guidance_scale=0.0,
        freeze_classifier=True,
    )

    class_labels = parse_int_list(args.class_labels)
    guidance_scales = parse_float_list(args.guidance_scales)

    shape = (args.num_images, 3, 32, 32)
    nrow = int(math.sqrt(args.num_images))

    if nrow * nrow != args.num_images:
        nrow = args.num_images

    print(f"Classes: {class_labels}")
    print(f"Guidance scales: {guidance_scales}")
    print(f"Samples per grid: {args.num_images}")

    for class_label in class_labels:
        if class_label not in CIFAR10_CLASSES:
            print(f"Skipping unknown class label: {class_label}")
            continue

        class_name = CIFAR10_CLASSES[class_label]
        print(f"Sampling class {class_label}: {class_name}")

        init_generator = make_generator(device, args.seed + 10_000 * class_label)
        initial_noise = torch.randn(shape, device=device, generator=init_generator)

        comparison_rows = []

        for scale in guidance_scales:
            step_generator = make_generator(device, args.seed + 20_000 * class_label)

            labels = torch.full(
                size=(args.num_images,),
                fill_value=class_label,
                device=device,
                dtype=torch.long,
            )

            samples = sampler.sample(
                shape=shape,
                class_labels=labels,
                guidance_scale=scale,
                clip_denoised=args.clip_denoised,
                initial_noise=initial_noise,
                generator=step_generator,
            )

            samples = denormalize_images(samples.detach().cpu())

            grid = make_grid(samples, nrow=nrow, padding=2)
            scale_name = str(scale).replace(".", "p")
            sample_path = out_dir / f"class_{class_label}_{class_name}_scale_{scale_name}.png"

            save_image(grid, sample_path)
            print(f"Saved: {sample_path}")

            comparison_rows.append(samples)

        comparison = torch.cat(comparison_rows, dim=0)
        comparison_grid = make_grid(comparison, nrow=args.num_images, padding=2)
        comparison_path = out_dir / f"class_{class_label}_{class_name}_scale_comparison.png"

        save_image(comparison_grid, comparison_path)
        print(f"Saved comparison: {comparison_path}")

    print("Sampling complete.")


if __name__ == "__main__":
    main()
