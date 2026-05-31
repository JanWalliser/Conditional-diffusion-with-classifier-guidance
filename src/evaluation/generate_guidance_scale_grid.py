from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image


CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "car": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

CIFAR10_ID_TO_NAME = {
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


DDPM_MODEL_BUILDERS = [
    ("src.training.train_ddpm", "build_model_from_config"),
    ("src.training.train_ddpm", "build_unet_from_config"),
    ("src.models.unet", "build_model_from_config"),
    ("src.models.unet", "build_unet_from_config"),
    ("src.models.ddpm_unet", "build_model_from_config"),
    ("src.models.ddpm_unet", "build_unet_from_config"),
    ("src.models.ddpm", "build_model_from_config"),
]

CLASSIFIER_BUILDERS = [
    ("src.models.classifier", "build_classifier_from_config"),
    ("src.training.train_classifier", "build_classifier_from_config"),
    ("src.models.classifier", "NoisyImageClassifier"),
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_symbol(module_name: str, symbol_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def try_build_from_config(
    candidates: list[tuple[str, str]],
    cfg: dict[str, Any],
    what: str,
) -> torch.nn.Module:
    errors: list[str] = []

    for module_name, symbol_name in candidates:
        try:
            symbol = import_symbol(module_name, symbol_name)
        except Exception as exc:
            errors.append(f"{module_name}.{symbol_name}: import failed: {exc}")
            continue

        call_attempts = [
            lambda: symbol(cfg),
            lambda: symbol(cfg.get("model", cfg)),
            lambda: symbol(**cfg.get("model", cfg)),
        ]

        for attempt in call_attempts:
            try:
                model = attempt()
                if isinstance(model, torch.nn.Module):
                    return model
            except Exception as exc:
                errors.append(f"{module_name}.{symbol_name}: build failed: {exc}")

    msg = "\n".join(errors[-16:])
    raise RuntimeError(
        f"Could not build {what} from config. "
        f"Add your builder to the candidate list.\nLast errors:\n{msg}"
    )


def strip_prefix_if_present(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state:
        return state
    if all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def extract_state_dict(payload: Any, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if prefer_ema:
            for key in ("ema_state_dict", "ema_model_state_dict", "model_ema", "ema"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
            "ddpm",
            "classifier",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                return value

        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload

    raise RuntimeError("Checkpoint does not contain a recognizable state_dict.")


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool,
    prefer_ema: bool = False,
) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    state = extract_state_dict(payload, prefer_ema=prefer_ema)

    for prefix in ("module.", "model.", "ddpm.", "classifier.", "net.", "network."):
        state = strip_prefix_if_present(state, prefix)

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError:
        if strict:
            raise
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[WARN] non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")


class DDPMLinearSchedule:
    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
        schedule: str = "linear",
    ):
        if schedule.lower() != "linear":
            raise ValueError(
                f"This script currently implements only the linear DDPM schedule, got: {schedule!r}"
            )

        self.timesteps = int(timesteps)
        self.betas = torch.linspace(beta_start, beta_end, self.timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bars[:-1]],
            dim=0,
        )

        self.sqrt_recip_alpha_bars = torch.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recipm1_alpha_bars = torch.sqrt(1.0 / self.alpha_bars - 1.0)

        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        )


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = v.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def call_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    attempts = []

    if y is not None:
        attempts.extend(
            [
                lambda: model(x, t, y),
                lambda: model(x, t, class_labels=y),
                lambda: model(x, t, labels=y),
                lambda: model(x, t, y=y),
                lambda: model(x, t, model_kwargs={"class_labels": y}),
            ]
        )

    attempts.append(lambda: model(x, t))

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            out = attempt()
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"Could not call DDPM model. Last error: {last_exc}") from last_exc

    if isinstance(out, dict):
        for key in ("eps", "epsilon", "pred_noise", "noise", "sample", "out"):
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(f"DDPM model returned dict without known prediction key: {out.keys()}")

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(f"DDPM model output is not a tensor: {type(out)}")

    return out


def call_classifier(
    classifier: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    attempts = [
        lambda: classifier(x, t),
        lambda: classifier(x_t=x, t=t),
        lambda: classifier(x, timestep=t),
    ]

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            out = attempt()
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"Could not call classifier. Last error: {last_exc}") from last_exc

    if isinstance(out, dict):
        for key in ("logits", "pred", "out"):
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(f"Classifier returned dict without known logits key: {out.keys()}")

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(f"Classifier output is not a tensor: {type(out)}")

    return out


def guidance_multiplier(t: torch.Tensor, timesteps: int, mode: str) -> torch.Tensor:
    mode = mode.lower()

    if mode in {"constant", "const"}:
        return torch.ones_like(t, dtype=torch.float32)

    # t is in [0, T-1]. 0 is clean end, T-1 is high-noise start.
    # sin(pi * t/T) gives low guidance near both ends and maximum guidance in the middle.
    x = t.float() / float(max(timesteps - 1, 1))
    sin_value = torch.sin(math.pi * x)

    if mode in {"sin", "sine"}:
        return sin_value

    if mode in {"sin2", "sin^2", "sine2", "sine_squared"}:
        return sin_value.square()

    raise ValueError(f"Unknown schedule mode: {mode}")


def classifier_gradient(
    classifier: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        logits = call_classifier(classifier, x_in, t)
        log_probs = F.log_softmax(logits, dim=1)
        selected = log_probs.gather(1, class_labels.view(-1, 1)).sum()
        grad = torch.autograd.grad(selected, x_in)[0]
    return grad.detach()


@torch.no_grad()
def predict_x0_from_eps(
    schedule: DDPMLinearSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    clip_denoised: bool,
) -> torch.Tensor:
    sqrt_recip_alpha_bar = extract(schedule.sqrt_recip_alpha_bars, t, x_t.shape)
    sqrt_recipm1_alpha_bar = extract(schedule.sqrt_recipm1_alpha_bars, t, x_t.shape)
    x0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * eps

    if clip_denoised:
        x0 = x0.clamp(-1.0, 1.0)

    return x0


def p_sample(
    model: torch.nn.Module,
    classifier: torch.nn.Module | None,
    schedule: DDPMLinearSchedule,
    x_t: torch.Tensor,
    timestep: int,
    class_labels: torch.Tensor,
    base_guidance_scale: float,
    scaling_mode: str,
    clip_denoised: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = x_t.shape[0]
    device = x_t.device
    t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        eps = call_model(model, x_t, t, class_labels)
        pred_x0 = predict_x0_from_eps(schedule, x_t, t, eps, clip_denoised)

        coef1 = extract(schedule.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(schedule.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * pred_x0 + coef2 * x_t
        variance = extract(schedule.posterior_variance, t, x_t.shape)

    if classifier is not None and float(base_guidance_scale) != 0.0:
        grad = classifier_gradient(classifier, x_t, t, class_labels)
        mult = guidance_multiplier(t, schedule.timesteps, scaling_mode).view(batch_size, 1, 1, 1)
        scale_t = float(base_guidance_scale) * mult
        mean = mean + variance * scale_t * grad

    if timestep == 0:
        return mean

    noise = torch.randn(
        x_t.shape,
        device=device,
        dtype=x_t.dtype,
        generator=generator,
    )
    return mean + torch.sqrt(variance) * noise


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()

    gen.manual_seed(seed)
    return gen


def sample_batch(
    model: torch.nn.Module,
    classifier: torch.nn.Module | None,
    schedule: DDPMLinearSchedule,
    class_id: int,
    num_samples: int,
    image_size: int,
    guidance_scale: float,
    scaling_mode: str,
    seed: int,
    device: torch.device,
    clip_denoised: bool,
    show_progress: bool,
) -> torch.Tensor:
    # Important: generator is reset per scale.
    # This gives the same initial noise and same stochastic noise sequence for each scale.
    generator = make_generator(device, seed)

    x = torch.randn(
        (num_samples, 3, image_size, image_size),
        device=device,
        generator=generator,
    )
    y = torch.full((num_samples,), class_id, device=device, dtype=torch.long)

    iterator: Iterable[int] = range(schedule.timesteps - 1, -1, -1)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=schedule.timesteps,
                desc=f"{scaling_mode}, s={guidance_scale}",
            )
        except Exception:
            pass

    for timestep in iterator:
        x = p_sample(
            model=model,
            classifier=classifier,
            schedule=schedule,
            x_t=x,
            timestep=timestep,
            class_labels=y,
            base_guidance_scale=guidance_scale,
            scaling_mode=scaling_mode,
            clip_denoised=clip_denoised,
            generator=generator,
        )

    return x.detach()


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def scale_tag(scale: float) -> str:
    if float(scale).is_integer():
        return str(int(scale))
    return str(scale).replace(".", "p")


def mode_tag(mode: str) -> str:
    return mode.lower().replace("^", "").replace(" ", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-seed classifier-guidance samples with different scales."
    )

    parser.add_argument("--ddpm-checkpoint", required=True, type=str)
    parser.add_argument("--ddpm-config", required=True, type=str)
    parser.add_argument("--classifier-checkpoint", default=None, type=str)
    parser.add_argument("--classifier-config", default=None, type=str)

    cls = parser.add_mutually_exclusive_group(required=True)
    cls.add_argument("--class-id", type=int, choices=range(10))
    cls.add_argument("--class-name", type=str, choices=sorted(CIFAR10_CLASSES.keys()))

    parser.add_argument(
        "--schedule",
        type=str,
        default="constant",
        choices=["constant", "const", "sin", "sin2", "sin^2"],
        help="Guidance schedule: constant, sin, or sin2.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.0, 3.0, 5.0, 10.0],
        help="Guidance scales to compare, e.g. --scales 0 3 5 10.",
    )

    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--grid-cols", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="results/guidance_scale_grid")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--clip-denoised", action="store_true", default=True)
    parser.add_argument("--no-clip-denoised", action="store_false", dest="clip_denoised")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--use-ema", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    if args.class_name is not None:
        class_id = CIFAR10_CLASSES[args.class_name]
    else:
        class_id = int(args.class_id)

    class_name = CIFAR10_ID_TO_NAME[class_id]

    needs_classifier = any(float(s) != 0.0 for s in args.scales)
    if needs_classifier and (args.classifier_checkpoint is None or args.classifier_config is None):
        raise ValueError(
            "For non-zero guidance scales you must pass "
            "--classifier-checkpoint and --classifier-config."
        )

    ddpm_cfg = load_yaml(args.ddpm_config)
    diffusion_cfg = ddpm_cfg.get("diffusion", {})

    schedule = DDPMLinearSchedule(
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
        schedule=str(diffusion_cfg.get("schedule", "linear")),
        device=device,
    )

    print("[INFO] Building DDPM model...")
    ddpm_model = try_build_from_config(DDPM_MODEL_BUILDERS, ddpm_cfg, "DDPM model")
    ddpm_model.to(device)
    load_checkpoint_into_model(
        ddpm_model,
        args.ddpm_checkpoint,
        device,
        strict=not args.non_strict_load,
        prefer_ema=args.use_ema,
    )
    ddpm_model.eval()

    classifier = None
    if needs_classifier:
        print("[INFO] Building classifier...")
        classifier_cfg = load_yaml(args.classifier_config)
        classifier = try_build_from_config(CLASSIFIER_BUILDERS, classifier_cfg, "classifier")
        classifier.to(device)
        load_checkpoint_into_model(
            classifier,
            args.classifier_checkpoint,
            device,
            strict=not args.non_strict_load,
        )
        classifier.eval()

        for p in classifier.parameters():
            p.requires_grad_(False)

    run_root = (
        Path(args.output_dir)
        / f"class_{class_id}_{class_name}"
        / f"seed_{args.seed}"
        / mode_tag(args.schedule)
    )
    run_root.mkdir(parents=True, exist_ok=True)

    first_sample_per_scale: list[torch.Tensor] = []
    metadata: dict[str, Any] = {
        "class_id": class_id,
        "class_name": class_name,
        "seed": args.seed,
        "schedule": args.schedule,
        "scales": [float(s) for s in args.scales],
        "num_samples": args.num_samples,
        "image_size": args.image_size,
        "ddpm_checkpoint": args.ddpm_checkpoint,
        "ddpm_config": args.ddpm_config,
        "classifier_checkpoint": args.classifier_checkpoint,
        "classifier_config": args.classifier_config,
        "outputs": {},
    }

    for scale in args.scales:
        scale = float(scale)
        tag = f"s{scale_tag(scale)}"
        out_dir = run_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[INFO] Sampling class={class_name} ({class_id}), "
            f"schedule={args.schedule}, scale={scale}, seed={args.seed}"
        )

        samples = sample_batch(
            model=ddpm_model,
            classifier=classifier,
            schedule=schedule,
            class_id=class_id,
            num_samples=args.num_samples,
            image_size=args.image_size,
            guidance_scale=scale,
            scaling_mode=args.schedule,
            seed=args.seed,
            device=device,
            clip_denoised=args.clip_denoised,
            show_progress=not args.no_progress,
        )

        images = to_image_range(samples)

        grid_path = out_dir / f"{class_name}_{mode_tag(args.schedule)}_{tag}_seed{args.seed}.png"
        save_image(images, grid_path, nrow=args.grid_cols)

        first_sample_per_scale.append(images[:1].cpu())

        if args.save_individual or args.num_samples > 1:
            indiv_dir = out_dir / "individual"
            indiv_dir.mkdir(exist_ok=True)

            for idx, img in enumerate(images):
                save_image(
                    img,
                    indiv_dir / f"{class_name}_{mode_tag(args.schedule)}_{tag}_sample{idx:02d}_seed{args.seed}.png",
                )

        metadata["outputs"][tag] = str(grid_path)

    combined = torch.cat(first_sample_per_scale, dim=0)
    combined_path = run_root / f"{class_name}_{mode_tag(args.schedule)}_scales_{'_'.join(scale_tag(s) for s in args.scales)}_seed{args.seed}_grid.png"
    save_image(combined, combined_path, nrow=len(args.scales), padding=4)

    metadata["combined_grid"] = str(combined_path)
    metadata_path = run_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(f"Images saved under: {run_root}")
    print(f"Combined scale grid: {combined_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()