from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ScheduleType = Literal["linear"]


def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: tuple[int, ...],
) -> torch.Tensor:


    batch_size = timesteps.shape[0]

    out = values.gather(dim=0, index=timesteps)
    out = out.reshape(batch_size, *((1,) * (len(broadcast_shape) - 1)))

    return out


@dataclass
class DiffusionSchedule:

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: ScheduleType = "linear"
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")

        if self.beta_start <= 0:
            raise ValueError("beta_start must be positive.")

        if self.beta_end <= 0:
            raise ValueError("beta_end must be positive.")

        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end.")

        if self.schedule != "linear":
            raise ValueError(f"Unsupported schedule type: {self.schedule}")

        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.timesteps,
            device=self.device,
            dtype=torch.float32,
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # Useful later for DDPM reverse sampling.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self._compute_posterior_variance()

    def _compute_posterior_variance(self) -> torch.Tensor:

        alpha_bars_prev = torch.cat(
            [
                torch.ones(1, device=self.device, dtype=torch.float32),
                self.alpha_bars[:-1],
            ],
            dim=0,
        )

        posterior_variance = (
            self.betas
            * (1.0 - alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )

        return posterior_variance

    def to(self, device: torch.device | str) -> DiffusionSchedule:
        """
        Move all schedule tensors to a device.
        """
        return DiffusionSchedule(
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            schedule=self.schedule,
            device=device,
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Uniformly sample timesteps from {0, ..., T - 1}.

        Args:
            batch_size:
                Number of timesteps to sample.

        Returns:
            Tensor of shape [batch_size].
        """
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0).

        Args:
            x_0:
                Clean input images of shape [B, C, H, W].
                Expected value range for our project: [-1, 1].
            t:
                Timesteps of shape [B].
            noise:
                Optional Gaussian noise with same shape as x_0.
                If None, new standard Gaussian noise is sampled.

        Returns:
            Noisy images x_t with shape [B, C, H, W].
        """
        if x_0.ndim != 4:
            raise ValueError(f"Expected x_0 with shape [B, C, H, W], got {tuple(x_0.shape)}")

        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

        if x_0.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_0 batch is {x_0.shape[0]}, "
                f"t batch is {t.shape[0]}"
            )

        if t.device != self.device:
            t = t.to(self.device)

        if x_0.device != self.device:
            x_0 = x_0.to(self.device)

        if noise is None:
            noise = torch.randn_like(x_0)
        else:
            if noise.shape != x_0.shape:
                raise ValueError(
                    f"Noise shape {tuple(noise.shape)} does not match "
                    f"x_0 shape {tuple(x_0.shape)}"
                )
            noise = noise.to(self.device)

        sqrt_alpha_bar_t = _extract(
            self.sqrt_alpha_bars,
            t,
            x_0.shape,
        )

        sqrt_one_minus_alpha_bar_t = _extract(
            self.sqrt_one_minus_alpha_bars,
            t,
            x_0.shape,
        )

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t


def build_schedule_from_config(
    cfg: dict,
    device: torch.device | str = "cpu",
) -> DiffusionSchedule:
    """
    Build a DiffusionSchedule from the YAML config dictionary.

    Expected config section:

    diffusion:
      timesteps: 1000
      beta_start: 0.0001
      beta_end: 0.02
      schedule: "linear"
    """
    diffusion_cfg = cfg.get("diffusion", {})

    return DiffusionSchedule(
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 2e-2)),
        schedule=str(diffusion_cfg.get("schedule", "linear")),
        device=device,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = DiffusionSchedule(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule="linear",
        device=device,
    )

    batch_size = 8
    x_0 = torch.randn(batch_size, 3, 32, 32, device=device).clamp(-1, 1)
    t = schedule.sample_timesteps(batch_size)
    noise = torch.randn_like(x_0)

    x_t = schedule.q_sample(x_0=x_0, t=t, noise=noise)

    print("Device:", device)
    print("timesteps:", schedule.timesteps)
    print("betas shape:", schedule.betas.shape)
    print("alpha_bars shape:", schedule.alpha_bars.shape)
    print("x_0 shape:", x_0.shape)
    print("t shape:", t.shape)
    print("x_t shape:", x_t.shape)
    print("x_0 min/max:", float(x_0.min()), float(x_0.max()))
    print("x_t min/max:", float(x_t.min()), float(x_t.max()))
    print("sampled timesteps:", t[:8].tolist())