from __future__ import annotations
#######Training usage ######################################################################
from dataclasses import dataclass
from typing import Literal

import torch


ScheduleType = Literal["linear"]
SamplingStrategyType = Literal["uniform", "linear_decay", "inverse_power", "exponential", "cosine"]


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
    sampling_strategy: SamplingStrategyType = "uniform"
    device: torch.device | str = "cpu"



    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
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
            sampling_strategy=self.sampling_strategy,
            device=device,
        )



    def _compute_timestep_weights(self) -> torch.Tensor:
       
        t_array = torch.arange(self.timesteps, dtype=torch.float32, device=self.device)
        t_normalized = t_array / (self.timesteps - 1)
        
        if self.sampling_strategy == "uniform":
            # Uniform distribution
            weights = torch.ones(self.timesteps, device=self.device, dtype=torch.float32)
        
        elif self.sampling_strategy == "linear_decay":
            # Linear decay: 1 - t/T
            weights = 1.0 - t_normalized
        
        elif self.sampling_strategy == "inverse_power":
            # Inverse power: 1/sqrt(t+1)
            weights = 1.0 / torch.sqrt(t_array + 1.0)
        
        elif self.sampling_strategy == "exponential":
            # Exponential decay: exp(-3 * t/T)
            weights = torch.exp(-3.0 * t_normalized)
        
        elif self.sampling_strategy == "cosine":
            # Cosine: cos(pi/2 * t/T)
            weights = torch.cos(t_normalized * torch.tensor(torch.pi / 2.0, device=self.device))
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights
    



    def sample_timesteps(self, batch_size: int) -> torch.Tensor:

        weights = self._compute_timestep_weights()
        indices = torch.multinomial(
            weights,
            batch_size,
            replacement=True,
        )
        
        return indices
    


    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if t.device != self.device:
            t = t.to(self.device)

        if x_0.device != self.device:
            x_0 = x_0.to(self.device)

        if noise is None:
            noise = torch.randn_like(x_0)
        else:
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
    diffusion_cfg = cfg.get("diffusion", {})

    return DiffusionSchedule(
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 2e-2)),
        schedule=str(diffusion_cfg.get("schedule", "linear")),
        sampling_strategy=str(diffusion_cfg.get("sampling_strategy", "uniform")),
        device=device,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = DiffusionSchedule(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule="linear",
        sampling_strategy="linear_decay",
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