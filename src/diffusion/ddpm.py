from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_schedule_value(schedule: Any, names: tuple[str, ...]) -> torch.Tensor:
    """Pick the first matching tensor from a schedule object or dict."""
    if isinstance(schedule, dict):
        for name in names:
            if name in schedule:
                return torch.as_tensor(schedule[name], dtype=torch.float32)
    else:
        for name in names:
            if hasattr(schedule, name):
                return torch.as_tensor(getattr(schedule, name), dtype=torch.float32)

    # No custom exception here. If the schedule is wrong, this will fail naturally.
    return torch.as_tensor(schedule[names[0]], dtype=torch.float32)


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Take the values for the current timesteps and reshape them for image broadcasting."""
    if t.ndim == 0:
        t = t[None]

    values = values.to(t.device)
    out = values.gather(0, t.long())
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    """
    Thin DDPM wrapper around the denoising network.

    The network predicts the noise epsilon from (x_t, t). This class handles the
    diffusion math around it: forward noising, training loss and reverse-step stats.
    """

    def __init__(self, model: nn.Module, schedule: Any):
        super().__init__()

        self.model = model

        betas = get_schedule_value(schedule, ("betas", "beta"))
        alphas = get_schedule_value(schedule, ("alphas", "alpha"))
        alpha_bars = get_schedule_value(
            schedule,
            ("alpha_bars", "alphas_cumprod", "alpha_bar", "alphas_bar"),
        )

        self.num_timesteps = int(betas.shape[0])

        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_bars[:-1]],
            dim=0,
        )

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0)
        )

        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer(
            "sqrt_recipm1_alpha_bars",
            torch.sqrt(1.0 / alpha_bars - 1.0),
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward noising: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)

        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a clean-image estimate from x_t and predicted noise."""
        sqrt_recip_alpha_bar_t = extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        sqrt_recipm1_alpha_bar_t = extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)

        return sqrt_recip_alpha_bar_t * x_t - sqrt_recipm1_alpha_bar_t * eps

    def q_posterior_mean_variance(
        self,
        x_0_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Gaussian parameters for q(x_{t-1} | x_t, x_0)."""
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)

        mean = coef1 * x_0_pred + coef2 * x_t
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, variance, log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One reverse-process estimate: predict noise, estimate x_0, compute mean/variance."""
        eps_pred = self.model(x_t, t)
        x_0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        mean, variance, log_variance = self.q_posterior_mean_variance(
            x_0_pred=x_0_pred,
            x_t=x_t,
            t=t,
        )

        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
            "x_0_pred": x_0_pred,
            "eps_pred": eps_pred,
        }

    def training_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """DDPM epsilon-prediction loss."""
        batch_size = x_0.shape[0]
        device = x_0.device

        if t is None:
            t = torch.randint(
                low=0,
                high=self.num_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        eps_pred = self.model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)

        return {
            "loss": loss,
            "x_t": x_t,
            "t": t,
            "noise": noise,
            "eps_pred": eps_pred,
        }

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        return self.training_loss(x_0)["loss"]
