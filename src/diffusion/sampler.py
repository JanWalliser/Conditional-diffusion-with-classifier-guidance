from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion.ddpm import DDPM

num_timesteps = 1000

class DDPMSampler:
    def __init__(
        self,
        ddpm: DDPM,
        classifier: Optional[nn.Module] = None,
        guidance_scale: float = 0.0,
        freeze_classifier: bool = True,
    ):
        self.ddpm = ddpm
        self.classifier = classifier
        self.guidance_scale = float(guidance_scale)

        self.ddpm.eval()

        if self.classifier is not None:
            self.classifier.eval()

            if freeze_classifier:
                for param in self.classifier.parameters():
                    param.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.ddpm.parameters()).device

    def _normalise_class_labels(
        self,
        class_labels: Union[int, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(class_labels, int):
            return torch.full(
                size=(batch_size,),
                fill_value=class_labels,
                device=device,
                dtype=torch.long,
            )

        class_labels = class_labels.to(device=device, dtype=torch.long)

        if class_labels.dim() == 0:
            class_labels = class_labels.expand(batch_size)

        return class_labels

    def _extract_logits(self, classifier_output: Any) -> torch.Tensor:
        if isinstance(classifier_output, torch.Tensor):
            return classifier_output

        if isinstance(classifier_output, dict):
            if "logits" in classifier_output:
                return classifier_output["logits"]

        if isinstance(classifier_output, (tuple, list)):
            return classifier_output[0]

    def classifier_gradient(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        
        """
        Computes:
            Nabla_{x_t} log p(y | x_t, t)
        """

        batch_size = x_t.shape[0]

        labels = self._normalise_class_labels(
            class_labels=class_labels,
            batch_size=batch_size,
            device=x_t.device,
        )

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)

            classifier_output = self.classifier(x_in, t)
            logits = self._extract_logits(classifier_output)

            log_probs = F.log_softmax(logits, dim=1)
            selected = log_probs[
                torch.arange(batch_size, device=x_t.device),
                labels,
            ]

            grad = torch.autograd.grad(
                outputs=selected.sum(),
                inputs=x_in,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        return grad.detach()

    def _p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]],
        clip_denoised: bool,
    ) -> Dict[str, torch.Tensor]:
        
        if model_kwargs is None:
            return self.ddpm.p_mean_variance(
                x_t=x_t,
                t=t,
                clip_denoised=clip_denoised,
            )

        try:
            return self.ddpm.p_mean_variance(
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
            )
        except TypeError:
            return self.ddpm.p_mean_variance(
                x_t=x_t,
                t=t,
                clip_denoised=clip_denoised,
            )

    #Helper functions for schedule
    def _num_timesteps(self) -> int:
        if hasattr(self.ddpm, "num_timesteps"):
            return int(self.ddpm.num_timesteps)

        if hasattr(self.ddpm, "schedule"):
            schedule = self.ddpm.schedule

            if isinstance(schedule, dict):
                if "betas" in schedule:
                    return int(schedule["betas"].shape[0])
                if "alpha_bars" in schedule:
                    return int(schedule["alpha_bars"].shape[0])

            if hasattr(schedule, "timesteps"):
                return int(schedule.timesteps)

        raise AttributeError(
            "Could not infer number of diffusion timesteps. "
            "Expected self.ddpm.num_timesteps or self.ddpm.schedule."
        )


    def _guidance_scale_at_t(
        self,
        base_scale: float,
        timestep: int,
        guidance_schedule: str,
    ) -> float:
        if base_scale == 0.0:
            return 0.0

        mode = guidance_schedule.lower()
        num_timesteps = self._num_timesteps()
        tau = float(timestep) / float(max(num_timesteps - 1, 1))

        if mode in ("constant", "const"):
            return base_scale

        if mode in ("sin", "sine"):
            return base_scale * math.sin(math.pi * tau)

        if mode in ("sin2", "sin^2", "sine2", "sine_squared"):
            return base_scale * (math.sin(math.pi * tau) ** 2)

        raise ValueError(
            f"Unknown guidance_schedule={guidance_schedule!r}. "
            "Use one of: constant, sin, sin2."
        )
    




    def p_sample(
        self,
        x_t: torch.Tensor,
        timestep: int,
        class_labels: Optional[Union[int, torch.Tensor]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        clip_denoised: bool = True,
        guidance_scale: Optional[float] = None,
        guidance_schedule: str = "sin",
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One reverse DDPM step:
            x_t -> x_{t-1}
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        t = torch.full(
            size=(batch_size,),
            fill_value=timestep,
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            out = self._p_mean_variance(
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
            )

            mean = out["mean"]
            variance = out["variance"]
            log_variance = out["log_variance"]

        # Base scale: either sampler default or scale passed to this call
        base_scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)

        if self.classifier is not None and class_labels is not None and base_scale != 0.0:
            grad = self.classifier_gradient(
                x_t=x_t,
                t=t,
                class_labels=class_labels,
            )

            # Actual timestep-dependent guidance scale
            scale_t = self._guidance_scale_at_t(
                base_scale=base_scale,
                timestep=timestep,
                guidance_schedule=guidance_schedule,
            )

            mean = mean + scale_t * variance * grad

        if timestep == 0:
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )

        sample = mean + torch.exp(0.5 * log_variance) * noise

        return {
            "sample": sample,
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
            "x_0_pred": out["x_0_pred"],
            "eps_pred": out["eps_pred"],
        }

    def sample(
        self,
        shape: tuple[int, int, int, int],
        class_labels: Optional[Union[int, torch.Tensor]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        clip_denoised: bool = True,
        guidance_scale: Optional[float] = None,
        guidance_schedule: str = "sin",
        return_intermediates: bool = False,
        intermediate_every: int = 100,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        self.ddpm.eval()

        if self.classifier is not None:
            self.classifier.eval()

        device = self.device

        if initial_noise is None:
            x_t = torch.randn(
                shape,
                device=device,
                generator=generator,
            )
        else:
            x_t = initial_noise.clone().to(device=device)

        intermediates = []

        for timestep in reversed(range(self.ddpm.num_timesteps)):
            out = self.p_sample(
                x_t=x_t,
                timestep=timestep,
                class_labels=class_labels,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                generator=generator,
            )

            x_t = out["sample"]

            if return_intermediates:
                if (
                    timestep % intermediate_every == 0
                    or timestep == self.ddpm.num_timesteps - 1
                    or timestep == 0
                ):
                    intermediates.append(
                        {
                            "t": timestep,
                            "x_t": x_t.detach().cpu(),
                            "x_0_pred": out["x_0_pred"].detach().cpu(),
                        }
                    )

        if return_intermediates:
            return {
                "samples": x_t,
                "intermediates": intermediates,
            }

        return x_t