from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from src.models.unet_classifier import NoisyUNetClassifier


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding.

    Input:
        t: Tensor of shape [B]

    Output:
        embedding: Tensor of shape [B, embedding_dim]
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")

        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

        half_dim = self.embedding_dim // 2
        t = t.float()

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )

        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return embedding


def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Create GroupNorm with a valid number of groups.
    """
    groups = min(max_groups, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class TimeConditionedResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    The timestep embedding is projected to the number of output channels and
    added after the first convolution.

    Input:
        x:     [B, in_channels, H, W]
        t_emb: [B, time_embedding_dim]

    Output:
        h:     [B, out_channels, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "gelu"] = "silu",
    ) -> None:
        super().__init__()

        if activation == "silu":
            act_layer = nn.SiLU
        elif activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.norm1 = make_group_norm(in_channels)
        self.act1 = act_layer()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.time_proj2 = nn.Linear(time_embedding_dim, out_channels)

        self.norm2 = make_group_norm(out_channels)
        self.act2 = act_layer()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_bias = self.time_proj(t_emb)
        h = h + time_bias[:, :, None, None]

        h = self.norm2(h)
        h = h + self.time_proj2(t_emb)[:, :, None, None]
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class NoisyImageClassifier(nn.Module):
    """
    Time-conditioned ResNet-style classifier for classifier-guided diffusion.

    This classifier predicts the original CIFAR-10 class label y from a noisy
    image x_t and timestep t.

    Input:
        x_t: [B, 3, 32, 32]
        t:   [B]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "gelu"] = "silu",
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()

        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1.")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        c = base_channels

        self.stem = nn.Conv2d(
            in_channels=3,
            out_channels=c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.stage1 = self._make_stage(
            in_channels=c,
            out_channels=c,
            time_embedding_dim=time_embedding_dim,
            blocks=blocks_per_stage,
            first_stride=1,
            dropout=dropout,
            activation=activation,
        )

        self.stage2 = self._make_stage(
            in_channels=c,
            out_channels=2 * c,
            time_embedding_dim=time_embedding_dim,
            blocks=blocks_per_stage,
            first_stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.stage3 = self._make_stage(
            in_channels=2 * c,
            out_channels=4 * c,
            time_embedding_dim=time_embedding_dim,
            blocks=blocks_per_stage,
            first_stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.stage4 = self._make_stage(
            in_channels=4 * c,
            out_channels=4 * c,
            time_embedding_dim=time_embedding_dim,
            blocks=blocks_per_stage,
            first_stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.stage5 = self._make_stage(
            in_channels=4 * c,
            out_channels=4 * c,
            time_embedding_dim=time_embedding_dim,
            blocks=blocks_per_stage,
            first_stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.final_norm = make_group_norm(4 * c)
        self.final_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(4 * c, num_classes)

        self._init_weights()

    def _make_stage(
        self,
        *,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        blocks: int,
        first_stride: int,
        dropout: float,
        activation: Literal["silu", "relu", "gelu"],
    ) -> nn.ModuleList:
        layers = nn.ModuleList()

        layers.append(
            TimeConditionedResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_embedding_dim=time_embedding_dim,
                stride=first_stride,
                dropout=dropout,
                activation=activation,
            )
        )

        for _ in range(blocks - 1):
            layers.append(
                TimeConditionedResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                    stride=1,
                    dropout=dropout,
                    activation=activation,
                )
            )

        return layers

    def _forward_stage(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        stage: nn.ModuleList,
    ) -> torch.Tensor:
        for block in stage:
            x = block(x, t_emb)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"Expected x_t with shape [B, C, H, W], got {tuple(x_t.shape)}")

        if x_t.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x_t.shape[1]}")

        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

        if x_t.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_t batch is {x_t.shape[0]}, "
                f"t batch is {t.shape[0]}"
            )

        t_emb = self.time_embedding(t)

        h = self.stem(x_t)

        h = self._forward_stage(h, t_emb, self.stage1)
        h = self._forward_stage(h, t_emb, self.stage2)
        h = self._forward_stage(h, t_emb, self.stage3)
        h = self._forward_stage(h, t_emb, self.stage4)
        h = self._forward_stage(h, t_emb, self.stage5)

        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.pool(h)
        h = torch.flatten(h, start_dim=1)

        logits = self.head(h)

        return logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def build_classifier_from_config(cfg: dict) -> NoisyImageClassifier:
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "resnet").lower()

    if name in {"resnet"}:
        return NoisyImageClassifier(
            num_classes=cfg.get("num_classes", 10),
            base_channels=cfg.get("base_channels", 64),
            time_embedding_dim=cfg.get("time_embedding_dim", 128),
            dropout=cfg.get("dropout", 0.0),
            activation=cfg.get("activation", "gelu"),
            blocks_per_stage=cfg.get("blocks_per_stage", 2),
        )

    if name in {"unet"}:
        return NoisyUNetClassifier(
            in_channels=cfg.get("in_channels", 3),
            num_classes=cfg.get("num_classes", 10),
            input_size=cfg.get("input_size", 32),
            base_channels=cfg.get("base_channels", 96),
            time_embedding_dim=cfg.get("time_embedding_dim", 384),
            channel_mults=tuple(cfg.get("channel_mults", [1, 2, 4])),
            blocks_per_stage=cfg.get("blocks_per_stage", 2),
            dropout=cfg.get("dropout", 0.1),
            attention_resolutions=tuple(cfg.get("attention_resolutions", [8])),
            channels_per_head=cfg.get("channels_per_head", 64),
        )
     



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoisyImageClassifier(
        num_classes=10,
        base_channels=64,
        time_embedding_dim=128,
        dropout=0.0,
        activation="gelu",
        blocks_per_stage=2,
    ).to(device)

    batch_size = 8
    x_t = torch.randn(batch_size, 3, 32, 32, device=device)
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=device)

    logits = model(x_t, t)

    print("Device:", device)
    print("x_t shape:", x_t.shape)
    print("t shape:", t.shape)
    print("logits shape:", logits.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))