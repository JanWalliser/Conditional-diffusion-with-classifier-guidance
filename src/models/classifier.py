from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from src.models.unet_classifier import NoisyUNetClassifier


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1).float()

        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )

        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


def get_activation(name: str) -> type[nn.Module]:
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU

    return nn.SiLU


class TimeConditionedResidualBlock(nn.Module):
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

        act_layer = get_activation(activation)

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

        self.time_proj1 = nn.Linear(time_embedding_dim, out_channels)

        self.norm2 = make_group_norm(out_channels)
        self.act2 = act_layer()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.time_proj2 = nn.Linear(time_embedding_dim, out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = h + self.time_proj1(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = h + self.time_proj2(t_emb)[:, :, None, None]
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class NoisyImageClassifier(nn.Module):
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

        blocks_per_stage = max(1, blocks_per_stage)
        c = base_channels

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        self.stem = nn.Conv2d(3, c, kernel_size=3, padding=1, bias=False)

        self.stage1 = self._make_stage(c, c, time_embedding_dim, blocks_per_stage, 1, dropout, activation)
        self.stage2 = self._make_stage(c, 2 * c, time_embedding_dim, blocks_per_stage, 2, dropout, activation)
        self.stage3 = self._make_stage(2 * c, 4 * c, time_embedding_dim, blocks_per_stage, 2, dropout, activation)
        self.stage4 = self._make_stage(4 * c, 4 * c, time_embedding_dim, blocks_per_stage, 2, dropout, activation)
        self.stage5 = self._make_stage(4 * c, 4 * c, time_embedding_dim, blocks_per_stage, 2, dropout, activation)

        self.final_norm = make_group_norm(4 * c)
        self.final_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4 * c, num_classes)

        self._init_weights()

    def _make_stage(
        self,
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
                in_channels,
                out_channels,
                time_embedding_dim,
                stride=first_stride,
                dropout=dropout,
                activation=activation,
            )
        )

        for _ in range(blocks - 1):
            layers.append(
                TimeConditionedResidualBlock(
                    out_channels,
                    out_channels,
                    time_embedding_dim,
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

        return self.head(h)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def build_classifier_from_config(cfg: dict) -> nn.Module:
    model_cfg = cfg.get("model", cfg)
    name = str(model_cfg.get("name", "resnet")).lower()

    if name == "unet":
        return NoisyUNetClassifier(
            in_channels=model_cfg.get("in_channels", 3),
            num_classes=model_cfg.get("num_classes", 10),
            input_size=model_cfg.get("input_size", 32),
            base_channels=model_cfg.get("base_channels", 96),
            time_embedding_dim=model_cfg.get("time_embedding_dim", 384),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
            blocks_per_stage=model_cfg.get("blocks_per_stage", 2),
            dropout=model_cfg.get("dropout", 0.1),
            attention_resolutions=tuple(model_cfg.get("attention_resolutions", [8])),
            channels_per_head=model_cfg.get("channels_per_head", 64),
        )

    return NoisyImageClassifier(
        num_classes=model_cfg.get("num_classes", 10),
        base_channels=model_cfg.get("base_channels", 64),
        time_embedding_dim=model_cfg.get("time_embedding_dim", 128),
        dropout=model_cfg.get("dropout", 0.0),
        activation=model_cfg.get("activation", "gelu"),
        blocks_per_stage=model_cfg.get("blocks_per_stage", 2),
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
    t = torch.randint(0, 1000, (batch_size,), device=device)

    logits = model(x_t, t)

    print("Device:", device)
    print("x_t shape:", x_t.shape)
    print("t shape:", t.shape)
    print("logits shape:", logits.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
