from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")

        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (batch_size,) containing timesteps.

        Returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape (batch_size,), got {tuple(t.shape)}")

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


class TimeConditionedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu"] = "silu",
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

        self.time_proj = nn.Linear(time_embedding_dim, out_channels)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, height, width)
            t_emb: Tensor of shape (batch_size, time_embedding_dim)

        Returns:
            Tensor of shape (batch_size, out_channels, height, width)
        """
        h = self.conv(x)
        h = self.norm(h)

        time_out = self.time_proj(t_emb)
        time_out = time_out[:, :, None, None]

        h = h + time_out
        h = self.activation(h)
        h = self.dropout(h)

        return h


class NoisyImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "silu"] = "silu",
    ) -> None:
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.conv1 = TimeConditionedConvBlock(
            in_channels=3,
            out_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            stride=1,
            dropout=dropout,
            activation=activation,
        )

        self.conv2 = TimeConditionedConvBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            time_embedding_dim=time_embedding_dim,
            stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.conv3 = TimeConditionedConvBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            time_embedding_dim=time_embedding_dim,
            stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.conv4 = TimeConditionedConvBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            time_embedding_dim=time_embedding_dim,
            stride=2,
            dropout=dropout,
            activation=activation,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 3, height, width)
            t: Tensor of shape (batch_size,)

        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}")

        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")

        if t.ndim != 1:
            raise ValueError(f"Expected t with shape (B,), got {tuple(t.shape)}")

        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has batch {x.shape[0]}, t has batch {t.shape[0]}"
            )

        t_emb = self.time_embedding(t)

        x = self.conv1(x, t_emb)
        x = self.conv2(x, t_emb)
        x = self.conv3(x, t_emb)
        x = self.conv4(x, t_emb)

        x = self.pool(x)
        logits = self.head(x)

        return logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def build_classifier_from_config(cfg: dict) -> NoisyImageClassifier:
    model_cfg = cfg.get("model", {})

    return NoisyImageClassifier(
        num_classes=int(model_cfg.get("num_classes", 10)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        time_embedding_dim=int(model_cfg.get("time_embedding_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        activation=str(model_cfg.get("activation", "silu")),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoisyImageClassifier(
        num_classes=10,
        base_channels=64,
        time_embedding_dim=128,
        dropout=0.1,
        activation="silu",
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