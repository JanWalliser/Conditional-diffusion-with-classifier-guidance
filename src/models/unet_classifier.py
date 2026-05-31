from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )

    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


def valid_group_count(channels: int, max_groups: int = 32) -> int:


    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_num_heads(channels: int, channels_per_head: int = 64) -> int:

    heads = max(1, channels // channels_per_head)

    while channels % heads != 0:
        heads -= 1

    return max(1, heads)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(valid_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, 2 * out_channels),
        )

        self.norm2 = nn.GroupNorm(valid_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.time_proj(t_emb)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = make_num_heads(channels, channels_per_head)
        self.head_dim = channels // self.num_heads

        self.norm = nn.GroupNorm(valid_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhct,bhcs->bhts", q * scale, k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhts,bhcs->bhct", attn, v)
        out = out.reshape(b, c, h * w)

        out = self.proj(out)
        out = out.view(b, c, h, w)

        return residual + out


class AttentionPool2d(nn.Module):

    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim

        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim * spatial_dim + 1, embed_dim) / math.sqrt(embed_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        cls_token = x.mean(dim=1, keepdim=True)      # [B, 1, C]
        x = torch.cat([cls_token, x], dim=1)         # [B, 1 + HW, C]

        x = x + self.positional_embedding[None, :, :].to(dtype=x.dtype)

        pooled, _ = self.attn(
            query=x[:, :1, :],
            key=x,
            value=x,
            need_weights=False,
        )

        pooled = pooled.squeeze(1)
        pooled = self.norm(pooled)

        return pooled


class NoisyUNetClassifier(nn.Module):###Downsapmling from Diffusion UNet classifier with time embedding conditioning.

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
        base_channels: int = 96,
        time_embedding_dim: int = 384,
        channel_mults: Sequence[int] = (1, 2, 4),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
        attention_resolutions: Sequence[int] = (8,),
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim
        self.channel_mults = tuple(channel_mults)
        self.blocks_per_stage = blocks_per_stage
        self.attention_resolutions = tuple(attention_resolutions)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        modules: list[nn.Module] = []

        current_channels = base_channels
        current_resolution = input_size

        for stage_idx, mult in enumerate(self.channel_mults):
            out_channels = base_channels * mult

            for _ in range(blocks_per_stage):
                modules.append(
                    ResBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        time_embedding_dim=time_embedding_dim,
                        dropout=dropout,
                    )
                )

                current_channels = out_channels

                if current_resolution in self.attention_resolutions:
                    modules.append(
                        AttentionBlock(
                            channels=current_channels,
                            channels_per_head=channels_per_head,
                        )
                    )

            is_last_stage = stage_idx == len(self.channel_mults) - 1

            if not is_last_stage:
                modules.append(Downsample(current_channels))
                current_resolution //= 2

        self.down = nn.ModuleList(modules)

        self.out_norm = nn.GroupNorm(
            valid_group_count(current_channels),
            current_channels,
        )

        pool_heads = make_num_heads(current_channels, channels_per_head)

        self.pool = AttentionPool2d(
            spatial_dim=current_resolution,
            embed_dim=current_channels,
            num_heads=pool_heads,
        )

        self.head = nn.Linear(current_channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)

        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)

        for module in self.down:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)

        h = F.silu(self.out_norm(h))
        h = self.pool(h)

        logits = self.head(h)
        return logits