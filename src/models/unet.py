from __future__ import annotations

import math
from typing import Optional, Sequence

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
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )

    args = timesteps.float()[:, None] * freqs[None, :]
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


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(valid_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, 2 * out_channels),
        )

        self.norm2 = nn.GroupNorm(valid_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.time_embedding_projection(time_embedding)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_connection(x)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.num_heads = make_num_heads(channels, channels_per_head)
        self.head_dim = channels // self.num_heads

        self.norm = nn.GroupNorm(valid_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w)

        qkv = self.qkv(x)
        query, key, value = torch.chunk(qkv, chunks=3, dim=1)

        query = query.view(b, self.num_heads, self.head_dim, h * w)
        key = key.view(b, self.num_heads, self.head_dim, h * w)
        value = value.view(b, self.num_heads, self.head_dim, h * w)

        scale = 1.0 / math.sqrt(self.head_dim)

        attention = torch.einsum("bhct,bhcs->bhts", query * scale, key)
        attention = torch.softmax(attention, dim=-1)

        out = torch.einsum("bhts,bhcs->bhct", attention, value)
        out = out.reshape(b, c, h * w)

        out = self.projection(out)
        out = out.view(b, c, h, w)

        return residual + out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
        use_attention: bool = False,
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.resnet_block = ResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )

        if use_attention:
            self.attention = SelfAttentionBlock(
                channels=out_channels,
                channels_per_head=channels_per_head,
            )
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = self.resnet_block(x, time_embedding)
        x = self.attention(x)
        return x


class UpLevelBlock(nn.Module):
    def __init__(
        self,
        blocks: Sequence[ResidualAttentionBlock],
        upsample: Optional[Upsample],
    ):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.upsample = upsample

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_embedding)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class DenoisingUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        input_size: int = 32,
        base_channels: int = 96,
        time_embedding_dim: Optional[int] = None,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
        attention_resolutions: Sequence[int] = (16, 8),
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim or base_channels * 4
        self.channel_mults = tuple(channel_mults)
        self.blocks_per_stage = blocks_per_stage
        self.attention_resolutions = tuple(attention_resolutions)
        self.channels_per_head = channels_per_head

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downsample_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_channels: list[int] = [base_channels]
        current_channels = base_channels
        current_resolution = input_size

        for stage_idx, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()

            for _ in range(self.blocks_per_stage):
                use_attention = current_resolution in self.attention_resolutions

                block = ResidualAttentionBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    time_embedding_dim=self.time_embedding_dim,
                    dropout=dropout,
                    use_attention=use_attention,
                    channels_per_head=channels_per_head,
                )

                stage_blocks.append(block)
                current_channels = out_ch
                skip_channels.append(current_channels)

            self.downsample_blocks.append(stage_blocks)

            is_last_stage = stage_idx == len(self.channel_mults) - 1
            if not is_last_stage:
                self.downsamples.append(Downsample(current_channels))
                skip_channels.append(current_channels)
                current_resolution //= 2

        self.middle = nn.ModuleList(
            [
                ResnetBlock(current_channels, current_channels, self.time_embedding_dim, dropout),
                SelfAttentionBlock(current_channels, channels_per_head),
                ResnetBlock(current_channels, current_channels, self.time_embedding_dim, dropout),
            ]
        )

        self.up_levels = nn.ModuleList()
        decoder_skip_channels = list(skip_channels)

        for stage_idx, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            use_attention = current_resolution in self.attention_resolutions
            blocks: list[ResidualAttentionBlock] = []

            for _ in range(self.blocks_per_stage + 1):
                skip_ch = decoder_skip_channels.pop()

                block = ResidualAttentionBlock(
                    in_channels=current_channels + skip_ch,
                    out_channels=out_ch,
                    time_embedding_dim=self.time_embedding_dim,
                    dropout=dropout,
                    use_attention=use_attention,
                    channels_per_head=channels_per_head,
                )

                blocks.append(block)
                current_channels = out_ch

            if stage_idx != 0:
                upsample = Upsample(current_channels)
                current_resolution *= 2
            else:
                upsample = None

            self.up_levels.append(UpLevelBlock(blocks=blocks, upsample=upsample))

        self.out_norm = nn.GroupNorm(valid_group_count(current_channels), current_channels)
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.view(-1)

        time_embedding = timestep_embedding(timesteps, self.base_channels)
        time_embedding = self.time_embedding_mlp(time_embedding)

        h = self.input_conv(x)
        skips: list[torch.Tensor] = [h]

        for stage_idx, stage_blocks in enumerate(self.downsample_blocks):
            for block in stage_blocks:
                h = block(h, time_embedding)
                skips.append(h)

            is_last_stage = stage_idx == len(self.channel_mults) - 1
            if not is_last_stage:
                h = self.downsamples[stage_idx](h)
                skips.append(h)

        h = self.middle[0](h, time_embedding)
        h = self.middle[1](h)
        h = self.middle[2](h, time_embedding)

        for up_level in self.up_levels:
            h = up_level(h, skips, time_embedding)

        h = F.silu(self.out_norm(h))
        out = self.out_conv(h)

        return out


def build_unet_from_config(cfg: dict) -> DenoisingUNet:
    return DenoisingUNet(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        input_size=cfg.get("input_size", 32),
        base_channels=cfg.get("base_channels", 96),
        time_embedding_dim=cfg.get("time_embedding_dim", None),
        channel_mults=tuple(cfg.get("channel_mults", (1, 2, 2, 4))),
        blocks_per_stage=cfg.get("blocks_per_stage", 2),
        dropout=cfg.get("dropout", 0.1),
        attention_resolutions=tuple(cfg.get("attention_resolutions", (16, 8))),
        channels_per_head=cfg.get("channels_per_head", 64),
    )


if __name__ == "__main__":
    model = DenoisingUNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))

    y = model(x, t)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
    print("Parameters:  ", sum(p.numel() for p in model.parameters()))
