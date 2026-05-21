from __future__ import annotations




from typing import List, Optional, Sequence
import math
import torch.nn.functional as Functional
import torch
import torch.nn as nn


def timestep_embedding(
        timesteps: torch.tensor,
        dim: int,
        max_period: int = 10_000,
) -> torch.tensor:
    
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1) #Error with data not 1D

        half = dim//2
        frequency = torch.exp(
            -math.log(max_period) * torch.arange(start= 0, end = half,dtype=torch.float32,device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * frequency[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If dim is odd, pad one extra zero column so output shape is exactly [B, dim].
    if dim % 2 == 1:
        embedding = Functional.pad(embedding, (0, 1))

    return embedding




#Group Normalisation and herlper function

def valid_group_count(channels: int, max_groups: int) -> int:

    for groups in range(1, max_groups + 1):
        if channels % groups == 0:
            return groups
        


#Attention heads and helper function
def make_num_heads(channels: int, channels_per_head: int = 64) -> int:


    heads = max(1,channels // channels_per_head)
    while channels % heads != 0:
        heads -= 1

    return max(1, heads)


##################################################
#Spartial resolution blocks
##################################################
class Downsample(nn.Module):
    def __init__(self, channels: int,):
        super().__init__()

        self.op = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.op(x)
    

class Upsample(nn.Module):
    def __init__(self, channels: int,):
        super().__init__()

        self.convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = Functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.convolution(x)
    


class ResnetBlock(nn.Module):
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

        self.norm1 = nn.GroupNorm(num_groups=valid_group_count(self.in_channels), num_channels=self.in_channels)
            
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.time_embedding_projection = nn.Linear(time_embedding_dim, self.out_channels*2)

        self.norm2 = nn.GroupNorm(num_groups=valid_group_count(self.out_channels), num_channels=self.out_channels)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        if self.in_channels != self.out_channels:
            self.residual_connection = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.residual_connection = nn.Identity()

    
    def forward(self, x: torch.tensor, time_embedding: torch.tensor) -> torch.tensor:
        h = self.conv1(Functional.silu(self.norm1(x)))
        scale_shift = self.time_embedding_projection(Functional.silu(time_embedding))[:, :, None, None]
        scale,shift = torch.chunk(scale_shift, 2, dim=1)

        #Reshaping to [B, C, 1, 1] so we can apply it to the feature map
        scale = scale [:,:, None, None]
        shift = shift [:,:, None, None]

        h = self.norm2(h) * (scale + 1) + shift
        h = Functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_connection(x)
    
    #Spartial selfattention block

class SelfAttentionBlock(nn.Module): #only low dimensions cuz its cheaper :)
    def __init__(
        self,
        channels: int,
        channels_per_head: int = 64,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = make_num_heads(channels, channels_per_head)
        self.head_dim = channels // self.num_heads

        # Normalise before attention.
        self.norm = nn.GroupNorm(valid_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        b,c,h,w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w)

        qkv = self.qkv(x)
        query,key,value = torch.chunk(qkv, 3, dim=1)

        query = query.view(b, self.num_heads, self.head_dim, h * w)
        key = key.view(b, self.num_heads, self.head_dim, h * w)
        value = value.view(b, self.num_heads, self.head_dim, h * w)

        scale = 1 / math.sqrt(self.head_dim)
        attention = torch.einsum("bhct,bhcs->bhts", query * scale, key)
        attention = torch.softmax(attention, dim=-1)

        out = out.reshape(b, c, h * w)
        out = self.projection(out)
        out = out.view(b, c, h, w)


class ResisdualAttentionBlock(nn.Module):
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

        self.resnet_block = ResnetBlock(in_channels, out_channels, time_embedding_dim, dropout)
        if use_attention:
            self.attention = SelfAttentionBlock(
                channels=out_channels,
                channels_per_head=channels_per_head,
            )
        else:
            self.attention = nn.Identity()
    def forward(self, x: torch.tensor, time_embedding: torch.tensor) -> torch.tensor:
        x = self.resnet_block(x, time_embedding)
        x = self.attention(x)
        return x
    



class UpLevelBlock(nn.Module):
    def __init__(
       self,
       blocks: Sequence[ResisdualAttentionBlock],
       upsample: Optional[Upsample] 
    ):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.upsample = upsample

    def forward(self, x: torch.tensor, skips: list[torch.tensor], time_embedding: torch.tensor) -> torch.tensor:
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


        # Time embedding MLP
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(self.base_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        skip_channels: list[int] = [base_channels]
        current_resolution = input_size


#####################################################
        #Encoder
####################################################
        for stage_idx, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()

            for _  in range(self.blocks_per_stage):
                use_attention = current_resolution in self.attention_resolutions
                block = ResisdualAttentionBlock(
                    in_channels=skip_channels[-1],
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


            

            ##Middle blocks
        self.middle = nn.ModuleList(
                    [
                        ResnetBlock(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            time_embedding_dim=self.time_embedding_dim,
                            dropout=dropout,
                        ),
                        SelfAttentionBlock(
                            channels=current_channels,
                            channels_per_head=channels_per_head,
                        ),
                        ResnetBlock(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            time_embedding_dim=self.time_embedding_dim,
                            dropout=dropout,
                        ),
                    ]
        )



################################################
        #Decoder
        ####################

        self.up_levels = nn.ModuleList()
        decoder_skip_channels = list(skip_channels)

        for stage_idx, mult in reversed(list(enumerate(self.channel_mults))):

            out_ch = base_channels * mult
            use_attention = current_resolution in self.attention_resolutions
            blocks: list[ResisdualAttentionBlock] = []

            for _ in range(blocks_per_stage + 1):

                skip_ch = decoder_skip_channels.pop()
                block = ResisdualAttentionBlock(
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

            self.up_levels.append(
                UpLevelBlock(
                    blocks=blocks,
                    upsample=upsample,
                )
            )

        self.out_norm = nn.GroupNorm(
            valid_group_count(current_channels),
            current_channels,
        )

        self.out_conv = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self._init_weights()




    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias) 


    def forward(self, x: torch.tensor, timesteps: torch.tensor) -> torch.tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.view(-1)

        self.time_embedding_dim = timestep_embedding(timesteps, self.base_channels)
        self.time_embedding_dim = self.time_embedding_mlp(self.time_embedding_dim)

        h = self.input_conv(x)
        skips: list[torch.tensor] = [h]

        for stage_idx, stage_blocks in enumerate(self.downsample_blocks):
            for block in stage_blocks:
                h = block(h, self.time_embedding_dim)
                skips.append(h)

            is_last_stage = stage_idx == len(self.channel_mults) - 1
            if not is_last_stage:
                h = self.downsamples[stage_idx](h)
                skips.append(h)

        h = self.middle[0](h, self.time_embedding_dim)
        h = self.middle[1](h)
        h = self.middle[2](h, self.time_embedding_dim)


        for up_level in self.up_levels:
            h = up_level(h, skips, self.time_embedding_dim)

        h = Functional.silu(self.out_norm(h))
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



##########################################Minimal smoke test



if __name__ == "__main__":
    model = DenoisingUNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = model(x, t)
    print("Input shapes: ", x.shape)
    print("Output shapes:", y.shape)
    print("Parameters:  ", sum(p.numel() for p in model.parameters()))
