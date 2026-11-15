"""四级双路编码 → 任务查询解码 → 卷积恢复。"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from ..config import ModelConfig
from .blocks import (
    Attention,
    FocalModulation,
    PatchEmbedding,
    TokenBlock,
    XPos,
    fp16_autocast_active,
    map_to_tokens,
    tokens_to_map,
)


class WeatherEncoder(nn.Module):
    def __init__(self, config, drop_rates):
        super().__init__()
        self.pe = XPos(config.channels[0])
        self.path_embed = nn.ModuleList()
        self.RPLFA = nn.ModuleList()
        self.HFDC = nn.ModuleList()
        in_channels = 3
        offset = 0
        for stage, (channels, depth) in enumerate(zip(config.channels, config.depths)):
            self.path_embed.append(
                PatchEmbedding(in_channels, channels, config.patch_kernel, 4 if stage == 0 else 2)
            )
            in_channels = channels
            rates = drop_rates[offset : offset + depth]
            self.RPLFA.append(
                nn.ModuleList(
                    TokenBlock(
                        channels,
                        config,
                        rate,
                        attention=Attention(
                            channels,
                            config.heads[stage],
                            config.dropout,
                            config.attention_dropout,
                            reduction=config.kv_reductions[stage],
                        ),
                    )
                    for rate in rates
                )
            )
            self.HFDC.append(
                nn.ModuleList(
                    TokenBlock(
                        channels,
                        config,
                        rate,
                        modulation=FocalModulation(
                            channels,
                            config.focal_levels,
                            config.focal_window,
                            config.dropout,
                        ),
                    )
                    for rate in rates
                )
            )
            offset += depth

    def forward(self, x):
        features = []
        for stage, (embed, global_blocks, local_blocks) in enumerate(
            zip(self.path_embed, self.RPLFA, self.HFDC)
        ):
            encoded, size = embed(x)
            if stage == 0:
                encoded = encoded + self.pe(encoded)
            global_features = local_features = encoded
            for global_block, local_block in zip(global_blocks, local_blocks):
                # 保留旧版：每一支的块内有残差，编码器在块外还会再加一次。
                global_features = global_features + global_block(global_features, size)
                local_features = local_features + local_block(local_features, size)
            x = tokens_to_map(global_features + local_features, size)
            features.append(x)
        return features


class CentralDecoder(nn.Module):
    def __init__(self, config, drop_rates):
        super().__init__()
        channels = config.channels[-1]
        self.block = nn.ModuleList(
            TokenBlock(
                channels,
                config,
                rate,
                attention=Attention(
                    channels,
                    config.heads[-1],
                    config.dropout,
                    config.attention_dropout,
                    query_grid=config.query_grid,
                ),
            )
            for rate in drop_rates
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        size = x.shape[-2:]
        original = tokens = map_to_tokens(x)
        for block in self.block:
            tokens = block(tokens, size)
        return tokens_to_map(original + self.norm(tokens), size)


class ConvLayer(nn.Module):
    """保留旧卷积的子层命名，以便加载旧权重。"""

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups)

    def forward(self, x):
        return self.conv2d(x)


def channel_norm(x, layer):
    """LayerNorm 在每个像素的通道维度上计算。"""
    return layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=(-2, -1)))[:, :, None, None]


class ResidualConv(nn.Module):
    def __init__(self, channels, drop_path):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.SE = ChannelGate(channels)
        self.drop_path = StochasticDepth(drop_path, "row")

    def forward(self, x):
        return x + self.drop_path(self.SE(channel_norm(self.dwconv(x), self.norm)))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Tconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv1 = ConvLayer(out_channels, out_channels, groups=out_channels)
        self.LayerNorm1 = nn.LayerNorm(out_channels)
        self.LayerNorm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = channel_norm(self.Tconv2d(x), self.LayerNorm1)
        return F.relu(channel_norm(self.conv1(x), self.LayerNorm2))


class ReconstructionHead(nn.Module):
    # 旧权重以这些名字保存六个尺度的卷积；运算顺序集中在下方循环。
    UPSAMPLE_NAMES = ("convd32x", "convd16x", "convd8x", "convd4x", "convd2x")

    def __init__(self, config):
        super().__init__()
        widths = (*reversed(config.channels), *config.tail_channels)
        for index, width in enumerate(widths):
            self.add_module(
                f"dense_{6 - index}", nn.Sequential(ResidualConv(width, config.tail_drop_path))
            )
        for name, source, target in zip(self.UPSAMPLE_NAMES, widths[:-1], widths[1:]):
            self.add_module(name, Upsample(source, target))
        self.conv_output = ConvLayer(widths[-1], 3)

    def forward(self, features, x):
        skips = tuple(reversed(features))
        for index in range(6):
            x = getattr(self, f"dense_{6 - index}")(x)
            if index < len(skips):
                x = x + skips[index]
            if index < len(self.UPSAMPLE_NAMES):
                x = getattr(self, self.UPSAMPLE_NAMES[index])(x)
        return self.conv_output(x).tanh()


class RocWeather(nn.Module):
    """接收 [-1, 1] 的 RGB 张量，返回与输入同尺寸的恢复图。"""

    def __init__(self, config=None):
        super().__init__()
        config = config or ModelConfig()
        encoder_rates = torch.linspace(0, config.drop_path, sum(config.depths)).tolist()
        decoder_rates = torch.linspace(
            0, config.drop_path, sum(config.depths[:-1]) + config.decoder_depth
        ).tolist()[-config.decoder_depth :]
        self.Tenc = WeatherEncoder(config, encoder_rates)
        self.Tdec = CentralDecoder(config, decoder_rates)
        self.convtail = ReconstructionHead(config)

    def forward(self, x):
        height, width = x.shape[-2:]
        # 五次上采样要求对齐到 32；补足右/下边缘并在输出中裁掉。
        if height % 32 or width % 32:
            x = F.pad(x, (0, -width % 32, 0, -height % 32), mode="replicate")
        features = self.Tenc(x)
        decoded = self.Tdec(features[-1])
        if fp16_autocast_active(x):
            # 旧权重的跳连相加后可超出 FP16 动态范围，卷积重建保留 FP32。
            with torch.autocast("cuda", enabled=False):
                restored = self.convtail([feature.float() for feature in features], decoded.float())
        else:
            restored = self.convtail(features, decoded)
        return restored[..., :height, :width]
