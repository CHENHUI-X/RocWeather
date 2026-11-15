"""四级双路编码 → 任务查询解码 → 卷积恢复。"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from ..config import ModelConfig
from .adaptive import (
    CONDITION_DIM,
    BandHint,
    BranchFusion,
    ConditionedSkip,
    DegradationCondition,
    FrequencyBands,
)
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


def pad_image(image):
    """编码器总步长为 32，只在右侧和下侧补齐。"""
    height, width = image.shape[-2:]
    if height % 32 or width % 32:
        padding = (0, -width % 32, 0, -height % 32)
        if image.dtype in (torch.float16, torch.bfloat16):
            # 部分 PyTorch 版本没有低精度复制补边内核，转到 FP32 做等值复制。
            return F.pad(image.float(), padding, mode="replicate").to(image.dtype)
        return F.pad(image, padding, mode="replicate")
    return image


class WeatherEncoder(nn.Module):
    def __init__(self, config, drop_rates):
        super().__init__()
        first_width = config.channels[0]
        self.fusion = None
        if config.variant == "adaptive":
            self.fusion = nn.ModuleList(BranchFusion(width) for width in config.channels)
            # 以二维邻域补充位置信息，避免将不同行的首尾当成同一行邻居。
            self.pe = nn.Conv2d(
                first_width, first_width, 3, padding=1, groups=first_width, bias=False
            )
            nn.init.normal_(self.pe.weight, std=0.02)
        else:
            self.pe = XPos(first_width)
        self.path_embed = nn.ModuleList()
        self.RPLFA = nn.ModuleList()
        self.HFDC = nn.ModuleList()
        in_channels = 3
        offset = 0
        kernels = config.patch_kernels or (config.patch_kernel,) * len(config.channels)
        for stage, (channels, depth) in enumerate(zip(config.channels, config.depths)):
            self.path_embed.append(
                PatchEmbedding(in_channels, channels, kernels[stage], 4 if stage == 0 else 2)
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
                position = (
                    self.pe(encoded)
                    if self.fusion is None
                    else map_to_tokens(self.pe(tokens_to_map(encoded, size)))
                )
                encoded = encoded + position
            global_features = local_features = encoded
            for global_block, local_block in zip(global_blocks, local_blocks):
                global_output = global_block(global_features, size)
                local_output = local_block(local_features, size)
                if self.fusion is None:
                    # 旧断点依赖这套累加方式，单独保留。
                    global_output = global_features + global_output
                    local_output = local_features + local_output
                global_features, local_features = global_output, local_output
            fused = (
                global_features + local_features
                if self.fusion is None
                else self.fusion[stage](global_features, local_features)
            )
            x = tokens_to_map(fused, size)
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
        self.query_condition = None
        if config.variant == "adaptive":
            self.query_condition = nn.Linear(CONDITION_DIM, channels, bias=False)
            nn.init.zeros_(self.query_condition.weight)

    def forward(self, x, condition=None):
        size = x.shape[-2:]
        original = tokens = map_to_tokens(x)
        query_bias = self.query_condition(condition) if self.query_condition is not None else None
        for block in self.block:
            tokens = block(tokens, size, query_bias=query_bias)
        tokens = self.norm(tokens)
        # 自适应版的块内残差已经传递瓶颈特征，解码出口直接使用归一化结果。
        if self.query_condition is None:
            tokens = original + tokens
        return tokens_to_map(tokens, size)


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
    # 六级依次为 1/32、1/16、1/8、1/4、1/2、原分辨率。频带编号从低到高。
    BAND_AT_LEVEL = (0, None, 1, 1, None, 2)

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
        self.skip_gates = None
        if config.variant == "adaptive":
            self.skip_gates = nn.ModuleList(ConditionedSkip(width) for width in widths[1:4])
            self.band_hints = nn.ModuleDict(
                {
                    str(level): BandHint(
                        widths[level], spatial_gate=config.spatial_detail_gate and band_index == 2
                    )
                    for level, band_index in enumerate(self.BAND_AT_LEVEL)
                    if band_index is not None
                }
            )
            # 从接近输入图的小幅校正起步，同时给上游模块留下梯度。
            nn.init.normal_(self.conv_output.conv2d.weight, std=0.001)
            nn.init.zeros_(self.conv_output.conv2d.bias)

    def forward(self, features, x, *, condition=None, bands=None):
        skips = tuple(reversed(features))
        for index, band_index in enumerate(self.BAND_AT_LEVEL):
            if self.skip_gates is not None and band_index is not None:
                x = self.band_hints[str(index)](x, bands[band_index], condition)
            x = getattr(self, f"dense_{6 - index}")(x)
            if index < len(skips):
                if self.skip_gates is None:
                    x = x + skips[index]
                elif index > 0:
                    # 瓶颈已送入中心解码器，重建头只接入后续三个尺度的跳连。
                    x = self.skip_gates[index - 1](x, skips[index], condition)
            if index < len(self.UPSAMPLE_NAMES):
                x = getattr(self, self.UPSAMPLE_NAMES[index])(x)
        output = self.conv_output(x)
        return output.tanh() if self.skip_gates is None else output


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
        self.conditioner = None
        if config.variant == "adaptive":
            self.conditioner = DegradationCondition(config.channels[0], config.channels[-1])
            self.frequency_bands = FrequencyBands()

    def condition_for_tiles(self, image, tile_size):
        """用保留长宽比的整图缩略图估计条件，供所有局部图块共用。"""
        if self.conditioner is None:
            return None
        statistics = self.frequency_bands.tiled_statistics(image, tile_size)
        size = image.shape[-2:]
        scale = tile_size / max(size)
        if scale < 1:
            size = tuple(max(1, round(edge * scale)) for edge in size)
            image = F.interpolate(image, size=size, mode="area")
        image = pad_image(image)
        return self.conditioner(self.Tenc(image), statistics, size)

    def forward(self, x, *, condition=None):
        height, width = x.shape[-2:]
        x = pad_image(x)
        features = self.Tenc(x)
        bands = None
        if self.conditioner is not None:
            bands = self.frequency_bands(x)
            if condition is None:
                valid_bands = (band[..., :height, :width] for band in bands)
                statistics = self.frequency_bands.summed_statistics(valid_bands) / (height * width)
                condition = self.conditioner(features, statistics, (height, width))
        decoded = self.Tdec(features[-1], condition)
        fp16 = fp16_autocast_active(x)
        if fp16:
            # 沿用 FP32 重建路径，防止大幅度的跳连特征在卷积中溢出。
            features = [feature.float() for feature in features]
            decoded = decoded.float()
            if condition is not None:
                condition = condition.float()
        with torch.autocast("cuda", enabled=False) if fp16 else nullcontext():
            restored = self.convtail(features, decoded, condition=condition, bands=bands)
        if self.conditioner is not None:
            # 数据加载器的标签是 [0, 1]，以同域的输入为底图加上残差。
            restored = (x + 1) * 0.5 + restored
        return restored[..., :height, :width]
