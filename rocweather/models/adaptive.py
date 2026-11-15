"""自适应版的双路融合、退化条件和频带提示。"""

import torch
import torch.nn.functional as F
from torch import nn

CONDITION_DIM = 64


class BranchFusion(nn.Module):
    """每个通道用一对互补权重融合两条分支，初始权重各为 0.5。"""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(2 * channels), nn.Linear(2 * channels, channels))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, global_features, local_features):
        summary = torch.cat((global_features.mean(dim=1), local_features.mean(dim=1)), dim=-1)
        weight = self.gate(summary).sigmoid().unsqueeze(1)
        return local_features + weight * (global_features - local_features)


class FrequencyBands(nn.Module):
    """在当前像素网格上提取三张互补频带，三者相加可还原输入。"""

    def __init__(self):
        super().__init__()
        # 两个尺度用可分离高斯核，计算量随像素数线性增长。
        for name, radius, sigma in (("small", 2, 1.0), ("large", 6, 3.0)):
            coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
            weights = torch.exp(-0.5 * (coordinates / sigma).square())
            self.register_buffer(name, weights, persistent=False)

    @staticmethod
    def _blur(image, weights):
        channels = image.shape[1]
        radius = weights.numel() // 2
        weights = weights.to(image.dtype)
        weights = weights / weights.sum()
        horizontal = weights.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
        vertical = horizontal.transpose(-1, -2).contiguous()
        smooth = F.conv2d(
            F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels
        )
        return F.conv2d(
            F.pad(smooth, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
        )

    def forward(self, image):
        # 中高频是相近像素值的差，在混合精度下保留 FP32 计算。
        with torch.autocast(image.device.type, enabled=False):
            source = image.to(torch.promote_types(image.dtype, torch.float32))
            fine = self._blur(source, self.small)
            low = self._blur(source, self.large)
        return low, fine - low, source - fine

    @staticmethod
    def summed_statistics(bands):
        low, mid, high = bands
        # 低频保留有符号和，中高频取绝对值再求和，避免边缘的正负响应抵消。
        return torch.cat(
            (low.sum(dim=(-2, -1)), mid.abs().sum(dim=(-2, -1)), high.abs().sum(dim=(-2, -1))),
            dim=-1,
        )

    def tiled_statistics(self, image, tile_size):
        """按原像素尺度汇总全图频带，只保留一块的频带图。"""
        height, width = image.shape[-2:]
        radius = self.large.numel() // 2
        dtype = torch.promote_types(image.dtype, torch.float32)
        totals = image.new_zeros((image.shape[0], 3 * image.shape[1]), dtype=dtype)
        for y in range(0, height, tile_size):
            bottom = min(y + tile_size, height)
            top_with_border = max(0, y - radius)
            for x in range(0, width, tile_size):
                right = min(x + tile_size, width)
                left_with_border = max(0, x - radius)
                # 多取一圈邻域供滤波，只统计内区，得到与整图滤波一致的频带。
                window = image[
                    ..., top_with_border : bottom + radius, left_with_border : right + radius
                ]
                center = (
                    ...,
                    slice(y - top_with_border, bottom - top_with_border),
                    slice(x - left_with_border, right - left_with_border),
                )
                totals = totals + self.summed_statistics(band[center] for band in self(window))
        return totals / (height * width)


class DegradationCondition(nn.Module):
    """用浅层纹理、深层场景和频带统计生成连续的退化表示。"""

    def __init__(self, shallow_channels, deep_channels):
        super().__init__()
        self.shallow = nn.Sequential(
            nn.LayerNorm(shallow_channels), nn.Linear(shallow_channels, CONDITION_DIM)
        )
        self.deep = nn.Sequential(
            nn.LayerNorm(deep_channels), nn.Linear(deep_channels, CONDITION_DIM)
        )
        self.mix = nn.Sequential(
            nn.Linear(2 * CONDITION_DIM + 9, CONDITION_DIM),
            nn.GELU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
            nn.LayerNorm(CONDITION_DIM),
        )

    def forward(self, features, statistics, image_size):
        height, width = image_size

        def pool_valid(feature, stride):
            # 按原图覆盖的 token 池化，避免右下补齐区域主导小图统计。
            rows = (height + stride - 1) // stride
            cols = (width + stride - 1) // stride
            return feature[..., :rows, :cols].mean(dim=(-2, -1))

        shallow = self.shallow(pool_valid(features[0], 4))
        deep = self.deep(pool_valid(features[-1], 32))
        summary = torch.cat((shallow, deep, statistics), dim=-1)
        return self.mix(summary.to(self.mix[0].weight.dtype))


class ConditionedSkip(nn.Module):
    """天气条件决定通道权重，当前像素的编解码特征决定空间权重。"""

    def __init__(self, channels):
        super().__init__()
        self.channel_gate = nn.Linear(CONDITION_DIM, channels)
        self.spatial_gate = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        nn.init.normal_(self.channel_gate.weight, std=0.01)
        nn.init.ones_(self.channel_gate.bias)
        nn.init.zeros_(self.spatial_gate.weight)

    def forward(self, decoded, skip, condition):
        cues = torch.cat(
            (decoded.abs().mean(dim=1, keepdim=True), skip.abs().mean(dim=1, keepdim=True)), dim=1
        )
        logits = self.channel_gate(condition)[:, :, None, None] + self.spatial_gate(cues)
        return decoded + logits.sigmoid() * skip


class SpatialDetailGate(nn.Module):
    """从解码特征和高频图中提取两种尺度的线索，逐位置判断保留量。"""

    def __init__(self, channels):
        super().__init__()
        hidden = min(channels, 8)
        self.reduce = nn.Conv2d(channels + 3, hidden, 1)
        # 只在少量通道上计算 3×3、5×5 的深度卷积，控制原尺寸开销。
        self.near = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.wide = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.logits = nn.Conv2d(hidden, channels, 1)
        # 从接近原通道门控的行为起步，同时让两条局部支路都能收到梯度。
        nn.init.normal_(self.logits.weight, std=0.01)
        nn.init.zeros_(self.logits.bias)

    def forward(self, decoded, band):
        cues = self.reduce(torch.cat((decoded, band.to(decoded.dtype)), dim=1))
        context = F.gelu((self.near(cues) + self.wide(cues)) * 0.5)
        return self.logits(context)


class BandHint(nn.Module):
    """把原图的一个频带投影到解码特征，并按退化条件调节强度。"""

    def __init__(self, channels, *, spatial_gate=False):
        super().__init__()
        self.project = nn.Conv2d(3, channels, 1, bias=False)
        self.gate = nn.Linear(CONDITION_DIM, channels)
        self.spatial_gate = SpatialDetailGate(channels) if spatial_gate else None
        nn.init.normal_(self.project.weight, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, decoded, band, condition):
        if band.shape[-2:] != decoded.shape[-2:]:
            band = F.adaptive_avg_pool2d(band, decoded.shape[-2:])
        logits = self.gate(condition)[:, :, None, None]
        if self.spatial_gate is not None:
            logits = logits + self.spatial_gate(decoded, band)
        return decoded + self.project(band.to(self.project.weight.dtype)) * logits.sigmoid()
