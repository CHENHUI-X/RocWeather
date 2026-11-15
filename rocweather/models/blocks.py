"""编码和解码共用的积木：注意力、局部调制以及卷积上采样。"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth


def map_to_tokens(x):
    """把 (B, C, H, W) 的特征图变为 (B, H×W, C) 的序列。"""
    return x.flatten(2).transpose(1, 2)


def tokens_to_map(x, size):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], *size)


def fp16_autocast_active(x):
    if not x.is_cuda or not torch.is_autocast_enabled():
        return False
    dtype = (
        torch.get_autocast_dtype("cuda")
        if hasattr(torch, "get_autocast_dtype")
        else torch.get_autocast_gpu_dtype()
    )
    return dtype == torch.float16


def init_transformer(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchEmbedding(nn.Module):
    """用重叠卷积下采样，保留相邻区域之间的上下文。"""

    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, channels, kernel, stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 旧权重在深层下采样前的残差幅度较大，FP16 卷积输出可能在 LayerNorm 前就溢出。
        if self.proj.stride[0] == 2 and fp16_autocast_active(x):
            with torch.autocast("cuda", enabled=False):
                x = self.proj(x.float())
        else:
            x = self.proj(x)
        return self.norm(map_to_tokens(x)), x.shape[-2:]


class XPos(nn.Module):
    """保留原版编码器首层的一维 XPos（原代码著作权：Microsoft, 2022，MIT）。"""

    def __init__(self, channels):
        super().__init__()
        indices = torch.arange(0, channels, 2, dtype=torch.float32)
        self.register_buffer("scale", (indices + 0.4 * channels) / (1.4 * channels))
        self.register_buffer("frequency", 10000 ** (-indices / channels), persistent=False)

    def forward(self, x):
        length = x.shape[1]
        pos = torch.arange(length, device=x.device, dtype=torch.float32)
        angles = pos[:, None] * self.frequency
        # 原版按 4096 token（256×256 输入）设计；更大图像限制指数跨度，避免低精度溢出。
        exponent = (pos - (length + 1) // 2) / (max(length, 4096) / 8)
        scale = self.scale ** exponent[:, None]
        pairs = x.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        cos = (angles.cos() * scale).to(x.dtype)[..., None]
        sin = (angles.sin() * scale).to(x.dtype)[..., None]
        return (pairs * cos + rotated * sin).flatten(-2)


class Attention(nn.Module):
    """编码时用自注意力；解码时用可学习的空间查询读取图像特征。"""

    def __init__(self, channels, heads, dropout, attn_dropout, reduction=1, query_grid=None):
        super().__init__()
        self.heads = heads
        self.reduction = reduction
        self.query_grid = query_grid
        self.attn_dropout = attn_dropout
        # 沿用原来的参数名，便于读取结构一致的旧版权重。
        self.q = nn.Linear(channels, channels, bias=False)
        self.kv = nn.Linear(channels, channels * 2, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)
        if query_grid is not None:
            self.task_query = nn.Parameter(torch.randn(1, query_grid**2, channels))

    def forward(self, x, size, query_bias=None):
        batch, length, channels = x.shape
        head_dim = channels // self.heads
        query = x
        if self.query_grid is not None:
            query = self.task_query
            if size != (self.query_grid, self.query_grid):
                grid = tokens_to_map(query, (self.query_grid, self.query_grid))
                query = map_to_tokens(
                    F.interpolate(grid, size=size, mode="bilinear", align_corners=False)
                )
            if query_bias is not None:
                # 同一张图的空间查询共享退化偏移，每个位置仍保留自己的可学习查询。
                query = query + query_bias.unsqueeze(1)

        # 紧凑配置可减少浅层 K/V 的数量；基线 reduction=1，保持原来的全局注意力。
        source = x
        if self.reduction > 1:
            reduced_size = tuple(max(1, edge // self.reduction) for edge in size)
            source = map_to_tokens(F.adaptive_avg_pool2d(tokens_to_map(x, size), reduced_size))

        q = self.q(query).reshape(-1, length, self.heads, head_dim).transpose(1, 2)
        q = q.expand(batch, -1, -1, -1)
        k, v = self.kv(source).reshape(batch, -1, 2, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(batch, length, channels)))


class MLP(nn.Module):
    def __init__(self, channels, ratio, dropout):
        super().__init__()
        hidden = int(channels * ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class FocalModulation(nn.Module):
    """用不同大小的深度卷积与全局平均提取上下文，再按位置加权。"""

    def __init__(self, channels, levels, window, dropout):
        super().__init__()
        self.levels = levels
        self.f = nn.Linear(channels, 2 * channels + levels + 1)
        self.h = nn.Conv2d(channels, channels, 1)
        self.focal_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False
                ),
                nn.GELU(),
            )
            for kernel in range(window, window + 2 * levels, 2)
        )
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, size):
        channels = x.shape[-1]
        query, context, gates = tokens_to_map(self.f(x), size).split(
            (channels, channels, self.levels + 1), dim=1
        )
        combined = 0
        for index, layer in enumerate(self.focal_layers):
            context = layer(context)
            combined = combined + context * gates[:, index : index + 1]
        global_context = F.gelu(context.mean(dim=(-2, -1), keepdim=True))
        combined = combined + global_context * gates[:, -1:]
        return self.proj_drop(self.proj(map_to_tokens(query * self.h(combined))))


class TokenBlock(nn.Module):
    """两条编码分支共享归一化、前馈层和残差逻辑。"""

    def __init__(self, channels, options, drop_path, *, attention=None, modulation=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = attention
        self.modulation = modulation
        self.mlp = MLP(channels, options.mlp_ratio, options.dropout)
        self.drop_path = StochasticDepth(drop_path, "row")
        # 旧版只对注意力块使用 Transformer 初始化，Focal 分支沿用 PyTorch 默认值。
        if attention is not None:
            self.apply(init_transformer)

    def forward(self, x, size, query_bias=None):
        normalized = self.norm1(x)
        if self.attn is None:
            update = self.modulation(normalized, size)
        else:
            update = self.attn(normalized, size, query_bias=query_bias)
        x = x + self.drop_path(update)
        return x + self.drop_path(self.mlp(self.norm2(x)))
