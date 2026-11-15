"""将任意尺寸的输入恢复为同尺寸图像，并在需要时分块控制显存。"""

import math

import torch

from .config import EvalConfig


def _origins(length, tile, overlap):
    if length <= tile:
        return [0]
    last = length - tile
    steps = math.ceil(last / (tile - overlap))
    return [round(i * last / steps) for i in range(steps + 1)]


def _blend_axis(start, index, origins, length, tile, device):
    end = min(start + tile, length)
    weights = torch.ones(end - start, device=device, dtype=torch.float32)
    # 使用实际重叠宽度渐变，处理最后一块未对齐固定步长的情况。
    if index > 0 and (left := origins[index - 1] + tile - start) > 0:
        weights[:left] *= torch.linspace(0, 1, left + 2, device=device)[1:-1]
    if index + 1 < len(origins) and (right := end - origins[index + 1]) > 0:
        weights[-right:] *= torch.linspace(1, 0, right + 2, device=device)[1:-1]
    return weights


def restore(model, image, config: EvalConfig):
    height, width = image.shape[-2:]
    tile = config.tile_size
    if not tile or max(height, width) <= tile:
        return model(image).float().clamp_(0, 1)

    # 自适应模型从整图估计一次退化条件。每块仍计算自己的细节特征。
    condition_for_tiles = getattr(model, "condition_for_tiles", None)
    condition = condition_for_tiles(image, tile) if condition_for_tiles is not None else None
    model_kwargs = {"condition": condition} if condition is not None else {}
    rows = _origins(height, tile, config.tile_overlap)
    cols = _origins(width, tile, config.tile_overlap)
    image_sum = torch.zeros_like(image, dtype=torch.float32)
    weight_sum = torch.zeros((height, width), device=image.device, dtype=torch.float32)
    x_weights = [_blend_axis(x, i, cols, width, tile, image.device) for i, x in enumerate(cols)]
    for i, y in enumerate(rows):
        y_weights = _blend_axis(y, i, rows, height, tile, image.device)
        for x, x_weight in zip(cols, x_weights):
            region = (..., slice(y, y + tile), slice(x, x + tile))
            weight = y_weights[:, None] * x_weight
            prediction = model(image[region], **model_kwargs).float()
            image_sum[region] += prediction * weight
            weight_sum[region] += weight
    return image_sum.div_(weight_sum).clamp_(0, 1)
