"""面向 RGB、值域 [0, 1] 的逐图 PSNR 和 SSIM。"""

import torch
import torch.nn.functional as F
from torch import nn


def psnr(prediction, target):
    mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
    # 完全一致记为 120 dB，保证日志和 checkpoint 中的数字可序列化。
    return -10 * torch.log10(mse.clamp_min(1e-12))


class SSIM(nn.Module):
    """使用可分离的高斯窗口计算局部亮度和结构相似度。"""

    def __init__(self):
        super().__init__()
        positions = torch.arange(-5, 6, dtype=torch.float32)
        self.register_buffer(
            "gaussian", (-positions.square() / (2 * 1.5**2)).exp(), persistent=False
        )

    def forward(self, prediction, target):
        x, y = prediction.float(), target.float()
        radius = min(self.gaussian.numel() // 2, (min(x.shape[-2:]) - 1) // 2)
        kernel = self.gaussian[5 - radius : 6 + radius]
        kernel = (kernel / kernel.sum()).view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1).contiguous()

        def blur(values):
            # 不补零，只比较有完整窗口的位置。
            values = F.conv2d(values, kernel, groups=x.shape[1])
            return F.conv2d(values, kernel.transpose(-2, -1).contiguous(), groups=x.shape[1])

        mean_x, mean_y, second_x, second_y, second_xy = blur(
            torch.cat((x, y, x.square(), y.square(), x * y), dim=0)
        ).chunk(5)
        var_x = (second_x - mean_x.square()).clamp_min(0)
        var_y = (second_y - mean_y.square()).clamp_min(0)
        covariance = second_xy - mean_x * mean_y
        luminance = (2 * mean_x * mean_y + 0.01**2) / (mean_x.square() + mean_y.square() + 0.01**2)
        structure = (2 * covariance + 0.03**2) / (var_x + var_y + 0.03**2)
        return (luminance * structure).mean(dim=(1, 2, 3))
