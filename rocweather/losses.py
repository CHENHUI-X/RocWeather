"""把各项损失分开计算，YAML 中的权重设为 0 即跳过该项。"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from .metrics import SSIM


class PerceptualLoss(nn.Module):
    """比较 VGG16 的 relu1_2、relu2_2、relu3_3 特征；只更新恢复模型。"""

    def __init__(self):
        super().__init__()
        self.features = (
            vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().requires_grad_(False)
        )
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def extract(self, image):
        image = (image - self.mean) / self.std
        result = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in (3, 8, 15):
                result.append(image)
        return result

    def forward(self, prediction, target):
        # 两次最大池化至少需要 4×4。
        if min(target.shape[-2:]) < 4:
            size = tuple(max(4, edge) for edge in target.shape[-2:])
            prediction = F.interpolate(prediction, size=size, mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            reference = self.extract(target)
        return sum(F.mse_loss(a, b) for a, b in zip(self.extract(prediction), reference)) / len(
            reference
        )


class RestorationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ssim = SSIM() if config.ssim_weight else None
        self.perceptual = PerceptualLoss() if config.perceptual_weight else None

    def charbonnier(self, prediction, target):
        return ((prediction - target).square() + self.config.charbonnier_epsilon).sqrt().mean()

    def forward(self, prediction, target, identity=None):
        # 低精度只用于模型前向；损失在 FP32 中计算，避免 SSIM 的方差消减误差。
        prediction = prediction.float()
        parts = {"charbonnier": self.charbonnier(prediction, target)}
        total = parts["charbonnier"]
        if self.ssim is not None:
            parts["ssim"] = 1 - self.ssim(prediction.clamp(0, 1), target).mean()
            total = total + self.config.ssim_weight * parts["ssim"]
        if self.perceptual is not None:
            parts["perceptual"] = self.perceptual(prediction.clamp(0, 1), target)
            total = total + self.config.perceptual_weight * parts["perceptual"]
        if identity is not None:
            parts["identity"] = self.charbonnier(identity.float(), target)
            total = total + self.config.identity_weight * parts["identity"]
        return total, parts
