"""集中管理训练参数；在启动时检查拼写和数值，避免静默使用默认值。"""

from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, ...] = (64, 128, 256, 512)
    depths: tuple[int, ...] = (2, 2, 6, 2)
    heads: tuple[int, ...] = (1, 2, 4, 8)
    kv_reductions: tuple[int, ...] = (1, 1, 1, 1)
    tail_channels: tuple[int, ...] = (32, 16)
    patch_kernel: int = 7
    focal_levels: int = 2
    focal_window: int = 3
    decoder_depth: int = 2
    query_grid: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path: float = 0.1
    tail_drop_path: float = 0.1
    variant: str = "legacy"
    patch_kernels: tuple[int, ...] = ()

    def __post_init__(self):
        if self.variant not in {"legacy", "adaptive"}:
            raise ValueError("model.variant 必须为 legacy 或 adaptive。")
        for name in ("channels", "depths", "heads", "kv_reductions", "tail_channels"):
            values = getattr(self, name)
            if len(values) != (2 if name == "tail_channels" else 4) or min(values) < 1:
                raise ValueError(f"model.{name} 的长度不正确，或存在小于 1 的数。")
        if self.channels[0] % 2 or any(c % h for c, h in zip(self.channels, self.heads)):
            raise ValueError("首层通道数须为偶数，且各层通道数须能整除各自头数。")
        if self.patch_kernels and len(self.patch_kernels) != 4:
            raise ValueError("model.patch_kernels 须为四个阶段各指定一个卷积核。")
        if any(
            k < 1 or k % 2 == 0 for k in (self.patch_kernel, *self.patch_kernels, self.focal_window)
        ):
            raise ValueError("下采样卷积核和 focal_window 必须为正奇数。")
        if (
            min(self.focal_levels, self.decoder_depth, self.query_grid) < 1
            or not isfinite(self.mlp_ratio)
            or self.mlp_ratio * min(self.channels) < 1
        ):
            raise ValueError("卷积层级、解码深度、查询网格和 MLP 隐藏层宽度必须大于零。")
        if any(
            not 0 <= p < 1
            for p in (self.dropout, self.attention_dropout, self.drop_path, self.tail_drop_path)
        ):
            raise ValueError("模型的 dropout 和 drop_path 概率必须在 [0, 1) 内。")


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    train_manifest: str = "train/train.txt"
    val_manifest: str = "test/test.txt"
    crop_size: int = 256
    horizontal_flip: bool = False
    workers: int = 2

    def __post_init__(self):
        if self.crop_size < 1 or self.workers < 0:
            raise ValueError("data.crop_size 必须为正，data.workers 不得为负。")


@dataclass(frozen=True)
class LossConfig:
    charbonnier_epsilon: float = 0.001
    ssim_weight: float = 0.05
    perceptual_weight: float = 0.04
    identity_weight: float = 0.05

    def __post_init__(self):
        if not isfinite(self.charbonnier_epsilon) or self.charbonnier_epsilon <= 0:
            raise ValueError("loss.charbonnier_epsilon 必须为正。")
        if any(
            not isfinite(w) or w < 0
            for w in (self.ssim_weight, self.perceptual_weight, self.identity_weight)
        ):
            raise ValueError("损失权重必须为有限的非负数。")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_decay: float = 0.99
    grad_clip: float = 1.0
    seed: int = 666
    log_every: int = 50
    validate_every: int = 1

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.log_every, self.validate_every) < 1:
            raise ValueError("epoch、batch size、日志和验证间隔必须为正整数。")
        if not 0 <= self.seed < 2**32:
            raise ValueError("train.seed 必须在 [0, 2³²) 内。")
        if any(not isfinite(v) for v in (self.learning_rate, self.weight_decay, self.grad_clip)):
            raise ValueError("优化器参数必须为有限值。")
        if (
            self.learning_rate <= 0
            or min(self.weight_decay, self.grad_clip) < 0
            or not 0 < self.lr_decay <= 1
        ):
            raise ValueError(
                "学习率必须为正，权重衰减和梯度裁剪不得为负，lr_decay 必须在 (0, 1] 内。"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    precision: str = "auto"
    backend: str = "auto"

    def __post_init__(self):
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("runtime.precision 必须为 auto、fp32、fp16 或 bf16。")
        if self.backend not in {"auto", "gloo", "nccl"}:
            raise ValueError("runtime.backend 必须为 auto、gloo 或 nccl。")


@dataclass(frozen=True)
class EvalConfig:
    resize: int = 256
    tile_size: int = 256
    tile_overlap: int = 32

    def __post_init__(self):
        if min(self.resize, self.tile_size, self.tile_overlap) < 0 or (
            self.tile_size and self.tile_overlap >= self.tile_size
        ):
            raise ValueError("eval 的尺寸不得为负；启用分块时，重叠须小于块大小。")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def as_dict(self):
        return asdict(self)


def _section(section_type, values, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} 必须是键值映射。")
    defaults = asdict(section_type())
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"{name} 包含未知参数：{', '.join(map(str, unknown))}")
    normalized = {}
    for key, value in values.items():
        default = defaults[key]
        expected = type(default)
        if (
            expected is tuple
            and isinstance(value, (tuple, list))
            and all(type(v) is int for v in value)
        ):
            normalized[key] = tuple(value)
        elif type(value) is expected or expected is float and type(value) is int:
            normalized[key] = value
        else:
            raise ValueError(f"{name}.{key} 的类型不正确，预期 {expected.__name__}。")
    return section_type(**normalized)


def config_from_dict(values):
    if not isinstance(values, dict):
        raise ValueError("配置根节点必须是键值映射。")
    defaults = Config()
    sections = {item.name: type(getattr(defaults, item.name)) for item in fields(defaults)}
    unknown = values.keys() - sections.keys()
    if unknown:
        raise ValueError(f"未知配置段：{', '.join(map(str, unknown))}")
    return Config(
        **{
            name: _section(sections[name], section_values, name)
            for name, section_values in values.items()
        }
    )


def load_config(path: Path):
    with path.open(encoding="utf-8-sig") as file:
        values = yaml.safe_load(file)
    return config_from_dict({} if values is None else values)
