"""读取成对数据，对低质图和清晰图同步应用几何变换。"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .config import DataConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def open_rgb(path: Path):
    with Image.open(path) as image:
        return image.convert("RGB")


def resolve_image(path: Path):
    """现有清单写的是 PNG，部分文件实际为 JPG。只在文件不存在时尝试替代。"""
    if path.is_file():
        return path
    for suffix in (".png", ".jpg", ".jpeg"):
        alternative = path.with_suffix(suffix)
        if alternative != path and alternative.is_file():
            return alternative
    raise FileNotFoundError(f"未找到图像：{path}（也检查了 .png/.jpg/.jpeg）")


def image_tensor(image):
    return TF.pil_to_tensor(image).to(torch.float32).div_(255)


def model_input(image):
    """模型沿用 [-1, 1] 输入；清晰标签与指标仍使用 [0, 1]。"""
    return image.mul(2).sub_(1)


def save_rgb(image, path: Path):
    image = image.detach().float().cpu()
    if not torch.isfinite(image).all():
        raise ValueError(f"图像含非有限值，未写入 {path}；可尝试使用 --precision fp32。")
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.clamp(0, 1).mul_(255).round_().to(torch.uint8)
    with path.open("xb") as file:
        TF.to_pil_image(pixels).save(file, format="PNG")


def plan_image_outputs(output, names):
    """先检查整批目标路径，避免同名图片或已有文件被覆盖。"""
    root = output.resolve()
    paths, seen = {}, set()
    for name in names:
        path = Path(name)
        if path.suffix != ".png":
            path = path.with_suffix(path.suffix + ".png")
        destination = (root / path).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"图片输出路径越过指定目录：{destination}")
        if destination in seen or destination.exists():
            raise ValueError(f"图片输出有重名或文件已存在：{destination}")
        paths[name] = destination
        seen.add(destination)
    return paths


class PairedDataset(Dataset):
    """每行可写 input/文件名，或用 Tab 显式分隔输入路径与标签路径。"""

    def __init__(self, config: DataConfig, *, training=False, resize=0, seed=0):
        manifest = Path(config.root) / (config.train_manifest if training else config.val_manifest)
        self.directory = manifest.parent
        self.training = training
        self.crop_size = config.crop_size
        self.horizontal_flip = config.horizontal_flip
        self.rotate_90 = config.rotate_90
        self.small_image_mode = config.small_image_mode
        self.resize = resize
        self.seed = seed
        self.pairs = []
        with manifest.open(encoding="utf-8-sig") as file:
            for index, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths = tuple(Path(part.strip().replace("\\", "/")) for part in line.split("\t"))
                if len(paths) not in (1, 2) or any(
                    not p.parts or p.anchor or ".." in p.parts for p in paths
                ):
                    raise ValueError(f"{manifest}:{index}：请输入一到两条相对图像路径。")
                if len(paths) == 1:
                    if paths[0].parts[0] != "input" or len(paths[0].parts) < 2:
                        raise ValueError(f"{manifest}:{index}：单列清单须以 input/ 开头。")
                    paths += (Path("gt", *paths[0].parts[1:]),)
                self.pairs.append(paths)
        if not self.pairs:
            raise ValueError(f"清单没有图像：{manifest}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # 训练采样器会同时传入 epoch；裁剪由样本和轮次决定，与 worker 是否重启无关。
        epoch = 0
        if isinstance(index, tuple):
            index, epoch = index
        source, target = self.pairs[index]
        degraded = open_rgb(resolve_image(self.directory / source))
        clean = open_rgb(resolve_image(self.directory / target))

        if self.training:
            if degraded.size != clean.size:
                raise ValueError(f"{source} 与 {target} 的尺寸不同：{degraded.size} / {clean.size}")
            rng = random.Random((self.seed << 64) + (epoch << 32) + index)
            degraded, clean = self._training_patch(degraded, clean, rng)
        elif self.resize:
            degraded, clean = self._resize_pair(degraded, clean, (self.resize, self.resize))
        elif degraded.size != clean.size:
            raise ValueError(f"{source} 与 {target} 的原图尺寸不同：{degraded.size} / {clean.size}")

        return model_input(image_tensor(degraded)), image_tensor(clean), source.as_posix()

    def _training_patch(self, degraded, clean, rng):
        width, height = degraded.size
        crop = self.crop_size
        if min(width, height) < crop:
            new_width, new_height = max(width, crop), max(height, crop)
            if self.small_image_mode == "edge_pad":
                # 小图复制边缘补足裁剪尺寸，保留原始雨纹、雪点的像素宽度。
                padding = [0, 0, new_width - width, new_height - height]
                degraded = TF.pad(degraded, padding, padding_mode="edge")
                clean = TF.pad(clean, padding, padding_mode="edge")
            else:
                degraded, clean = self._resize_pair(degraded, clean, (new_width, new_height))
            width, height = new_width, new_height

        x, y = rng.randint(0, width - crop), rng.randint(0, height - crop)
        box = (x, y, x + crop, y + crop)
        degraded, clean = degraded.crop(box), clean.crop(box)
        if self.horizontal_flip and rng.random() < 0.5:
            degraded, clean = TF.hflip(degraded), TF.hflip(clean)
        if self.rotate_90:
            # 整数次直角旋转只重排像素，不插值，也不改变方形裁剪的尺寸。
            turn = rng.randrange(4)
            if turn:
                operation = (
                    Image.Transpose.ROTATE_90,
                    Image.Transpose.ROTATE_180,
                    Image.Transpose.ROTATE_270,
                )[turn - 1]
                degraded, clean = degraded.transpose(operation), clean.transpose(operation)
        return degraded, clean

    @staticmethod
    def _resize_pair(degraded, clean, size):
        return tuple(image.resize(size, Image.Resampling.LANCZOS) for image in (degraded, clean))
