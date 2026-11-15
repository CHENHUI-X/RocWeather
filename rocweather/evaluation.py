"""训练内验证与独立评估使用同一套指标与逐图汇总逻辑。"""

from math import isfinite
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import PairedDataset, plan_image_outputs, save_rgb
from .metrics import SSIM, psnr
from .restoration import restore


def validation_loader(config, runtime):
    dataset = PairedDataset(config.data, resize=config.eval.resize)
    # 验证集按步长分片，不补重复样本。每 rank 的数量允许不同。
    shard = Subset(dataset, range(runtime.rank, len(dataset), runtime.world_size))
    return DataLoader(
        shard,
        batch_size=1,
        num_workers=config.data.workers,
        pin_memory=runtime.device.type == "cuda",
    )


@torch.inference_mode()
def evaluate(model, loader, config, runtime, precision, output: Path | None = None):
    destinations = None
    if output is not None:
        dataset = loader.dataset.dataset
        source_directory = dataset.directory.resolve()
        if output.resolve().is_relative_to(source_directory):
            raise ValueError("评估输出目录须位于配对数据目录之外。")
        destinations = plan_image_outputs(
            output, (source.as_posix() for source, _ in dataset.pairs)
        )
        if any(path.is_relative_to(source_directory) for path in destinations.values()):
            raise ValueError("评估结果不能写入配对数据目录。")
        runtime.barrier()
    model.eval()
    ssim = SSIM().to(runtime.device)
    totals = torch.zeros(3, device=runtime.device, dtype=torch.float64)
    for inputs, target, names in loader:
        inputs = inputs.to(runtime.device, non_blocking=True)
        target = target.to(runtime.device, non_blocking=True)
        with precision.autocast():
            prediction = restore(model, inputs, config)
        totals[0] += psnr(prediction, target).sum()
        totals[1] += ssim(prediction, target).sum()
        totals[2] += target.shape[0]
        if destinations is not None:
            for image, name in zip(prediction, names):
                save_rgb(image, destinations[name])
    sum_psnr, sum_ssim, count = runtime.sum(totals).tolist()
    if not (isfinite(sum_psnr) and isfinite(sum_ssim)):
        raise ValueError("评估产生了非有限指标；请检查权重或尝试 --precision fp32。")
    return {"images": int(count), "psnr": sum_psnr / count, "ssim": sum_ssim / count}
