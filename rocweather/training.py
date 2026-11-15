"""统一的训练循环：同一入口支持 CPU、单卡 CUDA 和 DDP。"""

import json
import logging
from contextlib import ExitStack
from dataclasses import replace
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoints import FORMAT_VERSION, is_training_checkpoint, load_weights, save_checkpoint
from .config import config_from_dict
from .data import PairedDataset, model_input
from .evaluation import evaluate, validation_loader
from .losses import RestorationLoss
from .models import RocWeather
from .monitor import (
    open_tensorboard,
    remember_tensorboard_gap,
    require_tensorboard,
    write_epoch,
    write_steps,
)
from .runtime import Precision, seed_everything, setup_runtime

LOG = logging.getLogger(__name__)


class EpochSampler(DistributedSampler):
    """分配各 rank 的训练样本，并把当前轮次传给数据增强。"""

    def __iter__(self):
        return ((index, self.epoch) for index in super().__iter__())


def _check_resume(checkpoint, config):
    if not is_training_checkpoint(checkpoint):
        raise ValueError("该文件不是新版训练断点。请用 --weights 仅加载权重。")
    previous = config_from_dict(checkpoint["config"])
    train = replace(
        config.train,
        epochs=previous.train.epochs,
        log_every=previous.train.log_every,
        validate_every=previous.train.validate_every,
        tensorboard=previous.train.tensorboard,
    )
    data = replace(config.data, root=previous.data.root, workers=previous.data.workers)
    if (config.model, config.loss, train, data, config.eval) != (
        previous.model,
        previous.loss,
        previous.train,
        previous.data,
        previous.eval,
    ):
        raise ValueError("续训须沿用模型、损失、优化、数据和评估设置；更改这些设置请用 --weights。")
    if config.train.epochs <= checkpoint["epoch"]:
        raise ValueError(f"断点已完成 {checkpoint['epoch']} 轮；train.epochs 必须大于已完成轮数。")


def _prepare_output(output, resume_path, config):
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "last.pt"
    if any(
        (output / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl", "config.yaml")
    ):
        if resume_path is None or resume_path.resolve() != latest.resolve():
            raise ValueError("输出目录已有训练记录。请换一个目录，或从该目录的 last.pt 续训。")
    with (output / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.as_dict(), file, sort_keys=False, allow_unicode=True)


def _train_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    precision,
    runtime,
    config,
    epoch,
    *,
    writer=None,
    global_step=0,
):
    model.train()
    names = ["loss", "charbonnier"] + [
        name
        for name in ("ssim", "perceptual", "identity")
        if getattr(config.loss, f"{name}_weight")
    ]
    totals = torch.zeros(len(names) + 1, device=runtime.device, dtype=torch.float64)
    pending = []
    step_names = [*names, "grad_norm"] if config.train.grad_clip else names
    lr = optimizer.param_groups[0]["lr"]
    for step, (images, target, _) in enumerate(loader, 1):
        images = images.to(runtime.device, non_blocking=True)
        target = target.to(runtime.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        identity = None
        with precision.autocast():
            if config.loss.identity_weight:
                # 将同一批清晰图也送入模型，约束它保留清晰图；DDP 每步只前向一次。
                prediction, identity = model(torch.cat((images, model_input(target)))).chunk(2)
            else:
                prediction = model(images)
        loss, parts = loss_fn(prediction, target, identity)
        scaler.scale(loss).backward()
        grad_norm = None
        if config.train.grad_clip:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # 训练中只累加张量；每轮汇总一次，避免每个 batch 跨卡同步。
        with torch.no_grad():
            values = torch.stack([loss, *(parts[name] for name in names[1:])])
            totals[:-1] += values * images.shape[0]
            totals[-1] += images.shape[0]
            if writer is not None:
                pending.append(
                    torch.cat((values, grad_norm.reshape(1))) if grad_norm is not None else values
                )
        if runtime.primary and (step % config.train.log_every == 0 or step == len(loader)):
            if writer is not None:
                write_steps(writer, step_names, pending, global_step + step - len(pending) + 1, lr)
                pending.clear()
            LOG.info(
                "epoch %d | batch %d/%d | 本进程累计 loss %.5f",
                epoch,
                step,
                len(loader),
                (totals[0] / totals[-1]).item(),
            )

    totals = runtime.sum(totals).tolist()
    if not isfinite(totals[0]):
        raise ValueError(f"第 {epoch} 轮的损失出现非有限值；请检查数据、学习率或训练精度。")
    return dict(zip(names, (value / totals[-1] for value in totals[:-1])))


def train(config, output: Path, *, checkpoint=None, resume_path=None):
    if resume_path is not None:
        _check_resume(checkpoint, config)

    with setup_runtime(config.runtime) as runtime, ExitStack() as logs:
        writer_type = (
            require_tensorboard() if runtime.primary and config.train.tensorboard else None
        )
        precision = Precision.for_runtime(config.runtime.precision, runtime)
        seed_everything(config.train.seed, runtime.device)
        dataset = PairedDataset(config.data, training=True, seed=config.train.seed)
        sampler = EpochSampler(
            dataset, num_replicas=runtime.world_size, rank=runtime.rank, seed=config.train.seed
        )
        loader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.data.workers,
            pin_memory=runtime.device.type == "cuda",
            persistent_workers=config.data.workers > 0,
            generator=torch.Generator().manual_seed(config.train.seed),
        )
        val_loader = validation_loader(config, runtime)

        base_model = RocWeather(config.model)
        if checkpoint is not None:
            load_weights(base_model, checkpoint, allow_detail_upgrade=resume_path is None)
        # 卷积的存储布局与逐像素 LayerNorm 对齐，也避免 DDP 桶重复调整深度卷积的梯度布局。
        base_model.to(device=runtime.device, memory_format=torch.channels_last)
        model = runtime.wrap_model(base_model)
        optimizer = AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scheduler = ExponentialLR(optimizer, gamma=config.train.lr_decay)
        scaler = precision.scaler()
        previous_epoch, global_step, best_psnr = 0, 0, None
        if resume_path is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint["precision"] == precision.mode:
                scaler.load_state_dict(checkpoint["scaler"])
            previous_epoch, global_step, best_psnr = (
                checkpoint["epoch"],
                checkpoint["global_step"],
                checkpoint["best_psnr"],
            )
            # 新目录没有旧 best.pt，用本次续训的结果重新选择最佳权重。
            if (
                resume_path.resolve() != (output / "last.pt").resolve()
                or not (output / "best.pt").is_file()
            ):
                best_psnr = None
        # 释放调用者传入的 CPU 断点，训练只保留正在使用的模型和优化器。
        if checkpoint is not None:
            checkpoint.clear()

        writer = None
        if runtime.primary:
            _prepare_output(output, resume_path, config)
            if writer_type is not None:
                writer = logs.enter_context(
                    open_tensorboard(
                        writer_type,
                        output,
                        config,
                        base_model,
                        optimizer,
                        runtime,
                        precision,
                        (len(dataset), len(val_loader.dataset.dataset)),
                        global_step,
                    )
                )
            elif resume_path is not None:
                remember_tensorboard_gap(output, global_step)
            LOG.info(
                "设备 %s | 精度 %s | 进程 %d (%s) | 全局批大小上限 %d | 训练 %d / 验证 %d 张",
                runtime.device,
                precision.mode,
                runtime.world_size,
                runtime.backend or "single",
                config.train.batch_size * runtime.world_size,
                len(dataset),
                len(val_loader.dataset.dataset),
            )
        if runtime.distributed and config.loss.perceptual_weight:
            # 主进程先把 VGG 权重写到共享缓存，同机各 rank 避免重复下载。
            loss_fn = RestorationLoss(config.loss).to(runtime.device) if runtime.primary else None
            runtime.barrier()
            if loss_fn is None:
                loss_fn = RestorationLoss(config.loss).to(runtime.device)
        else:
            loss_fn = RestorationLoss(config.loss).to(runtime.device)

        for epoch in range(previous_epoch + 1, config.train.epochs + 1):
            sampler.set_epoch(epoch)
            # 以 epoch/rank 派生随机种子，恢复到同一轮时可复现采样和增强顺序。
            epoch_seed = config.train.seed + epoch * runtime.world_size + runtime.rank
            seed_everything(epoch_seed, runtime.device)
            lr = optimizer.param_groups[0]["lr"]
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            started = perf_counter()
            training = _train_epoch(
                model,
                loader,
                loss_fn,
                optimizer,
                scaler,
                precision,
                runtime,
                config,
                epoch,
                writer=writer,
                global_step=global_step,
            )
            train_seconds = perf_counter() - started
            scheduler.step()
            global_step += len(loader)

            performance = {
                "train_seconds": train_seconds,
                "train_images_per_second": len(sampler) * runtime.world_size / train_seconds,
            }
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": lr,
                "train": training,
                "performance": performance,
            }
            improved = False
            if epoch % config.train.validate_every == 0 or epoch == config.train.epochs:
                # 直接调用底层模型；不同 rank 的验证批数不必一致。
                started = perf_counter()
                validation = evaluate(base_model, val_loader, config.eval, runtime, precision)
                performance["validation_seconds"] = perf_counter() - started
                record["validation"] = validation
                if best_psnr is None or validation["psnr"] > best_psnr:
                    best_psnr = validation["psnr"]
                    improved = True

            if runtime.primary:
                if runtime.device.type == "cuda":
                    performance["rank0_peak_allocated_mib"] = (
                        torch.cuda.max_memory_allocated(runtime.device) / 2**20
                    )
                    performance["rank0_peak_reserved_mib"] = (
                        torch.cuda.max_memory_reserved(runtime.device) / 2**20
                    )
                if "validation" in record:
                    LOG.info(
                        "验证 %d 张 | PSNR %.3f dB | SSIM %.5f",
                        validation["images"],
                        validation["psnr"],
                        validation["ssim"],
                    )
                state = {
                    "format_version": FORMAT_VERSION,
                    "config": config.as_dict(),
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_psnr": best_psnr,
                    "precision": precision.mode,
                }
                if improved:
                    save_checkpoint(output / "best.pt", state)
                save_checkpoint(output / "last.pt", state)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                if writer is not None:
                    write_epoch(writer, record)
