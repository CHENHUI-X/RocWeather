"""写入 TensorBoard 的运行配置、每步损失和每轮汇总。"""

from contextlib import contextmanager

import torch
import yaml


def require_tensorboard():
    """启动时先检查依赖，让失败的命令可以使用同一目录重试。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ValueError(
            "TensorBoard 未安装，请运行 python -m pip install -e '.[tensorboard]'。"
        ) from exc
    return SummaryWriter


def remember_tensorboard_gap(output, step):
    """关闭看板续训时，保留需要清除旧事件的最早训练步。"""
    directory = output / "tensorboard"
    if directory.is_dir():
        marker = directory / ".purge-step"
        first = min(step + 1, int(marker.read_text())) if marker.exists() else step + 1
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(str(first), encoding="ascii")
        temporary.replace(marker)


@contextmanager
def open_tensorboard(
    writer_type, output, config, model, optimizer, runtime, precision, dataset_sizes, step
):
    """在当前实验目录记录事件，续训时从最近保存的训练步接上。"""
    directory = output / "tensorboard"
    marker = directory / ".purge-step"
    purge_step = step + 1 if step else None
    if marker.exists():
        earlier = int(marker.read_text())
        purge_step = min(purge_step, earlier) if purge_step is not None else earlier
    with writer_type(
        log_dir=str(directory),
        purge_step=purge_step,
        max_queue=min(max(64, config.train.log_every), 512),
        flush_secs=15,
    ) as writer:
        resolved = {
            "config": config.as_dict(),
            "optimizer": {
                "name": type(optimizer).__name__,
                "param_groups": [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.param_groups
                ],
            },
            "actual": {
                "device": str(runtime.device),
                "gpu": torch.cuda.get_device_name(runtime.device)
                if runtime.device.type == "cuda"
                else None,
                "precision": precision.mode,
                "world_size": runtime.world_size,
                "global_batch_size": config.train.batch_size * runtime.world_size,
                "torch_version": str(torch.__version__),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "training_samples": dataset_sizes[0],
                "validation_samples": dataset_sizes[1],
            },
        }
        text = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
        writer.add_text("run/resolved_configuration", f"```yaml\n{text}```", step)
        writer.flush()
        marker.unlink(missing_ok=True)
        yield writer


def _write_scalars(writer, values, step):
    from tensorboard.compat.proto.summary_pb2 import Summary

    # 同一步汇成一个事件，减少写入器入队次数。
    summary = Summary(value=[Summary.Value(tag=tag, simple_value=value) for tag, value in values])
    writer.file_writer.add_summary(summary, step)


def write_steps(writer, names, samples, first_step, lr):
    # 将整组 GPU 标量一次取回，避免每项损失每个训练步都同步显卡。
    for step, values in enumerate(torch.stack(samples).tolist(), first_step):
        # DDP 时每步记录主进程的观测，每轮另有全局汇总。
        scalars = [("step/learning_rate", lr)]
        scalars.extend((f"step_rank0/{name}", value) for name, value in zip(names, values))
        _write_scalars(writer, scalars, step)


def write_epoch(writer, record):
    step = record["global_step"]
    scalars = [("epoch/learning_rate", record["lr"]), ("epoch/number", record["epoch"])]
    for field in ("train", "validation", "performance"):
        scalars.extend(
            (f"epoch/{field}/{name}", value) for name, value in record.get(field, {}).items()
        )
    _write_scalars(writer, scalars, step)
    writer.flush()
