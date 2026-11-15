"""单进程与 torchrun 共用的设备选择、分布式和混合精度设置。"""

import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def get_device(requested, local_rank=None):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("设备必须是 auto、cpu、cuda 或 cuda:<编号>。")
    if device.type == "cuda":
        if local_rank is not None and device.index is not None:
            raise ValueError("DDP 请用 --device cuda；torchrun 会为每个进程分配 GPU。")
        index = local_rank if local_rank is not None else (device.index or 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f"没有可用的 CUDA 设备 cuda:{index}。")
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    backend: str | None

    @property
    def primary(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    def wrap_model(self, model):
        if not self.distributed:
            return model
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        # 模型没有 BatchNorm 等运行期变化的缓冲区；评估时各 rank 可以独立前向。
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    def sum(self, values):
        if self.distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

    def barrier(self):
        if self.distributed:
            dist.barrier(device_ids=[self.device.index] if self.backend == "nccl" else None)


@contextmanager
def setup_runtime(config):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    device = get_device(config.device, local_rank)
    backend = None
    if world_size > 1:
        if not dist.is_available() or local_rank is None:
            raise RuntimeError("请使用带 distributed 支持的 PyTorch，并用 torchrun 启动 DDP。")
        backend = config.backend
        if backend == "auto":
            backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
        if backend == "nccl" and (device.type != "cuda" or not dist.is_nccl_available()):
            raise ValueError("NCCL 需要 CUDA 和带 NCCL 的 PyTorch；可改用 gloo。")
        if backend == "gloo" and not dist.is_gloo_available():
            raise ValueError("当前 PyTorch 未提供 Gloo 后端。")
        dist.init_process_group(backend=backend)
    try:
        yield Runtime(device, rank, world_size, backend)
    finally:
        if backend is not None:
            dist.destroy_process_group()


def seed_everything(seed, device):
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


class Precision:
    @classmethod
    def for_runtime(cls, mode, runtime):
        if runtime.distributed and runtime.device.type == "cuda" and mode in {"auto", "bf16"}:
            # 混用 BF16 / FP16 会在 DDP 梯度归约时混入有缩放与无缩放的梯度。
            support = torch.tensor(int(torch.cuda.is_bf16_supported()), device=runtime.device)
            all_support = runtime.sum(support).item() == runtime.world_size
            if mode == "bf16" and not all_support:
                raise ValueError("有 CUDA rank 不支持 BF16；请为所有进程使用 fp16 或 fp32。")
            mode = "bf16" if all_support else "fp16"
        return cls(mode, runtime.device)

    def __init__(self, mode, device):
        if mode == "auto":
            mode = (
                ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
                if device.type == "cuda"
                else "fp32"
            )
        if mode == "fp16" and device.type != "cuda":
            raise ValueError("FP16 混合精度需要 CUDA；CPU 请用 fp32 或 bf16。")
        if mode == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("该 GPU 不支持 BF16；请选择 fp16 或 fp32。")
        self.mode = mode
        self.device = device

    def autocast(self):
        if self.mode == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.mode == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)

    def scaler(self):
        enabled = self.mode == "fp16"
        if hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)
