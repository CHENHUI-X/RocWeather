"""区分继续训练与仅加载模型权重；只让主进程写断点。"""

import logging
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

FORMAT_VERSION = 1


def read_checkpoint(path: Path, *, trust_legacy=False):
    with path.open("rb") as file:
        # 只允许标准 torch.save ZIP，避免旧 PyTorch 的历史归档加载路径和 TorchScript 自动分派。
        # https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6
        if file.read(4) != b"PK\x03\x04":
            raise ValueError(f"{path} 不是标准的 torch.save ZIP 断点。")
        try:
            with ZipFile(file) as archive:
                filenames = {name.rsplit("/", 1)[-1] for name in archive.namelist()}
        except BadZipFile as exc:
            raise ValueError(f"断点 ZIP 已损坏：{path}") from exc
        if "data.pkl" not in filenames or "constants.pkl" in filenames:
            raise ValueError(f"{path} 不是 RocWeather 可用的权重或训练断点。")
        file.seek(0)
        # 老版保存过完整调度器对象。显式确认来源可信时才允许反序列化旧对象。
        try:
            return torch.load(file, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            if not trust_legacy:
                raise ValueError(
                    f"无法读取 {path} 中的旧对象。若这是你可信的旧 RocWeather 文件，"
                    "可加 --trust-legacy-checkpoint 后仅加载权重。"
                ) from exc
            file.seek(0)
            return torch.load(file, map_location="cpu", weights_only=False)


def load_weights(model, checkpoint, *, allow_detail_upgrade=False):
    if not isinstance(checkpoint, dict):
        raise ValueError("权重文件必须包含模型 state_dict。")
    state = next(
        (checkpoint[key] for key in ("model", "net", "state_dict") if key in checkpoint), checkpoint
    )
    if not isinstance(state, dict) or not state:
        raise ValueError("权重文件的模型 state_dict 为空或无效。")
    state = state.copy()
    consume_prefix_in_state_dict_if_present(state, "module.")
    # 旧版注册了四组从未参与前向的 LayerNorm。
    if "format_version" not in checkpoint:
        for index in range(4):
            for field in ("weight", "bias"):
                state.pop(f"Tenc.norm.{index}.{field}", None)
    upgraded = False
    if allow_detail_upgrade:
        current = model.state_dict()
        added = {name for name in current if name.startswith("convtail.band_hints.5.spatial_gate.")}
        saved_model = (
            checkpoint.get("config", {}).get("model", {})
            if is_training_checkpoint(checkpoint)
            else {}
        )
        # 只迁移完全缺少旧版未定义的门控参数的权重。缺失其他权重仍由严格检查报错。
        if (
            added
            and current.keys() - state.keys() == added
            and not saved_model.get("spatial_detail_gate", False)
        ):
            state.update({name: current[name] for name in added})
            upgraded = True
    model.load_state_dict(state, strict=True)
    if upgraded:
        logging.getLogger(__name__).info("旧自适应权重已加载，局部高频门控使用新初始化参数。")


def is_training_checkpoint(checkpoint):
    return isinstance(checkpoint, dict) and checkpoint.get("format_version") == FORMAT_VERSION


def save_checkpoint(path: Path, checkpoint):
    # 同目录写临时文件，完成后一次替换，避免中断时破坏上一个可用断点。
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
