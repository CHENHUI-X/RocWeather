"""命令行统一入口：train、evaluate、predict。"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from .checkpoints import is_training_checkpoint, load_weights, read_checkpoint
from .config import Config, config_from_dict, load_config
from .data import IMAGE_SUFFIXES, image_tensor, model_input, open_rgb, plan_image_outputs, save_rgb
from .evaluation import evaluate, validation_loader
from .models import RocWeather
from .restoration import restore
from .runtime import Precision, setup_runtime
from .training import train


def _parser():
    parser = argparse.ArgumentParser(prog="rocweather", description="统一恢复雨、雪、雾图像")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="YAML 配置文件")
    common.add_argument("--device", help="auto、cpu、cuda 或 cuda:<编号>")
    common.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"))
    common.add_argument(
        "--trust-legacy-checkpoint", action="store_true", help="允许读取你信任的旧版断点"
    )
    common.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)

    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", parents=[common], help="训练（直接启动或用 torchrun）")
    training.add_argument("--output", type=Path, help="训练输出目录；续训时默认使用断点所在目录")
    training.add_argument("--epochs", type=int, help="覆盖训练总轮数（包括已经完成的轮数）")
    weights = training.add_mutually_exclusive_group()
    weights.add_argument("--resume", type=Path, help="从新版断点继续训练")
    weights.add_argument("--weights", type=Path, help="仅加载权重并开始新训练")

    evaluation = commands.add_parser("evaluate", parents=[common], help="评估配对数据集")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, help="可选：保存恢复后的图片")

    prediction = commands.add_parser("predict", parents=[common], help="恢复单张图片或整个目录")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True, help="保存预测图片的目录")
    return parser


def _config(args, checkpoint):
    saved = config_from_dict(checkpoint["config"]) if is_training_checkpoint(checkpoint) else None
    config = load_config(args.config) if args.config else (saved or Config())
    if saved is not None and args.command != "train":
        # 新版断点自带模型结构；评估/预测的外部配置只覆盖数据、运行与分块设置。
        config = replace(config, model=saved.model)
    overrides = {
        key: getattr(args, key) for key in ("device", "precision") if getattr(args, key) is not None
    }
    return replace(config, runtime=replace(config.runtime, **overrides))


def _prediction_inputs(source, output):
    if source.is_file():
        if source.suffix.lower() in IMAGE_SUFFIXES:
            return source.parent, [source]
    elif source.is_dir():
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("目录推理的输出目录须位于输入目录之外。")
        images = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            return source, images
    raise ValueError(f"没有找到可读图像：{source}")


@torch.inference_mode()
def _predict(model, source, output, eval_config, precision):
    root, images = _prediction_inputs(source, output)
    names = [path.relative_to(root).as_posix() for path in images]
    destinations = plan_image_outputs(output, names)
    for path, name in zip(images, names):
        image = model_input(image_tensor(open_rgb(path))).unsqueeze(0).to(precision.device)
        with precision.autocast():
            prediction = restore(model, image, eval_config)
        save_rgb(prediction[0], destinations[name])
    print(f"已恢复 {len(images)} 张图片，保存到 {output}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command == "train":
            path = args.resume or args.weights
            if args.output is None and args.resume is None:
                parser.error("新训练必须指定 --output。")
        else:
            path = args.checkpoint
        checkpoint = (
            read_checkpoint(path, trust_legacy=args.trust_legacy_checkpoint) if path else None
        )
        config = _config(args, checkpoint)

        if args.command == "train":
            if args.epochs is not None:
                config = replace(config, train=replace(config.train, epochs=args.epochs))
            output = args.output if args.output is not None else args.resume.parent
            train(config, output, checkpoint=checkpoint, resume_path=args.resume)
            return

        with setup_runtime(config.runtime) as runtime:
            if args.command == "predict" and runtime.distributed:
                raise ValueError("predict 请使用单进程；evaluate 支持 DDP。")
            precision = Precision.for_runtime(config.runtime.precision, runtime)
            model = RocWeather(config.model)
            load_weights(model, checkpoint)
            del checkpoint
            model.to(runtime.device).eval()
            if args.command == "evaluate":
                summary = evaluate(
                    model,
                    validation_loader(config, runtime),
                    config.eval,
                    runtime,
                    precision,
                    args.output,
                )
                if runtime.primary:
                    print(json.dumps(summary, ensure_ascii=False))
            else:
                _predict(model, args.input, args.output, config.eval, precision)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"rocweather: {exc}\n")
