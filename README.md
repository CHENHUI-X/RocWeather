# RocWeather

RocWeather 用同一个模型恢复受雨、雪、雾影响的图像。输入低质 RGB 图，输出同尺寸恢复图，无需在推理时指定天气。

本轮保留原代码的主干：四级 RPLFA 注意力与 Focal CNN 并行编码，最深层由可学习的任务查询读取特征，卷积解码器通过跳连逐级恢复空间分辨率。任务查询是隐式调节，没有单独输出雨、雪、雾类别。不同天气对应的频段是后续做消融与频谱分析的假设。

## 安装

需要 Python 3.10 及以上。已有能使用 CUDA 的 PyTorch 环境，激活后直接安装本项目；新建环境时先按 [PyTorch 安装指引](https://pytorch.org/get-started/locally/) 安装对应的 `torch` 和 `torchvision`。

```bash
python -m pip install -e .
python -m rocweather --help
```

`rocweather` 与 `python -m rocweather` 是同一入口。下列命令均从仓库根目录启动。

## 数据

```text
data/
  train/{train.txt,input/,gt/}
  test/{test.txt,input/,gt/}
```

- 每行 `input/example.png` 自动配对 `gt/example.png`；若文件不存在，会尝试 PNG、JPG、JPEG 扩展名。
- 自己组织的数据也可用 **Tab** 分隔低质图与清晰图路径，路径相对于清单所在目录。修改 YAML 的 `data.root`、`train_manifest`、`val_manifest` 即可切换数据。
- 训练图和标签须像素对齐。两张图使用相同的随机裁剪；紧凑配置另启用同步水平翻转。
- 本仓库测试集部分输入与标签尺寸不同，因此默认缩放到 256×256 后评估。确认原图对齐后，可设 `eval.resize: 0` 做原尺寸评估。正式比较时，建议用独立验证集选最佳断点，留出测试集做最终报告。

## 训练

[紧凑配置](configs/compact.yaml) 减少通道、块数和浅层注意力 K/V，并关闭 VGG 与清晰图恒等损失，适合显存有限时试训。[原结构基线](configs/baseline.yaml) 保留旧模型宽度、深度和四类损失；开启感知损失时，首次运行会下载未缓存的 VGG16 权重。

```bash
# 单张 GPU
python -m rocweather train --config configs/compact.yaml --device cuda --output runs/compact

# 单机多卡，每张卡一个进程
torchrun --standalone --nproc-per-node=4 -m rocweather train --config configs/baseline.yaml --device cuda --output runs/baseline
```

CPU 调试用 `--device cpu`。`runtime.precision: auto` 在 CPU 用 FP32，所有 CUDA 进程均支持 BF16 时用 BF16，否则统一用 FP16。也可用 `--precision` 指定。FP16 时深层下采样和卷积重建保留 FP32，避免旧权重溢出。显存不足时可试紧凑配置、降低每卡批大小或减小 `data.crop_size`。

DDP 的 `train.batch_size` 是每个进程的批大小，乘进程数得到常规步的全局批大小。CUDA 优先使用 NCCL，CPU 使用 Gloo；原生 Windows 没有 NCCL，GPU 集群建议使用 Linux。多机启动用 [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)，每台机器使用相同的 `--nnodes`、`--nproc-per-node` 和会合地址、配置与数据，训练输出由全局 rank 0 写入。

### 续训和权重迁移

每轮保存 `last.pt`，验证 PSNR 刷新时保存 `best.pt`。目录还保存实际配置 `config.yaml` 和逐轮日志 `metrics.jsonl`。

```bash
# 按原配置继续训练到总共 80 轮
python -m rocweather train --resume runs/compact/last.pt --epochs 80

# 严格加载旧模型权重，按新训练设置重新开始
python -m rocweather train --config configs/baseline.yaml --weights checkpoint/2023_12_16_17_44_12/best_model.pth --trust-legacy-checkpoint --output runs/from-legacy
```

新版断点恢复模型、AdamW、学习率调度、AMP 缩放器、轮次与最佳 PSNR。历史断点可用 `--output` 分到新目录，该目录会从续训后的评估开始选 `best.pt`。换设备或进程数仍可续训，但批大小或浮点舍入会改变后续轨迹；切换精度会重新初始化 AMP 缩放器。断点使用标准 `torch.save` ZIP 格式。旧文件的优化器和调度器不迁移；仅对含序列化对象且来源可信的旧文件使用 `--trust-legacy-checkpoint`。

## 评估和推理

```bash
# 逐图计算 PSNR / SSIM 后平均；--output 可选
python -m rocweather evaluate --checkpoint runs/compact/best.pt --output runs/compact/predictions

# 单张图；--input 也可传目录，会保留输入的子目录
python -m rocweather predict --checkpoint runs/compact/best.pt --input data/test/input/0001.jpg --output runs/restored
```

`evaluate` 也支持 `torchrun`，每张验证图只统计一次。默认读取断点中的数据和评估设置；可用 `--config` 切换清单、尺寸或分块设置，新版模型结构始终以断点为准。图片输出使用新路径，评估结果放在配对数据目录之外。推理保留原图尺寸；大图默认按 256 像素分块，重叠 32 像素并渐变融合。显存充足可设 `eval.tile_size: 0` 尝试整图推理；分块会缩小全局注意力能看到的范围，输出可能不同。

## 代码组织

| 路径 | 职责 |
| --- | --- |
| `configs/` | 实验配置 |
| `rocweather/cli.py` | 训练、评估和推理入口 |
| `rocweather/models/` | 编码、任务查询与卷积重建 |
| `rocweather/data.py` | 配对、裁剪和图像读写 |
| `rocweather/losses.py`、`rocweather/metrics.py` | 训练损失和质量指标 |
| `rocweather/training.py`、`rocweather/runtime.py` | 训练循环、设备、DDP 和混合精度 |
| `rocweather/checkpoints.py` | 断点保存与权重迁移 |
| `rocweather/evaluation.py`、`rocweather/restoration.py` | 验证汇总与分块恢复 |

注意力使用 PyTorch SDPA，是否能使用更快的计算内核取决于 PyTorch、设备和精度。新版改用逐轮衰减的 AdamW，PSNR/SSIM 在裁剪到 [0, 1] 的恢复图上计算，旧日志不可直接横比。比较配置时请固定测试清单、尺寸和精度，并记录图像质量、速度和峰值显存。
