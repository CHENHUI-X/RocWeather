# RocWeather

RocWeather 用同一个模型恢复受雨、雪、雾影响的图像。输入低质 RGB 图，输出同尺寸恢复图，无需在推理时指定天气。

本轮保留原代码的主干：四级 RPLFA 注意力与 Focal CNN 并行编码，最深层由可学习的任务查询读取特征，卷积解码器通过跳连逐级恢复空间分辨率。任务查询是隐式调节，没有单独输出雨、雪、雾类别。不同天气对应的频段是后续做消融与频谱分析的假设。

## 自适应结构

[自适应配置](configs/adaptive.yaml) 使用 `model.variant: adaptive`。它保留四级双路编码、两层中心解码和卷积重建，使用以下连接：

- Transformer 与 Focal CNN 各自使用块内残差，每级通过按通道学习的权重融合，初始权重各为 0.5。
- 浅层特征、深层特征和输入频带统计共同生成 64 维连续退化条件。中心解码器用它偏移空间查询，重建头用它与局部特征一起调节三个尺度的跳连。最深层特征通过中心解码器接入重建头。
- 原图通过两种尺度的高斯平滑拆成互补的低、中、高频提示，分别注入 1/32、1/8 与 1/4、原分辨率的解码特征。各频带都有独立的条件权重。输出是在输入 RGB 上加学习到的校正量。
- 原分辨率的高频注入同时参考全图条件和当前位置的解码特征。局部门控用并行的 3×3、5×5 深度卷积补充不同尺度的纹理线索。先前的自适应断点可按其中的原配置续训，也可用 `--weights` 配合新版 YAML 开启新训练，此时只初始化新增的门控参数。
- 大图分块时，场景特征来自整图缩略图，频带统计按原像素尺度分块汇总。两者生成一份供所有图块使用的退化条件，每块另有自己的局部频带提示。

旧配置和缺少 `variant` 字段的断点默认使用 `legacy`。自适应配置有自己的参数形状和权重，选择这套配置时从新的模型权重开始。

## 安装

需要 Python 3.10 及以上。已有能使用 CUDA 的 PyTorch 环境，激活后直接安装本项目；新建环境时先按 [PyTorch 安装指引](https://pytorch.org/get-started/locally/) 安装对应的 `torch` 和 `torchvision`。

```bash
python -m pip install -e .
python -m rocweather --help
```

`rocweather` 与 `python -m rocweather` 是同一入口。下列命令均从仓库根目录启动。

自适应配置开启 TensorBoard 日志，可用 `python -m pip install -e '.[tensorboard]'` 安装所需依赖。

## 数据

```text
data/
  train/{train.txt,input/,gt/}
  test/{test.txt,input/,gt/}
```

- 每行 `input/example.png` 自动配对 `gt/example.png`；若文件不存在，会尝试 PNG、JPG、JPEG 扩展名。
- 自己组织的数据也可用 **Tab** 分隔低质图与清晰图路径，路径相对于清单所在目录。修改 YAML 的 `data.root`、`train_manifest`、`val_manifest` 即可切换数据。
- 训练图和标签须像素对齐。两张图使用同一组随机裁剪和几何变换。自适应配置默认启用 `horizontal_flip`，小图用 `small_image_mode: edge_pad` 保持雨纹、雪点原来的像素尺度。旋转可通过 `rotate_90: true` 启用，随机选择 0°、90°、180° 或 270°，无需插值。姿态固定的场景可沿用默认的关闭设置。
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

启用 `train.tensorboard` 后，实验目录下的 `tensorboard/` 记录生效的模型和训练配置、优化器设置、逐步学习率和各项损失、裁剪前梯度范数，以及每轮的验证指标、耗时和显存。如果使用 DDP，逐步曲线和显存来自 rank 0，每轮训练和验证指标均为全局汇总。以下命令在本机启动看板。

```bash
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

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
