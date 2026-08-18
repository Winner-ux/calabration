# Fusion ResNet

这是一个进行中的 PyTorch IR/RGB 图像融合项目。当前数据入口采用 manifest，模型输入
契约为：

- VIS：`FloatTensor[B,3,H,W]`
- IR：`FloatTensor[B,1,H,W]`
- 输入值域：`[0,1]`
- 训练裁剪默认：448×448

## RGB/IR 相机标定

棋盘格标定、IR → RGB 配准、批量模型应用和几何质量检查已整理到
[`calibration_tools/`](calibration_tools/README.md)。该模块原目录名
`calabration/` 存在拼写错误，本次统一采用 `calibration_tools`；原始图像和生成结果
均由 `.gitignore` 排除，不进入公开仓库。

## 已验证环境

本次实际验证使用：Python 3.12.0、PyTorch 2.5.1、CUDA 12.1、RTX 4060
Laptop GPU。对应解释器为：

```text
D:\anaconda\envs\pytorch\python.exe
```

项目现有 `.venv` 未安装 PyTorch 等依赖，因此以下命令应在已安装依赖的环境中运行。

## 数据工作流

完整契约见 [`data/README.md`](data/README.md) 和
[`data/dataset.yaml`](data/dataset.yaml)。正式图片只允许存在于 `data/paired/`，
split CSV 只保存 `sample_id`。

外部参考数据 dry-run（不会复制）：

```powershell
python scripts/import_paired_data.py `
  --source-root "C:\Users\qwhzn\Desktop\dataset\processing_workspace\output\experiment"
```

如以后明确决定导入，再在相同命令后增加 `--execute-copy`。工具不会覆盖内容不同的
已有文件。

导入后必须人工填写：

```text
data/manifests/group_assignments.csv
```

然后依次执行：

```powershell
python scripts/build_manifest.py
python scripts/make_split.py --version v1 --seed 42
python scripts/audit_dataset.py --split-version v1 --overwrite
```

每个实验必须包含 `health`、`health_sick`、`sick` 三类且每类恰好一个 clip；
同一实验的三个 clip 必须使用同一个 `leakage_group_id`。少于 6 个完整独立
group 时，`make_split.py` 会拒绝生成正式 split。

划分采用两阶段固定策略：先按 group 划分 `80% train_pool / 20% test`，再从
`train_pool` 划分 `80% train / 20% val`，有效目标约为 `64/16/20`。输出同时
包含 `train_pool.csv`、`train.csv`、`val.csv`、`test.csv` 以及 group/experiment
assignment。数据增加后使用 `v2`、`v3` 等新版本，不能覆盖旧版本。

## Dataset

```python
from dataset import PairedFusionDataset

dataset = PairedFusionDataset(
    data_root="data",
    samples_manifest="data/manifests/samples.csv",
    split_manifest="data/splits/v1/train.csv",
    paired_transform=paired_transform,
    rgb_transform=rgb_transform,
    ir_conversion="bt601",
)
```

当前 RGB 编码的 IR PNG 在内存中使用 BT.601 转为单通道，源文件不会被修改。

## 最小训练入口

只有 full audit 通过的 formal split 能启动正式训练：

```powershell
python train.py --split-version v1
```

split 在训练前固定，`train.py` 不会动态重划验证集。训练集使用确定性的
`epoch_shuffle`：同一 seed/epoch 顺序相同，不同 epoch 顺序不同；验证集和测试集
保持固定顺序。测试集不会在训练 epoch 中反复使用。

单步集成检查：

```powershell
python train.py --split-version v1 --smoke-only --max-steps 1
```

`--smoke-only` 不保存 `best.pt` 或 `last.pt`，不能作为正式训练结果。

## 融合推理与图像导出

单对同名 PNG 使用训练 checkpoint 推理：

```powershell
python inference.py `
  --vis "path\to\rgb\000001.png" `
  --ir "path\to\ir\000001.png" `
  --checkpoint "runs\train-...\best.pt"
```

严格按相对路径批量配对目录：

```powershell
python inference.py `
  --vis-dir "path\to\rgb" `
  --ir-dir "path\to\ir" `
  --checkpoint "runs\train-...\best.pt"
```

正式训练完成后，严格按已完整审计的 `test.csv` 导出持出集融合图像：

```powershell
python inference.py `
  --data-root data `
  --split-version v1 `
  --split test `
  --checkpoint "runs\train-...\best.pt"
```

该模式拒绝非 formal split、未通过 full audit 的 split、跨 split 重复样本/防泄漏组
以及随机未训练权重。输出保留 day/experiment/category/clip 层级，并在 `run.json`
中记录 checkpoint、dataset manifest 和 test manifest 的 SHA-256。

默认 `--mode auto`。当 pad 后首层全局注意力 token 数超过 `12544` 时，自动
回退到 `448×448`、overlap 64 的 Hann 加权滑窗；这避免 1920×1080 直接整图
注意力造成显存耗尽。也可显式选择 `--mode full` 或 `--mode sliding`。超过安全
阈值的 `full` 会提前拒绝，除非显式传入 `--force-full`。

当前项目尚无正式训练 checkpoint。若只检查代码路径，可显式使用：

```powershell
python inference.py `
  --vis "path\to\rgb\000001.png" `
  --ir "path\to\ir\000001.png" `
  --allow-random-weights `
  --device cpu
```

这种输出的文件名带 `SMOKE_UNTRAINED__` 前缀，`run.json` 也会标记为
`smoke_only_untrained`；它只能证明读取、模型前向和保存流程可运行，不能用于
融合质量、检测性能或科研结论。

## 测试

测试只使用临时合成数据，不复制或训练外部参考集：

```powershell
D:\anaconda\envs\pytorch\python.exe -B -m unittest discover -s tests -v
```
