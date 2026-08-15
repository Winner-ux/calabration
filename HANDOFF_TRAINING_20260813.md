# Fusion ResNet 校准数据初训与融合结果交接文档

更新时间：2026-08-13  
项目目录：`C:\Users\qwhzn\Desktop\Fusion_resnet`  
文档范围：校准数据导入、正式划分与审计、10 epoch 初步训练、20 张融合图导出、损失曲线和初步质量评价。

## 1. 当前结论

本轮已经完成从校准图像到训练、独立测试集评分、20 张全分辨率融合图导出的基础闭环。

- 正式候选数据为 3,900 对 IR/RGB 图像，来自 8 个完整 experiment。
- 按 experiment 作为 `leakage_group_id`，使用 seed 42 确定性划分为：train 2,250、val 600、test 1,050。
- 正式数据审计结果为 `formal_ready=true`、`fatal_count=0`。
- 已完成 10 epoch 初步训练。最终稳定验证损失最低的 checkpoint 为 epoch 9。
- 已在完整的 1,050 张独立测试图上评分，并导出 20 张全分辨率融合图。
- 当前模型可作为“基础融合基线”，但不应表述为高质量最终模型：融合结果偏 RGB、对比度下降，部分区域存在轻微网格纹理。

## 2. 数据来源与边界

原始校准图像目录：

```text
C:\Users\qwhzn\Desktop\dataset\calibrated_output
```

项目内独立数据根目录：

```text
C:\Users\qwhzn\Desktop\Fusion_resnet\datasets\calibrated_v1
```

处理原则：

- 原始 `calibrated_output` 未删除、未覆盖。
- 原有 `data/` 未删除或替换。
- IR/RGB 图像使用 NTFS 硬链接导入，元数据复制到独立数据根目录。
- 上游 `processing_workspace\output\experiment` 只用于帧号和源视频追溯元数据。
- 缺少 `health+sick` 的 `day5/experiment2` 被排除；正式训练候选为 8 个完整 experiment、3,900 对图像。

关键数据文件：

```text
datasets/calibrated_v1/dataset.yaml
datasets/calibrated_v1/manifests/samples.csv
datasets/calibrated_v1/manifests/groups.csv
datasets/calibrated_v1/splits/calibrated_v1/train.csv
datasets/calibrated_v1/splits/calibrated_v1/val.csv
datasets/calibrated_v1/splits/calibrated_v1/test.csv
datasets/calibrated_v1/splits/calibrated_v1/split_config.json
datasets/calibrated_v1/splits/calibrated_v1/audit.json
datasets/calibrated_v1/quality/registration.csv
datasets/calibrated_v1/quality/near_duplicates.csv
```

可复现标识：

- dataset manifest SHA-256：`492c73792fada656d72f520b18d95a57be64c75711b1bcf3d5e3ee52181477f8`
- test manifest SHA-256：`418b80f2ac4f8ec8a3332f0c86437aa0c0ec3b8ccf005ecdf4e5f55d99901852`
- split 算法：`deterministic_nested_group_split_v2`
- split schema：v2
- seed：42

## 3. 正式划分与审计

划分单元是完整 experiment，不是单帧或单个类别 clip。实际划分数量：

| Split | 图像对数 |
|---|---:|
| train | 2,250 |
| val | 600 |
| test | 1,050 |
| 合计 | 3,900 |

审计报告中的已验证事实：

- `formal_ready=true`
- `fatal_count=0`
- `warning_count=0`
- `sample_count=3900`
- `usable_sample_count=3900`
- `experiment_count=8`
- `group_count=8`
- `near_duplicate_candidate_count=2023`
- `registration_manual_review_count=1462`

需要特别注意：`registration_manual_review_count=1462` 是低边缘相关性阈值触发的人工复核提示，不等于 1,462 对图像已经确认配准错误。正式审计允许训练，但配准质量尚未完成逐张人工验收。`day4/experiment2` 和 `day8/experiment1` 还存在 calibration diversity warning，校准模型应继续视为 provisional。

## 4. 代码实现与关键修正

本轮主要实现或调整如下：

- 增加 `main/__init__.py`，统一支持 `python -m main.train`、`python -m main.inference` 等 module-style execution。
- 包内导入改为相对导入，保持 `model` 为模型模块边界。
- 扩展 `scripts/import_paired_data.py`：支持独立 `--metadata-root`、硬链接导入、完整 experiment 过滤、追溯元数据和哈希检查。
- 优化 `data_pipeline/audit.py` 的跨 split 近重复报告，限制 pHash 候选数量并保留扫描/截断统计。
- `main/train.py` 增加分量损失日志、运行环境记录、AMP、checkpoint/resume 和正式审计校验。
- `main/loss.py` 修正 SSIM 数值稳定性，并确保关键损失计算使用 FP32。
- `model/attention.py` 使用 PyTorch SDPA，降低注意力显存和时间开销，但保持张量形状契约。
- 新增 `main/select_best.py`，完成测试集逐图代理评分、代表性选择和全分辨率滑窗导出。
- 新增 `scripts/benchmark_dataloader.py`，用于 workers 0/2 基准比较。
- 新增 `scripts/plot_training_curves.py`，生成训练损失曲线。

模型张量契约未改变：

```text
VIS/RGB: FloatTensor[B, 3, H, W]
IR:      FloatTensor[B, 1, H, W]
Output:  FloatTensor[B, 3, H, W]
Value range: [0, 1]
```

checkpoint 仍使用当前训练 checkpoint 格式，未有意改变模型参数名或权重布局。

## 5. 训练环境与配置

实际记录的运行环境：

```text
Python: 3.12.0
PyTorch: 2.5.1
CUDA build reported by PyTorch: 12.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
GPU memory: 8,585,216,000 bytes
Interpreter: D:\anaconda\envs\pytorch\python.exe
```

正式训练配置：

```text
seed: 42
crop: 448 x 448
batch_size: 2
optimizer: Adam
AMP: true
epochs: 10
sampler: epoch_shuffle
num_workers: 0
```

由于训练中发现数值稳定性问题，10 个 epoch 分为三个阶段。不同损失定义下的原始数值不能简单连接比较：

| 阶段 | Epoch | 学习率 | 关键状态 |
|---|---|---:|---|
| A | 0–1 | `1e-4` | 旧 SSIM 实现出现数值异常，原始 val loss 不可用于选模 |
| B | 2–5 | `2e-5` | 损失切到 FP32，数值恢复有限且可解释 |
| C | 6–9 | `1e-5` | 稳定 SSIM、SDPA、重置 optimizer，最终选模阶段 |

原始逐 epoch 总损失：

| Epoch | Train loss | Val loss | 说明 |
|---:|---:|---:|---|
| 0 | 1.415948 | 0.095681 | 旧损失定义，不可与最终阶段直接比较 |
| 1 | 1.351354 | -0.392555 | SSIM 数值异常产生伪低值，不是有效改进 |
| 2 | 1.241312 | 0.629119 | FP32 损失阶段开始 |
| 3 | 0.928816 | 0.270340 |  |
| 4 | 0.772401 | 0.266403 |  |
| 5 | 0.713598 | 0.245180 |  |
| 6 | 0.674525 | 0.236082 | 最终稳定损失阶段开始 |
| 7 | 0.652228 | 0.231612 |  |
| 8 | 0.633415 | 0.227076 |  |
| 9 | 0.614975 | 0.223159 | 最终选择 |

使用最终稳定损失重新评估关键 checkpoint：

| Checkpoint epoch | 稳定 val loss |
|---:|---:|
| 1 | 0.915884 |
| 5 | 0.245186 |
| 9 | 0.223159 |

据此，epoch 9 相对 epoch 5 改善约 9.0%，且是当前可比较 checkpoint 中的最佳结果。epoch 1 的原始负损失必须视为无效数值，不能作为 best checkpoint。

## 6. 最终 checkpoint

正式交付 checkpoint：

```text
runs/calibrated_v1/final_selection_checkpoint.pt
```

元数据：

```text
epoch: 9
SHA-256: 045cddacf7e07d834bce1d98fe19a6439e6218ae08d630dc318e1facbfa916fc
```

最终阶段训练目录：

```text
runs/calibrated_v1/train-20260812-210920-731613/
  best.pt
  last.pt
  metrics.csv
  run_config.json
```

不要使用 epoch 1 的原始“最优”记录，因为它来自不稳定 SSIM 数值。继续训练时应从 `final_selection_checkpoint.pt` 或最终阶段的 epoch 9 checkpoint 恢复，并保持最终稳定损失定义。

## 7. 20 张融合图导出

结果目录：

```text
runs/calibrated_v1/best20-20260812-214141-423689/
```

目录内容：

```text
fused/             # 严格 20 张融合 PNG
ranking.csv        # 排名、样本信息、各损失分量和选择理由
contact_sheet.png  # 20 张缩略总览
run.json           # checkpoint、split、哈希和输出记录
```

导出策略：

- 使用 epoch 9 checkpoint 对 1,050 张独立 test 图进行确定性 `448 x 448` 中心裁剪评分。
- 主要代理分数是当前 `FusionLoss` 的逐图总损失，越低越优。
- 按 6 个测试 clip 各选 3 张，再补充全局最佳候选，并使用 pHash 和源帧间距避免近重复画面垄断结果。
- 仅对最终 20 张运行全分辨率滑窗推理：tile 448、overlap 64、Hann 加权。
- 20 张输出均为 RGB PNG，尺寸均为 `1920 x 1080`，输出哈希已记录。

测试集代理分数统计：

```text
mean: 0.244169
min:  0.136993
max:  0.446235
```

这些分数是无 ground truth 条件下的目标函数代理，不是绝对视觉质量分数，也不能直接证明下游鱼体/健康状态检测性能。

## 8. 损失曲线与融合效果评价

损失曲线：

```text
runs/calibrated_v1/analysis/loss_curves.png
runs/calibrated_v1/analysis/loss_curves.svg
```

生成脚本：

```text
scripts/plot_training_curves.py
```

### 8.1 已验证的正面表现

- 最终稳定阶段 train/val loss 同步下降，未发现明显发散或严重过拟合。
- 水箱结构、主要边缘和部分小目标基本保留。
- 20 张全分辨率结果未发现明显滑窗拼接缝。
- 对 20 张结果缩放到 `512 x 288` 后的辅助统计显示，融合图平均梯度为 `0.0170`，约为较强输入梯度的 89.4%。

### 8.2 主要问题

- 输出对比度偏低：融合灰度标准差 `0.1175`，RGB 为 `0.1463`，约保留较强输入对比度的 80.3%。
- 融合明显偏向 RGB：融合图与 RGB 的互信息约 `3.1926 bits`，与 IR 约 `0.3765 bits`。
- 部分画面整体偏平、偏灰褐色，IR 显著目标没有被充分强化。
- 部分区域可见轻微棋盘格/网格纹理，推测与 `ConvTranspose` 解码上采样有关；这是待验证原因，不是已完成的因果证明。
- 饱和度统计不能简单支持“饱和度不足”：融合平均饱和度 `0.1741`，RGB 为 `0.1284`。当前视觉发灰更可能来自对比度下降和统一色偏。

### 8.3 总体评价

当前模型已经是可运行、可复现的基础融合基线，但尚不适合作为最终高质量模型。最需要继续优化的方向是：

1. 提高 IR 信息进入融合结果的比例和目标显著性。
2. 改善局部对比度，避免整体动态范围被压缩。
3. 检查并缓解解码上采样造成的网格伪影。
4. 完成人工配准抽检，并引入更可靠的无参考指标或下游任务指标。

## 9. 已执行验证

本轮工作实际完成或检查过：

- Python 包导入和 module-style execution 检查。
- CPU synthetic forward 与目标损失检查。
- GPU 单步 smoke。
- 数据 manifest、split 和 full audit。
- 10 epoch 正式训练，train/val 分量损失均有日志。
- 关键 checkpoint 使用最终稳定损失重新评估。
- 1,050 张 test 图完整代理评分。
- 20 张全分辨率融合输出的数量、模式、尺寸和 SHA-256 记录。
- 损失曲线 PNG/SVG 渲染和视觉布局检查。
- 20 张融合总览人工视觉检查。

常用测试命令：

```powershell
D:\anaconda\envs\pytorch\python.exe -B -m unittest discover -s tests -v
```

绘制损失曲线：

```powershell
D:\anaconda\envs\pytorch\python.exe scripts\plot_training_curves.py
```

## 10. 当前工作树状态

截至本交接文档生成时，项目工作树包含尚未提交的代码修改和新增文件。不要使用 `git reset --hard`、`git checkout -- .` 等命令清理，否则可能丢失本轮实现。

已修改文件包括：

```text
data/dataset.yaml
data_pipeline/audit.py
main/inference.py
main/loss.py
main/train.py
model/attention.py
scripts/import_paired_data.py
tests/test_data_pipeline.py
tests/test_inference.py
```

新增未跟踪文件包括：

```text
main/__init__.py
main/select_best.py
scripts/benchmark_dataloader.py
scripts/plot_training_curves.py
```

数据集、checkpoint 和生成图像属于用户数据/输出，不要删除、覆盖或重新生成，除非用户明确要求。

## 11. 剩余风险与推荐后续顺序

### 已知风险

1. 无融合 ground truth，现有损失和统计只能作为代理。
2. test 集已经用于一次性结果展示，不应继续反复根据 test 图调参；后续架构和 loss 选择应使用 train/val。
3. 1,462 对配准样本触发人工复核提示，尚未逐张确认。
4. 只有 8 个 experiment，跨对象、跨场景泛化证据有限。
5. 当前环境尚未形成锁定依赖文件；只能声明上述实际测试版本，不能假设其他环境等价。

### 推荐执行顺序

1. 先对 `registration.csv` 中的人工复核候选分层抽样检查，确认偏差是否会损害训练。
2. 在不接触 test 的前提下，用 val 比较 IR 权重、局部对比度 loss 和上采样实现。
3. 增加至少一种公认无参考融合指标，并用下游目标检测/分割指标验证 IR 信息是否真正有用。
4. 对确认有效的改动重新训练一个新 run；保留当前 checkpoint 和输出作为 baseline，不覆盖旧结果。
5. 待方案稳定后再建立新的独立 test 或追加实验数据，避免对当前 test 过拟合。

## 12. 新对话可直接使用的开场说明

> 请先阅读 `C:\Users\qwhzn\Desktop\Fusion_resnet\AGENTS.md` 和 `C:\Users\qwhzn\Desktop\Fusion_resnet\HANDOFF_TRAINING_20260813.md`，再核对当前 Git 工作树。当前已完成 3,900 对校准图的 experiment 级正式 split、10 epoch 初训、epoch 9 checkpoint 选择和 20 张全分辨率融合图导出。不要把 epoch 1 的负 val loss 当成有效最优值，不要删除或覆盖 `datasets/`、`runs/` 和现有用户输出。下一阶段优先在 train/val 上处理 IR 信息偏弱、对比度下降、网格伪影和配准人工复核问题，不要反复使用当前 test 调参。
