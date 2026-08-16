# Fusion ResNet 数据管线与最小训练入口交接文档

更新时间：2026-08-06  
项目路径：`C:\Users\qwhzn\Desktop\Fusion_resnet`

## 0. 本窗口续作更新

数据划分已进一步改为实验级两阶段策略：先按 `leakage_group_id` 生成
`80% train_pool / 20% test`，再只在 `train_pool` 中生成 `80% train / 20% val`。
split schema 独立升级为 v2，图像/manifest schema 和模型输入保持 v1。训练集改用
确定性的 `epoch_shuffle` sampler；split 固定，但每个 epoch 的训练顺序不同。

交接后的安全代码环节已经继续完成：新增独立融合推理与图像导出入口，但没有
复制外部数据、运行正式训练或生成冒充正式结果的融合图。

- `inference.py`：支持单对 PNG 和递归目录批量严格配对，checkpoint 严格加载，
  输出到唯一 run 目录并记录 `run.json`。
- `inference_utils.py`：实现 schema v1 PNG/模式/尺寸校验、BT.601 IR 转换、32 倍
  padding、整图推理、Hann 加权滑窗、原尺寸恢复和首层注意力 token 风险估算。
- 默认 `auto` 模式：首层 token 数超过 `12544` 时由整图回退到 448 tile、64 overlap
  的滑窗；显式 `full` 超阈值会在 OOM 前拒绝，除非传入 `--force-full`。
- 正常推理必须提供 `--checkpoint`。只有显式 `--allow-random-weights` 才能运行未训练
  smoke；输出文件带 `SMOKE_UNTRAINED__` 前缀，元数据标记
  `smoke_only_untrained`，不能用于质量或检测结论。
- 单元测试现为 15 项，新增任意尺寸恢复、右/下边缘覆盖、恒等模型无接缝、auto
  回退/整图保护、BT.601 契约、checkpoint 严格加载和真实模型 CPU 端到端未训练 smoke。

## 1. 当前任务目标

本阶段目标是为 IR/RGB 图像融合项目建立可维护、可审计、避免视频帧泄漏的数据管理方案，并补全最小训练入口。模型接口保持：

- VIS/RGB：`FloatTensor[B, 3, H, W]`
- IR：`FloatTensor[B, 1, H, W]`
- Dataset 输出值域：`[0, 1]`
- 当前默认同步裁剪：`448 x 448`
- 类别只用于覆盖检查、分层划分和结果分析，不进入融合训练 loss

本阶段不修改网络结构和 loss，不运行正式训练，不修改 `develop_list/`。

## 2. 已完成实现

### 数据目录和配置

- 建立 `data/` 标准结构：`paired/`、`manifests/`、`splits/`、`quality/`、`cache/`。
- `data/paired/` 是唯一正式图片层；split 仅通过 CSV 中的 `sample_id` 引用图片。
- `data/dataset.yaml` 已记录 schema v1、类别映射、图像契约、增强、划分、审计和训练默认参数。
- 当前 IR 源仅允许 8-bit `RGB` PNG，Dataset 在内存中使用 BT.601 转成单通道，不修改源文件。
- 已提供 `data/manifests/group_assignments.csv` 人工映射模板。

### Manifest、划分和审计

- `data_pipeline/schema.py`：配置读取、字段定义、ID/slug 和路径校验。
- `data_pipeline/manifest.py`：严格按 stem 配对 IR/RGB，解析 metadata，生成 SHA-256、pHash、`samples.csv` 和 `groups.csv`。
- `data_pipeline/splitting.py`：确定性两阶段 `leakage_group_id` 划分；6 个完整独立组时生成 `4 train / 1 val / 1 test`，并写出 train pool、group 和 experiment assignment。
- 每个正式实验必须包含三个类别且每类恰好一个 clip；同一实验不能跨 group 或 split。
- `data_pipeline/audit.py`：文件、模式、尺寸、重复、split/group/fingerprint 交集、类别覆盖、配准提示和确定性检查。
- `scripts/import_paired_data.py`：默认 dry-run；只有 `--execute-copy` 才复制，不复制 MP4、不覆盖不同内容的已有文件。
- `scripts/build_manifest.py`：只扫描 `data/paired/` 的受控目录层级。
- `scripts/make_split.py`：生成不可覆盖的版本化 split。
- `scripts/audit_dataset.py`：生成完整 `audit.json`、`registration.csv`、`near_duplicates.csv`。

### Dataset 和同步变换

- `dataset.py` 已实现 `PairedFusionDataset`，只读取 manifest，不扫描图片目录，不根据目录名推断 split。
- VIS 输出 `(3,H,W)`，RGB 编码 IR 经 BT.601 输出 `(1,H,W)`。
- 随机裁剪、padding、水平/垂直翻转和可选旋转对 IR/RGB 使用同一组参数。
- RGB 颜色增强与 IR 分离。
- val/test 使用确定性的中心裁剪或 padding。
- schema v1 遇到原生灰度、16-bit 或混合 IR 类型会拒绝加载，防止静默混用。

### 最小训练入口

- `train.py` 已实现 Adam、AMP、`GradScaler`、checkpoint/resume、训练/验证日志、确定性 seed 和显式 `epoch_shuffle` sampler。
- checkpoint/run metadata 记录 split schema、split manifest 哈希和 sampler；恢复训练时不允许换用另一份 split。
- 正式启动前校验 full audit、manifest hash、sample/group 互斥以及 val/test 类别组覆盖。
- `--smoke-only --max-steps 1` 只运行一个优化 step，不生成正式 best/last checkpoint。
- `model/fusion_net.py`、`model/Encoder.py`、`model/decoder.py` 已清理无用强制 import，并统一使用 package-relative imports。
- 网络层、参数名、输出形状和 loss 未改变，预期保持已有 checkpoint 兼容性。

### 独立推理入口

- `inference.py` 支持 `--vis/--ir` 单对输入和 `--vis-dir/--ir-dir` 批量输入。
- `--mode auto` 根据浅层全局注意力 token 数选择整图或滑窗，1920×1080 默认滑窗。
- checkpoint 支持当前 `train.py` 的 `{"model": state_dict}`、`state_dict` wrapper 和纯
  state dict，全部使用严格 key/shape 校验。
- 输出始终恢复为原图尺寸；目录模式保留相对目录结构，不覆盖已有 run。

## 3. 主要文件

```text
Fusion_resnet/
├─ HANDOFF.md
├─ README.md
├─ dataset.py
├─ train.py
├─ inference.py
├─ inference_utils.py
├─ data/
│  ├─ README.md
│  ├─ dataset.yaml
│  ├─ paired/
│  ├─ manifests/group_assignments.csv
│  ├─ splits/
│  ├─ quality/
│  └─ cache/
├─ data_pipeline/
│  ├─ __init__.py
│  ├─ schema.py
│  ├─ manifest.py
│  ├─ splitting.py
│  ├─ sampling.py
│  ├─ audit.py
│  └─ transforms.py
├─ scripts/
│  ├─ import_paired_data.py
│  ├─ build_manifest.py
│  ├─ make_split.py
│  └─ audit_dataset.py
└─ tests/
   ├─ test_data_pipeline.py
   └─ test_inference.py
```

## 4. 已执行验证

已使用以下实际环境验证，但没有把环境版本写成强制依赖：

```text
Python: 3.12.0
PyTorch: 2.5.1
CUDA runtime reported by PyTorch: 12.1
GPU: RTX 4060 Laptop GPU, 8 GB
Interpreter: D:\anaconda\envs\pytorch\python.exe
```

验证结果：

1. 项目 Python 源文件语法编译通过，共检查 20 个文件。
2. `data_pipeline`、`dataset`、`model`、`train` 导入通过。
3. 模型模块前向通过：VIS `(2,3,224,224)`、IR `(2,1,224,224)`，输出 `(2,3,224,224)`。
4. `unittest discover` 共 19 项测试全部通过，其中数据管线 13 项、推理 6 项。
5. 合成数据使用 6 个完整实验、每实验三类各一个 clip，验证得到 `4/1/1`，相同 seed 生成相同划分。
6. 缺失配对、尺寸不一致、非法路径、缺失人工 group mapping 等错误会被拒绝。
7. Dataset 的通道、范围、同步裁剪/翻转和 val/test 确定性通过。
8. 完全重复、近重复候选、跨 split fingerprint/group 风险能进入审计结果。
9. CPU DataLoader 和单步训练 smoke test 通过；输入为 `64 x 64`、batch size 2，只执行一个 step，没有生成正式 checkpoint。

复现测试命令：

```powershell
D:\anaconda\envs\pytorch\python.exe -B -m unittest discover -s tests -v
```

注意：项目自带 `.venv` 当前没有完整安装 PyTorch 等依赖，不能假设它可直接运行。

## 5. 当前外部数据的只读检查结果

外部参考数据路径：

```text
C:\Users\qwhzn\Desktop\dataset\processing_workspace\output\experiment
```

已执行导入 dry-run，未复制、删除、覆盖或修改任何外部数据。检查结果：

- 共发现 3 个 clip、36 对 IR/RGB 图片。
- 三类各 1 个 clip：`health`、`health+sick`、`sick`。
- `health+sick` 的项目内安全 slug 为 `health_sick`。
- IR PNG 实际模式为 `RGB`，并非原生单通道；这与 schema v1 的当前转换策略一致。
- 当前 metadata 无法可靠推断 `subject_id`、`batch_id`、`scene_id`、`session_id` 和跨 clip 的真实关联。
- 每类只有一个来源 clip，远少于正式划分要求的 6 个独立组。
- 现有数据只能用于结构检查或 smoke test，不能通过随机拆帧生成正式 train/val/test。
- 配准目前只能结合 `sync_preview.png` 和信息性边缘相关指标人工复核。

当前项目内状态：

- `data/paired/` 中正式 PNG 数量为 0。
- 没有创建虚假的 `data/splits/v1/`。
- 没有对外部参考集执行训练。

## 6. 正式数据工作流

### 6.1 先执行导入 dry-run

```powershell
python scripts/import_paired_data.py `
  --source-root "C:\Users\qwhzn\Desktop\dataset\processing_workspace\output\experiment"
```

确认规划无误并明确决定复制后，才可增加：

```text
--execute-copy
```

不要由后续对话自行执行 `--execute-copy`，除非用户明确授权复制外部数据。

### 6.2 填写人工泄漏组映射

编辑：

```text
data/manifests/group_assignments.csv
```

同一实验的三个类别 clip 必须共享一个 `leakage_group_id`。同一对象、生物批次、实验容器、采集 session 或源视频跨实验出现时，也必须合并到同组。不能把类别 clip 当成独立划分单位。

### 6.3 构建 manifest

```powershell
python scripts/build_manifest.py
```

输出：

```text
data/manifests/samples.csv
data/manifests/groups.csv
```

### 6.4 创建版本化 split

```powershell
python scripts/make_split.py --version v1 --seed 42
```

如果数据不足，少于 6 个完整独立 group 时应保持失败，不得通过拆分同一实验或视频帧绕过约束。划分先生成 train pool/test，再生成 train/val；数据增加后使用 `v2`、`v3`，不要覆盖旧版本。

### 6.5 运行完整审计

`make_split.py` 会先生成基础审计；完整审计需替换该文件，因此当前推荐命令为：

```powershell
python scripts/audit_dataset.py --split-version v1 --overwrite
```

重点检查：

```text
data/splits/v1/audit.json
data/quality/registration.csv
data/quality/near_duplicates.csv
```

只有 `audit.json` 无 fatal 且覆盖约束满足，才能启动训练。

### 6.6 最小 smoke 训练

```powershell
python train.py --split-version v1 --smoke-only --max-steps 1
```

正式训练入口：

```powershell
python train.py --split-version v1
```

在正式数据和完整审计尚未就绪前，不要运行正式训练。

## 7. 剩余问题和风险

1. **核心阻塞：独立组不足。** 当前每类只有一个来源 clip，无法建立无泄漏正式 split。
2. **人工元数据缺失。** 必须由用户确认对象、批次、场景、session 与源视频的关系并填写映射。
3. **配准质量未人工确认。** 边缘相关分数仅作提示，不能替代对 `sync_preview.png` 的复核。
4. **近重复只报告不删除。** `near_duplicates.csv` 中候选需要人工判断。
5. **原生灰度/16-bit IR 尚不支持。** 未来加入时必须升级 schema/version 和转换策略。
6. **正式训练尚未验证。** 目前只完成合成数据的 CPU 单步 smoke test，不代表训练收敛或融合质量。
7. **环境未可复现声明。** 当前没有锁定 requirements；若后续要建设环境，应从实际 imports 生成并验证，不能猜测版本。
8. **没有正式 checkpoint。** 推理入口已可运行，但在正式 split、审计和训练完成前，
   只能执行明确标记的未训练 smoke，不能评价融合效果或用于下游检测。

## 8. 建议下一个对话窗口先做什么

建议先让新对话阅读本文件、`AGENTS.md`、`data/dataset.yaml` 和相关源码，然后根据用户授权选择以下方向之一：

1. 若用户准备好更多独立数据和对象/批次信息：协助填写 `group_assignments.csv`，再构建 manifest、split 和完整审计。
2. 若用户只想验证现有 36 对图片：保持标记为 smoke-only，仅做 Dataset/DataLoader/融合输出检查，不生成正式评估结论。
3. 若用户要继续完善训练：先确认正式 split 已通过审计，再检查 loss 日志、checkpoint 和融合图像导出；不要用 test 反复调参。
4. 若用户要支持原生灰度或 16-bit IR：先设计 schema v2，并明确量化、动态范围和归一化策略，不能直接混入 schema v1。

可给新对话窗口的开场指令：

> 请先阅读 `C:\Users\qwhzn\Desktop\Fusion_resnet\HANDOFF.md` 和 `AGENTS.md`，核对当前代码与交接记录是否一致。保持现有 manifest 数据边界，不修改外部数据或 `develop_list/`。先说明准备继续的具体环节，再按递增范围验证。
