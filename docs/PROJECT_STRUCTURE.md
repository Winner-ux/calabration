# Fusion ResNet 项目结构规范

本文档规定 Fusion ResNet 仓库中代码、配置、文档、数据和运行产物的保存位置。后续新增或修改内容应遵循本规范，避免在根目录随意增加脚本、实验结果或临时文件。

## 1. 项目边界

本仓库只负责已经完成标定和配准的 IR/RGB 图像融合流程：

1. 导入并审计成对图像。
2. 生成可复现的数据划分。
3. 训练、恢复和选择模型。
4. 执行推理、评估和消融实验。
5. 保存 checkpoint、日志、图像和实验报告。

图像标定、几何配准和原始视频预处理不属于本仓库的核心职责。正式标定实现位于独立项目：

```text
C:\Users\qwhzn\Desktop\RGB_IR_Calibration
```

两个项目之间只通过“已配准 IR/RGB 图像、标定报告和可核验元数据”交接，不允许 Fusion ResNet 的训练或推理代码直接导入外部标定项目的 Python 模块。

## 2. 当前受支持的目录结构

```text
Fusion_resnet/
├── README.md
├── AGENTS.md
├── .gitignore
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   └── handoff/
│       └── archive/
├── model/
├── main/
├── data_pipeline/
├── scripts/
├── tests/
├── tools/
│   └── image_processing/
├── develop_list/
├── datasets/
└── runs/
```

`main/`、`model/` 和 `data_pipeline/` 是当前代码已经使用的模块路径。除非某次任务明确允许同步更新 import、命令、测试和文档，否则不得只移动或重命名这些目录及其模块。

## 3. 各目录职责

### `model/`

只保存网络结构及与网络结构直接相关的模块，例如编码器、注意力、融合块和解码器。

- 不保存训练循环、数据读取、命令行解析或实验结果。
- 新的 Python 文件使用 `snake_case.py`。
- 模型输入输出形状必须写在模块或类的 docstring 中。
- 修改参数名、子模块属性名或层级前，必须评估 `state_dict` key 兼容性。

当前稳定张量契约：

```text
VIS/RGB: FloatTensor[B, 3, H, W]
IR:      FloatTensor[B, 1, H, W]
Output:  FloatTensor[B, 3, H, W]
Range:   [0, 1]
```

### `main/`

保存当前正式训练、推理、损失、质量评估和 checkpoint 使用流程。该名称目前为兼容路径，不代表允许继续向其中堆放任意代码。

- 正式训练入口保存在 `main/train.py`。
- 正式推理入口保存在 `main/inference.py`。
- 多个入口共享的逻辑应进入职责明确的模块，避免继续扩大通用 `utils` 文件。
- 新实验不得直接写入正式训练或推理流程；只有验证后准备成为稳定功能时才可合并。

### `data_pipeline/`

保存数据契约、manifest、划分、采样、变换和审计逻辑。

- 该包只通过 manifest 访问正式样本。
- 不根据目录名称隐式推断训练、验证或测试划分。
- 不在此目录保存真实图像、缓存、CSV 输出或审计产物。
- 数据格式变化必须同步记录 schema 版本和兼容策略。

### `scripts/`

保存可从命令行运行的项目维护与实验入口。

- 禁止在仓库根目录新增临时 `.py` 脚本。
- 脚本应优先调用 `main`、`model` 或 `data_pipeline` 中的实现。
- 新脚本名称必须包含明确动作和对象，例如 `audit_dataset.py`、`plot_training_curves.py`。
- 一次性诊断脚本验证完毕后应删除，或在确有复现价值时整理为测试或正式实验脚本。
- 同一主题出现三个以上脚本时，应在允许同步修改 import 和命令的重构任务中迁入 `experiments/<topic>/`，不能继续无限扩张平铺目录。

当前两组实验主题为：

```text
IR 灰度模式消融: ir_gray_ablation
损失权重消融:    loss_weight_ablation
```

### `tests/`

保存自动化测试，不保存正式数据集或大型图像。

- 单元测试文件使用 `test_<module>.py`。
- 跨模块训练或推理检查属于集成测试。
- 测试资源必须体积小、可重建且不含用户数据。
- 验证顺序为：语法和 import、目标单元测试、CPU 合成前向、完整测试；只有任务要求时才运行训练。

### `tools/`

保存不属于模型核心运行链路的独立工具。

`tools/image_processing/` 是保留的旧视频/图像预处理工具，当前训练、推理和数据管线不得依赖它。后续确认其功能已由 `RGB_IR_Calibration` 完整接管后，应在单独任务中迁出本仓库；迁移前不要在其中新增模型功能。

### `docs/`

保存长期有效的项目说明。

推荐布局：

```text
docs/
├── PROJECT_STRUCTURE.md
├── architecture.md
├── data_contract.md
├── guides/
├── experiments/
├── planning/
└── handoff/
    ├── current.md
    └── archive/
```

- `handoff/current.md` 只描述当前有效状态。
- 过期交接文档移入 `handoff/archive/`，文件名使用 `YYYY-MM-DD_topic.md`。
- 实验结论进入 `docs/experiments/`，不得只存在于聊天记录或 `runs/` 中。
- `develop_list/` 是当前保留的历史规划目录，不视为可执行事实；新增规划文档应放入 `docs/planning/`。

### `datasets/`

保存用户数据和数据集版本，不纳入 Git。

```text
datasets/<dataset_version>/
├── dataset.yaml
├── paired/
├── manifests/
├── splits/
└── quality/
```

约束：

- `paired/` 保存正式 IR/RGB 图像。
- `manifests/` 保存样本和分组索引。
- `splits/` 保存版本化且不可静默覆盖的数据划分。
- `quality/` 保存审计和人工复核信息。
- 数据集、manifest、split 或质量报告不得因代码整理而删除、覆盖或重生成。

当前正式默认数据根目录已经统一为 `datasets/calibrated_v1`，训练、推理和数据维护脚本默认读取同目录下的 `dataset.yaml`。需要切换数据集版本时，应同时显式传入对应的数据根目录和配置：

```powershell
python -m main.train `
  --data-root datasets/calibrated_v1 `
  --config datasets/calibrated_v1/dataset.yaml `
  --split-version calibrated_v1

python -m main.inference `
  --data-root datasets/calibrated_v1 `
  --config datasets/calibrated_v1/dataset.yaml `
  <其他必要参数>
```

不得为了兼容历史命令重新复制一份正式数据到 `data/`。测试 fixture 应从当前正式数据集读取配置模板，再在临时目录内生成合成数据。

### `runs/`

保存全部生成产物，不纳入 Git。新运行建议使用以下分类：

```text
runs/
├── training/
├── inference/
├── evaluations/
└── ablations/
    ├── ir_gray/
    └── loss_weights/
```

每次运行使用独立目录，不覆盖旧结果。目录名应至少包含任务类型和时间戳；多随机种子实验还应包含 seed。

## 4. 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| Python 包、模块 | `snake_case` | `fusion_network.py` |
| 函数、变量、参数 | `snake_case` | `load_checkpoint` |
| 类 | `PascalCase` | `FusionResNet` |
| 常量 | `UPPER_SNAKE_CASE` | `DEFAULT_LOSS_WEIGHTS` |
| CLI 参数 | `--kebab-case` | `--split-version` |
| 数据集版本 | 小写名称加版本 | `calibrated_v1` |
| 实验主题 | `snake_case` | `loss_weight_ablation` |
| 文档 | 小写 `snake_case.md`；固定规范文档可大写 | `training_guide.md` |
| 运行目录 | 任务、时间戳、必要标识 | `train_20260816_153000_seed42` |

现存的 `model/Encoder.py`、`ResNetFusion` 等名称属于兼容项。不能只为统一风格而直接改名；正式改名时必须保留兼容导出或明确声明 checkpoint/API 影响。

## 5. 新文件保存决策

新增文件前依次判断：

1. 是否是网络层或模型结构？放入 `model/`。
2. 是否定义数据读取、变换、划分或审计？放入 `data_pipeline/`。
3. 是否属于稳定训练、推理或评估流程？放入 `main/`。
4. 是否是特定假设或超参数实验？放入实验代码区域，并由 `scripts/` 提供入口。
5. 是否是自动验证？放入 `tests/`。
6. 是否是说明、结论、计划或交接？放入 `docs/` 对应子目录。
7. 是否是外部预处理工具？优先放入独立项目；临时保留时只能进入 `tools/`。
8. 是否是数据或生成结果？分别放入 `datasets/` 或 `runs/`。

无法归类时，不得直接在根目录创建文件，应先更新本规范或明确新的模块边界。

## 6. 禁止事项

- 禁止在根目录新增训练、推理、评估或临时 Python 脚本。
- 禁止把 checkpoint、融合图、日志或实验 CSV 保存到源码包。
- 禁止复制外部标定项目代码回本仓库形成第二份实现。
- 禁止同时维护两种互相竞争的 import 方式。
- 禁止为了整理目录而修改数据集、checkpoint 或生成结果。
- 禁止未验证就移动 `main/`、`model/`、`data_pipeline/` 或 `scripts/` 中的模块。
- 禁止覆盖已有 split 和正式运行目录。

## 7. 结构变更流程

任何涉及模块移动或重命名的任务必须：

1. 先搜索全部 import、命令、测试和文档引用。
2. 记录旧路径到新路径的映射。
3. 尽可能保留兼容导出，尤其是模型公共类名。
4. 不修改网络成员属性和 `state_dict` key，除非明确接受 checkpoint 不兼容。
5. 按递增范围执行语法、模块导入、CPU 合成前向、目标测试和完整测试。
6. 更新本文档和当前交接文档。
7. 单独提交结构变更，避免与网络结构或损失算法修改混在同一提交中。

## 8. 提交前检查清单

- [ ] 新文件位于职责明确的目录。
- [ ] 根目录没有新增临时脚本或输出文件。
- [ ] Python 名称符合统一规范，或已说明兼容原因。
- [ ] import 使用统一的包路径。
- [ ] 数据、checkpoint 和运行结果未被意外修改。
- [ ] 张量输入输出形状保持不变，或变化已明确记录。
- [ ] checkpoint 兼容性已检查。
- [ ] 已执行与改动范围相称的测试。
- [ ] 文档中的命令和路径与当前代码一致。
