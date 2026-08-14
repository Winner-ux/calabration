# RGB/IR Calibration Tools

本模块用于可见光（RGB）与红外（IR）相机的棋盘格标定、单对图像配准、批量模型应用和结果质量检查。核心变换方向统一为 **IR → RGB**。

## 文件说明

```text
calibration_tools/
├── calibrate.py           # 从多组棋盘图像估计全局 Homography
├── registration.py        # 对单组测试图像执行配准
├── apply_registration.py  # 将已有标定模型批量应用到同名图像对
├── check.py               # 生成叠加预览与几何质量指标
├── dataset_pipeline.py    # 按 day/experiment 组织的数据集批处理入口
├── config.py              # 路径、棋盘、RANSAC、质量阈值等唯一配置源
├── input/README.md        # 批量配准输入说明
└── requirements.txt       # 独立运行依赖
```

## 安装

在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r calibration_tools/requirements.txt
Set-Location calibration_tools
```

## 数据路径

所有算法参数集中在 `config.py`。路径默认相对于本模块，也可通过环境变量覆盖：

| 环境变量 | 默认路径 | 用途 |
| --- | --- | --- |
| `RGB_IR_CALIBRATION_RGB_PATH` | `calibration/rgb` | RGB 标定图像 |
| `RGB_IR_CALIBRATION_IR_PATH` | `calibration/ir` | IR 标定图像 |
| `RGB_IR_TEST_RGB_PATH` | `test/rgb_test/001.png` | 单对 RGB 测试图 |
| `RGB_IR_TEST_IR_PATH` | `test/ir_test/001.png` | 单对 IR 测试图 |
| `RGB_IR_OUTPUT_PATH` | `output` | 模型、报告和预览输出 |
| `RGB_IR_DATASET_ROOT` | `dataset` | 数据集批处理根目录 |

RGB 与 IR 图像应使用相同的文件名主干，例如 `001.jpg` 与 `001.png`。公开仓库不会包含原始图像、模型或生成预览。

## 标准工作流

```powershell
python calibrate.py
python registration.py
python check.py
```

1. `calibrate.py` 检测成对棋盘角点并保存 `output/calibration.yaml`。
2. `registration.py` 将测试 IR 图像变换到 RGB 坐标系，并输出对齐图和报告。
3. `check.py` 生成棋盘叠加、边缘叠加与 Chamfer 距离等质量证据。

标定是否成功不能只看进程退出码，还应检查 `output/` 中的预览和误差报告。棋盘尺寸、参考分辨率、旋转方向及质量阈值应在运行前按实际设备修改 `config.py`。

## 批量应用

按照 [input/README.md](input/README.md) 放置同名图像对，然后运行：

```powershell
python apply_registration.py
```

结果写入 `output/registered/`。按实验分组的数据集可先查看：

```powershell
python dataset_pipeline.py --help
```
