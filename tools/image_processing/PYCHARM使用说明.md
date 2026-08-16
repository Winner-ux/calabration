# RGB/IR 图像处理程序（PyCharm）

## 一、首次打开

1. 在 PyCharm 中打开 `Fusion_resnet/tools/image_processing` 文件夹。
2. 在 **Settings → Project → Python Interpreter** 中选择当前 Anaconda Python。
3. 如果 PyCharm 提示安装 `requirements.txt`，在界面中点击安装即可；不需要打开终端。
4. 右上角运行列表中应显示 **标定处理** 和 **实验处理**。

如果运行列表没有自动出现，也可以在项目树中右键 `calibration_app.py` 或
`experiment_app.py`，选择 **Run**。程序路径不依赖当前工作目录。

## 二、固定运行顺序

### 新标定数据

1. 运行 **标定处理**。
2. 默认源目录为项目中的 `cabration_video`；可在窗口中另选目录。
3. 点击“扫描候选配对”。只有同名数字文件会形成自动候选。
4. 选择一项，点击“预览所选 RGB/IR”，核对首帧和中间帧。
5. 点击“确认所选配对”。随机文件名必须使用“手工添加配对”。
6. 点击“导入全部已确认项”；视频会复制到
   `processing_workspace/input/calibration`，原视频不会改变。
7. 点击“校验 input”，通过后点击“开始批量处理”。
8. 在 `processing_workspace/output/calibration` 查看规范化视频、标定图像、预览和报告。

### 新实验数据

1. 先在剪辑软件中完成人工对时、方向修正，并将 RGB/IR 都导出为
   1920×1080、相同 FPS 的横屏成片。
2. 确认标定程序没有在处理，然后运行 **实验处理**。
3. 默认源目录为项目中的 `vedio`。
4. 扫描、预览，并核对 `health`、`health+sick`、`sick` 类别。
5. 确认配对后导入。尺寸或 FPS 不合格的视频不会进入 input。
6. 默认每个 clip 抽取 750 对，可在窗口中修改。
7. 校验后开始批量处理，在 `processing_workspace/output/experiment` 查看结果。

## 三、安全规则

- 标定程序和实验程序不能同时生成结果；全局锁会阻止第二个任务启动。
- 已有完整结果默认跳过。当前 day3 的 200 对结果不会因默认值为 750 而自动重做。
- 只有主动勾选“重新处理”并再次确认才会重做，旧结果保存在 `_system/backups`。
- 新结果先写到 `_system/staging`，全部校验通过后才进入正式 output。
- 导入使用完整 SHA-256 校验。目标文件同名但内容不同时，程序会停止并提示冲突。
- 不要手工修改 `_system`、`job.yaml` 或 `READY`。
- `cabration_video/day4/1` 与 `day3/2` 是已确认的重复标定源，扫描时仍会提示并记录。

## 四、输入和输出约定

- 日期目录：`dayN`，例如 `day5`。
- 实验目录：`experimentN`；源目录中的数字 `1` 会规范为 `experiment1`。
- 标准 clip 文件名：`01.mp4`、`02.mp4` 等，同编号 IR/RGB 为一对。
- 实验类别只允许 `health`、`health+sick`、`sick`。
- `test` 与 `cabration` 是历史图像目录，两个新程序不会修改它们。

## 五、常见提示

- **共同帧不足**：降低窗口中的抽帧数量，或提供更长的成片。
- **FPS 不一致**：重新导出人工对时视频，确保两边使用相同 FPS。
- **已有不完整结果**：先人工检查；确认要重做后勾选“重新处理”。
- **已有处理任务正在运行**：先等待另一个程序结束。异常退出时，确认没有 Python
  处理进程后再人工检查 `_system/locks/processing.lock`。
