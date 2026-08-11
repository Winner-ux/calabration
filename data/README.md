# Fusion ResNet 数据目录

本目录是项目内唯一的正式配对图像入口。`PairedFusionDataset` 只读取
manifest，不会递归扫描目录来决定训练样本。

## 目录约定

```text
data/
├─ dataset.yaml
├─ paired/                  # 唯一正式图片层，不按 split 复制图片
├─ manifests/
│  ├─ group_assignments.csv # 人工维护的泄漏组映射
│  ├─ samples.csv           # build_manifest.py 生成
│  └─ groups.csv            # build_manifest.py 生成
├─ splits/<version>/        # make_split.py 生成，不覆盖旧版本
├─ quality/                 # audit_dataset.py 生成
└─ cache/                   # 可删除并重新生成，不能作为数据源
```

规范化路径使用零填充 ID 和安全类别 slug，例如：

```text
paired/day003/experiment001/health_sick/clip001/{ir,rgb}
```

当前外部参考数据不会自动复制到这里。可先运行导入工具的 dry-run：

```powershell
python scripts/import_paired_data.py `
  --source-root "C:\Users\qwhzn\Desktop\dataset\processing_workspace\output\experiment"
```

只有显式增加 `--execute-copy` 才会复制 PNG 与 clip 元数据。导入工具不会复制
MP4，也不会覆盖内容不同的已有文件。

## 工作流

1. 将数据放入 `paired/`（人工放置或显式执行导入工具）。
2. 填写 `manifests/group_assignments.csv`。同一实验的三个类别 clip 默认使用同一
   `leakage_group_id`；同一对象、批次、容器、session 或源视频跨实验重复时，
   对应实验也必须合并到同一个 group。
3. 构建 manifest：

   ```powershell
   python scripts/build_manifest.py
   ```

4. 生成不可覆盖的版本化 split：

   ```powershell
   python scripts/make_split.py --version v1 --seed 42
   ```

5. 运行审计：

   ```powershell
   python scripts/audit_dataset.py --split-version v1
   ```

正式实验必须包含 `health`、`health_sick`、`sick`，每类恰好一个 clip。少于
`dataset.yaml` 配置的 6 个完整独立组时，正式划分会失败。禁止通过拆分同一
实验或视频的帧绕过这一约束。

每个 split 版本包含：

```text
group_assignments.csv
experiment_assignments.csv
train_pool.csv
train.csv
val.csv
test.csv
split_config.json
audit.json
```

外层按 group 生成 `80% train_pool / 20% test`，内层只在 `train_pool` 中生成
`80% train / 20% val`。`train_pool.csv` 必须严格等于 `train.csv ∪ val.csv`。
这些文件一旦生成即冻结；每个 epoch 只打乱 `train.csv` 的读取顺序，不重新划分。
