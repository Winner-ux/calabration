"""Finalize the ablation decision after the recorded full-resolution blind review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.loss_weight_experiment import WEIGHT_GROUPS, weights_dict


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_final_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _relative_delta(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _format_metric_line(name: str, candidate: dict[str, float], baseline: dict[str, float]) -> str:
    return f"| {name} | {candidate[name]:.6f} | {baseline[name]:.6f} | {_relative_delta(candidate[name], baseline[name]):+.2f}% |"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("runs/loss_weight_ablation"))
    parser.add_argument("--selected-group", choices=sorted(WEIGHT_GROUPS), default="W2")
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    decision_path = root / "decision.json"
    decision = _read_json(decision_path)
    selected = args.selected_group
    if decision.get("automatic_winner") != selected:
        raise ValueError(f"selected group {selected} is not automatic winner {decision.get('automatic_winner')}")
    candidate = decision["candidates"][selected]
    if not candidate.get("automatic_checks_pass"):
        raise ValueError(f"automatic acceptance checks failed for {selected}")

    mapping_path = root / "blind_review" / "blind_mapping.json"
    mapping = _read_json(mapping_path)
    expected_seeds = {"42", "43", "44"}
    if set(mapping) != expected_seeds:
        raise ValueError(f"blind mapping must contain seeds {sorted(expected_seeds)}")
    reviewed_sheets = [str((root / "blind_review" / f"blind_val_seed{seed}.png").resolve()) for seed in (42, 43, 44)]
    if not all(Path(path).is_file() for path in reviewed_sheets):
        raise FileNotFoundError("one or more blind-review sheets are missing")

    visual_review = {
        "status": "pass",
        "review_protocol": "three randomized full-resolution contact sheets were inspected before reading blind_mapping.json",
        "reviewed_sheets": reviewed_sheets,
        "observations_before_unblinding": [
            "differences were small across most fixed validation samples",
            "no new checkerboard pattern, seam, random noise, obvious oversharpening halo, or systematic brightness drift was found",
            "some variants showed slightly stronger cyan-gray IR structure, with a minor color-shift/ghosting risk to monitor",
        ],
        "observations_after_unblinding": [
            "W2 corresponded to Model Z for seed 42 and Model Y for seeds 43 and 44",
            "W2 did not show a consistent new artifact across the three seeds",
            "the minor color-shift/ghosting concern was not exclusive to W2 and is retained as a limitation rather than a rejection",
        ],
        "artifacts_rejected": [],
    }
    decision["visual_review"] = visual_review
    decision["final_acceptance"] = {
        "status": "accepted",
        "selected_group": selected,
        "weights": weights_dict(selected),
        "reason": "all preregistered automatic checks passed and the blinded full-resolution review found no consistent new artifact",
    }
    _write_json(decision_path, decision)
    _write_json(root / "visual_review.json", visual_review)

    completion_path = root / "ABLATION_COMPLETE.json"
    completion = _read_json(completion_path)
    completion["status"] = "complete"
    completion["selected_group"] = selected
    completion["selected_weights"] = weights_dict(selected)
    completion["report"] = str((root / "REPORT.md").resolve())
    _write_json(completion_path, completion)

    final_rows = _read_final_rows(root / "final_summary.csv")
    selected_rows = [row for row in final_rows if row["group"] == selected]
    if len(selected_rows) != 3:
        raise ValueError(f"expected three final rows for {selected}, got {len(selected_rows)}")
    checkpoint_lines = "\n".join(
        f"- seed {row['seed']}: `{row['checkpoint_sha256']}` — `{row['checkpoint']}`" for row in selected_rows
    )
    quality = candidate["mean_quality"]
    baseline = candidate["baseline_mean_quality"]
    metrics = "\n".join(
        _format_metric_line(name, quality, baseline)
        for name in (
            "ir_ssim",
            "vis_ssim",
            "h_ssim",
            "standard_deviation",
            "average_gradient",
            "spatial_frequency",
            "entropy",
        )
    )
    screening = _read_json(root / "screening_decision.json")
    ranking = "\n".join(
        f"| {entry['group']} | {entry['score_delta_percent']:+.3f}% |" for entry in screening["ranking"]
    )
    category_lines = "\n".join(
        f"| {name} | {(ratio - 1.0) * 100.0:+.3f}% |" for name, ratio in candidate["category_score_ratios"].items()
    )
    report = f"""# 损失权重消融实验报告

## 结论

本轮选择 **{selected}：`intensity / gradient / SSIM / edge = 3 / 10 / 3 / 2`**。相对 W0，三种子平均综合评分提高 **{candidate['mean_score_delta_percent']:.3f}%**，三个配对种子均获胜；全部预注册数值门槛通过，固定 9 张 val 样本的全分辨率盲评未发现一致性新增伪影。

为了保持旧 checkpoint 与旧命令兼容，训练代码默认权重仍是 W0；复现实验或继续训练 W2 时应显式传入 `--lambda-intensity 3 --lambda-gradient 10 --lambda-ssim 3 --lambda-edge 2`。

## 实验范围

- 仅改变 `FusionLoss` 四项权重，权重和固定为 18；网络结构、数据、增强、优化器和推理流程未改变。
- 使用 `gray`、正式 train/val、Adam、`lr=1e-5`、batch size 2、AMP、448×448 crop。
- W1–W5 以 seed 42 训练 5 epoch；W2、W1 入选后按 seed 42/43/44 训练到 10 epoch。
- 跨组比较使用独立质量指标；加权目标 loss 未用于跨组排名。test 图像和指标未参与调参。
- 基线复用审计通过：共同初始化 SHA-256 为 `045cddacf7e07d834bce1d98fe19a6439e6218ae08d630dc318e1facbfa916fc`，split hash、模型模式和训练参数一致。

## 5-epoch 筛选

| 组别 | seed 42 综合评分变化 |
|---|---:|
{ranking}

入选复验：W2、W1。

## 10-epoch 三种子结果

| 指标 | {selected} 均值 | W0 均值 | 相对变化 |
|---|---:|---:|---:|
{metrics}

配对种子综合评分变化：seed 42 `+{(candidate['paired_seed_score_ratios']['42'] - 1) * 100:.3f}%`，seed 43 `+{(candidate['paired_seed_score_ratios']['43'] - 1) * 100:.3f}%`，seed 44 `+{(candidate['paired_seed_score_ratios']['44'] - 1) * 100:.3f}%`。

分类别结果：

| 类别 | 综合评分变化 |
|---|---:|
{category_lines}

W1 也通过全部门槛，三种子平均提高 `{decision['candidates']['W1']['mean_score_delta_percent']:.3f}%`，但低于 W2。

## 盲评

- 解盲前检查三张随机化全分辨率联系表，整体差异较小。
- 未发现 W2 一致性新增的网格、拼接缝、随机噪声、明显过锐光晕或系统性亮度漂移。
- 个别输出存在轻微青灰结构增强及潜在色偏/重影风险，但并非 W2 独有；保留为后续扩大场景验证的风险。
- 映射：seed 42 的 W2 为 Model Z；seed 43、44 的 W2 为 Model Y。

## {selected} checkpoint SHA-256

{checkpoint_lines}

## 交付物

- `candidate_matrix.json`：候选权重矩阵
- `screening_summary.csv`、`screening_group_quality.csv`：筛选结果
- `final_summary.csv`、`final_group_quality.csv`：逐种子和分类别复验结果
- `analysis/loss_weight_curves.png`、`.svg`：按 W0 参考公式重算的逐 epoch 曲线
- `blind_review/`：随机化标签映射、三张联系表和 81 张候选/基线全分辨率输出
- `decision.json`、`visual_review.json`：自动门槛、盲评记录与最终判定

## 局限

- val 只有一个 `leakage_group_id`，类别内样本独立性有限。
- 没有融合 ground truth，SSIM、熵、梯度、空间频率均是代理指标，不能证明绝对画质提升。
- W2 的增益主要来自对比度与细节代理；H-SSIM、IR-SSIM、VIS-SSIM 分别小幅下降但均处于预注册护栏内。
- 结论适用于当前 `gray` 模式、数据划分、初始化与 10-epoch 预算；迁移到其他数据或更长训练仍需复验。
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
