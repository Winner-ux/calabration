"""Build the auditable Markdown and canonical portable-report artifact."""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.schema import DataContractError
from scripts.batch_lr_experiment import (
    BASELINE_GROUP,
    GROUPS,
    latest_completed_run,
    read_json,
    read_metrics,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "batch_lr_optimization"


def _number(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _relative_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _source(source_id: str, label: str, path: str) -> dict[str, str]:
    return {"id": source_id, "label": label, "path": path}


def _artifact(
    root: Path,
    completion: dict[str, Any],
    preflight: dict[str, Any],
    screening: dict[str, Any],
    refinement: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    selected = str(completion["selected_group"])
    automatic_winner = quality.get("automatic_winner")
    improvement = quality.get("improvement_vs_G0") or {}
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    init_sha = str(read_json(root / "training_complete.json")["initialization_sha256"])

    metric_row = {
        "validation_loss_reduction": _number(
            improvement.get("validation_loss_reduction_percent")
        )
        / 100.0,
        "roughness_reduction": _number(improvement.get("roughness_reduction_percent"))
        / 100.0,
        "throughput_increase": _number(improvement.get("throughput_increase_percent"))
        / 100.0,
        "epoch_time_reduction": _number(
            improvement.get("epoch_time_reduction_percent")
        )
        / 100.0,
    }

    curve_rows: list[dict[str, Any]] = []
    curve_groups = [BASELINE_GROUP]
    if selected != BASELINE_GROUP:
        curve_groups.append(selected)
    for group in curve_groups:
        run_dir = latest_completed_run(
            root, group, 42, 15, initialization_sha256=init_sha
        )
        for row in read_metrics(run_dir):
            curve_rows.extend(
                (
                    {
                        "epoch": int(row["epoch"]) + 1,
                        "series": f"{group} train",
                        "loss": float(row["train_loss"]),
                    },
                    {
                        "epoch": int(row["epoch"]) + 1,
                        "series": f"{group} val",
                        "loss": float(row["val_loss"]),
                    },
                )
            )

    ranking_rows = [
        {
            "group": str(row["group"]),
            "batch_size": int(row["batch_size"]),
            "learning_rate": float(row["learning_rate"]),
            "validation_loss": float(row["final3_val_loss"]),
            "roughness": float(row["roughness"]),
            "combined_score": float(row["combined_score"]),
            "curve_eligible": bool(row["eligible_curve"]),
        }
        for row in refinement["ranking"]
    ]
    # Keep the full refinement records here. ``ranking_rows`` is deliberately
    # reduced to chart fields and therefore does not contain throughput/time.
    refinement_by_group = {
        str(row["group"]): row for row in refinement["ranking"]
    }
    memory_rows: list[dict[str, Any]] = []
    for group, row in preflight["groups"].items():
        curve = refinement_by_group.get(group)
        memory_rows.append(
            {
                "group": group,
                "batch_size": GROUPS[group].batch_size,
                "status": str(row["status"]),
                "peak_allocated_gib": _number(row.get("peak_gpu_memory_bytes"))
                / (1024**3),
                "peak_reserved_gib": _number(
                    row.get("peak_gpu_memory_reserved_bytes")
                )
                / (1024**3),
                "samples_per_second": _number(
                    curve.get("median_samples_per_second") if curve else None
                ),
                "epoch_seconds": _number(
                    curve.get("median_epoch_seconds") if curve else None
                ),
            }
        )

    quality_rows: list[dict[str, Any]] = []
    baseline_quality = quality.get("baseline_quality", {})
    if selected != BASELINE_GROUP and selected in quality.get("candidates", {}):
        selected_quality = quality["candidates"][selected]["mean_quality"]
        for field in (
            "ir_ssim",
            "vis_ssim",
            "h_ssim",
            "standard_deviation",
            "average_gradient",
            "spatial_frequency",
            "entropy",
        ):
            base = float(baseline_quality[field])
            candidate = float(selected_quality[field])
            quality_rows.append(
                {
                    "metric": field,
                    "G0": base,
                    "selected": candidate,
                    "relative_change_percent": _relative_change(candidate, base),
                }
            )

    sources = [
        _source(
            "training_metrics",
            "Batch/LR epoch metrics",
            "runs/batch_lr_optimization/training",
        ),
        _source(
            "preflight",
            "GPU real-data one-step preflight",
            "runs/batch_lr_optimization/preflight.json",
        ),
        _source(
            "decisions",
            "Screening, refinement, and quality decisions",
            "runs/batch_lr_optimization/quality_decision.json",
        ),
    ]
    status_text = (
        f"选择 **{selected}**（batch {GROUPS[selected].batch_size}, "
        f"lr `{GROUPS[selected].learning_rate:.0e}`）。"
        if selected != BASELINE_GROUP
        else "没有非对照组同时通过稳定性与质量门槛，保留 **G0**；提升按 0% 报告。"
    )
    summary_body = (
        "## 结论\n\n"
        + status_text
        + f" 验证损失降低 **{metric_row['validation_loss_reduction']:.2%}**，"
        + f"曲线粗糙度降低 **{metric_row['roughness_reduction']:.2%}**。"
        + "这些数值是损失与稳定性变化，不是准确率提升。"
    )
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Batch Size 与学习率稳定性优化实验报告",
        "description": "W2 模型 batch size / fixed learning-rate optimization on train/val only.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "improvement_card",
                "description": "Relative changes versus G0 using the registered formulas.",
                "dataset": "headline_metrics",
                "sourceId": "decisions",
                "metrics": [
                    {
                        "label": "Validation loss reduction",
                        "field": "validation_loss_reduction",
                        "format": "percent",
                        "signed": True,
                    },
                    {
                        "label": "Roughness reduction",
                        "field": "roughness_reduction",
                        "format": "percent",
                        "signed": True,
                    },
                    {
                        "label": "Throughput increase",
                        "field": "throughput_increase",
                        "format": "percent",
                        "signed": True,
                    },
                    {
                        "label": "Epoch time reduction",
                        "field": "epoch_time_reduction",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            }
        ],
        "charts": [
            {
                "id": "selected_loss_curve",
                "title": "Selected group versus G0 total loss",
                "subtitle": "Raw train and validation losses by epoch; no smoothing applied.",
                "type": "line",
                "dataset": "loss_curves",
                "sourceId": "training_metrics",
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "loss", "type": "quantitative", "label": "Total loss"},
                    "color": {"field": "series", "type": "nominal", "label": "Series"},
                },
            },
            {
                "id": "memory_chart",
                "title": "Real-data preflight peak allocated memory",
                "subtitle": "G8 reached the unsafe threshold and has no long-training curve.",
                "type": "bar",
                "dataset": "memory",
                "sourceId": "preflight",
                "encodings": {
                    "x": {"field": "group", "type": "nominal", "label": "Group"},
                    "y": {
                        "field": "peak_allocated_gib",
                        "type": "quantitative",
                        "label": "Allocated GiB",
                    },
                    "color": {"field": "status", "type": "nominal", "label": "Status"},
                },
            },
        ],
        "tables": [
            {
                "id": "ranking_table",
                "title": "15-epoch curve ranking",
                "subtitle": "Lower combined score is better; final acceptance also requires quality and blind review.",
                "dataset": "ranking",
                "sourceId": "decisions",
                "defaultSort": {"field": "combined_score", "direction": "asc"},
                "columns": [
                    {"field": "group", "label": "Group", "type": "text"},
                    {"field": "batch_size", "label": "Batch", "format": "number"},
                    {"field": "learning_rate", "label": "LR", "format": "number"},
                    {"field": "validation_loss", "label": "Q", "format": "number"},
                    {"field": "roughness", "label": "R", "format": "number"},
                    {"field": "combined_score", "label": "Score", "format": "number"},
                    {"field": "curve_eligible", "label": "Curve gate", "type": "text"},
                ],
            },
            {
                "id": "quality_table",
                "title": "Fusion quality proxy changes",
                "subtitle": "Validation-only metrics; no fusion ground truth exists.",
                "dataset": "quality",
                "sourceId": "decisions",
                "columns": [
                    {"field": "metric", "label": "Metric", "type": "text"},
                    {"field": "G0", "label": "G0", "format": "number"},
                    {"field": "selected", "label": "Selected", "format": "number"},
                    {
                        "field": "relative_change_percent",
                        "label": "Change (%)",
                        "format": "number",
                        "movement": True,
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Batch Size 与学习率稳定性优化实验报告",
            },
            {
                "id": "summary",
                "type": "markdown",
                "body": summary_body,
                "sourceId": "decisions",
            },
            {"id": "headline", "type": "metric-strip", "cardIds": ["improvement_card"]},
            {
                "id": "curve_explanation",
                "type": "markdown",
                "body": "## 收敛与稳定性\n\n下图展示原始 train/val epoch 点。静态交付物另含全部组总图、小图与只用于观察的 EMA；排名始终使用原始曲线。",
                "sourceId": "training_metrics",
            },
            {"id": "curve", "type": "chart", "chartId": "selected_loss_curve", "layout": "full"},
            {
                "id": "memory_explanation",
                "type": "markdown",
                "body": "## 显存与可行性\n\n预检使用真实数据执行完整 forward/backward/Adam step。G8 的物理 batch 32 allocated 达到总显存 100%，按 95% 规则标记 unsafe，因此没有损失曲线且不参与排名。",
                "sourceId": "preflight",
            },
            {"id": "memory", "type": "chart", "chartId": "memory_chart", "layout": "full"},
            {
                "id": "ranking_explanation",
                "type": "markdown",
                "body": "## 15-epoch 排名\n\n综合分为 `0.5×Q/Q_G0 + 0.5×R/R_G0`；更低更好。该表只判断曲线，最终还需质量代理、分类别 guardrail 与盲审。",
                "sourceId": "decisions",
            },
            {"id": "ranking", "type": "table", "tableId": "ranking_table", "layout": "full"},
            {
                "id": "quality_explanation",
                "type": "markdown",
                "body": "## 融合质量复核\n\n质量指标来自 val 全集，test 未用于选择。指标是无监督融合代理，不能等同于准确率或绝对视觉质量。",
                "sourceId": "decisions",
            },
            {"id": "quality", "type": "table", "tableId": "quality_table", "layout": "full"},
            {
                "id": "method",
                "type": "markdown",
                "body": "## 方法与复现\n\n固定 W2 损失权重 `(3,10,3,2)`、gray IR、448×448 crop、AMP、数据划分与增强，只改变物理 batch size 和固定 LR。所有组由同一初始化 checkpoint 加载模型权重并重置 Adam/GradScaler；初始化 SHA-256 为 `" + init_sha + "`。",
                "sourceId": "decisions",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## 局限\n\n- val 仅包含一个 leakage group，独立性有限。\n- 没有融合 ground truth，质量指标均为代理。\n- G0 在本轮续训只使用 seed 42；赢家使用 seed 42/43/44。\n- 固定 epoch 比较意味着大 batch 每轮 optimizer step 更少，这是本次物理 batch 实验定义的一部分。",
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": [metric_row],
                "loss_curves": curve_rows,
                "memory": memory_rows,
                "ranking": ranking_rows,
                "quality": quality_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }


def _markdown(
    completion: dict[str, Any],
    refinement: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    selected = str(completion["selected_group"])
    spec = GROUPS[selected]
    improvement = quality.get("improvement_vs_G0") or {}
    loss_improvement = _number(improvement.get("validation_loss_reduction_percent"))
    roughness_improvement = _number(improvement.get("roughness_reduction_percent"))
    throughput = _number(improvement.get("throughput_increase_percent"))
    epoch_time = _number(improvement.get("epoch_time_reduction_percent"))
    ranking_lines = "\n".join(
        f"| {row['group']} | {row['batch_size']} | `{float(row['learning_rate']):.0e}` | "
        f"{float(row['final3_val_loss']):.6f} | {float(row['roughness']):.6f} | "
        f"{float(row['combined_score']):.4f} | {row['eligible_curve']} |"
        for row in refinement["ranking"]
    )
    conclusion = (
        f"选择 **{selected}（batch {spec.batch_size}, lr `{spec.learning_rate:.0e}`）**。"
        if selected != BASELINE_GROUP
        else "没有非对照组通过全部门槛，保留 **G0（batch 2, lr `1e-5`）**。"
    )
    return f"""# Batch Size 与学习率稳定性优化实验报告

## 结论

{conclusion} 相对 G0，验证损失降低 **{loss_improvement:.3f}%**，曲线粗糙度降低 **{roughness_improvement:.3f}%**，训练吞吐变化 **{throughput:+.3f}%**，每 epoch 时间降低 **{epoch_time:.3f}%**。这些是损失、稳定性与效率变化，不是准确率提升。

## 15-epoch 曲线排名

| Group | Batch | LR | Q（末三轮 val 均值） | R | 综合分 | 曲线门槛 |
|---|---:|---:|---:|---:|---:|:---:|
{ranking_lines}

## 静态图表

- `plots/all_groups_total_loss.png` / `.svg`
- `plots/all_groups_small_multiples.png` / `.svg`
- `plots/component_loss_grid.png` / `.svg`
- `plots/best_vs_baseline.png` / `.svg`
- `plots/memory_throughput_comparison.png` / `.svg`

## 质量与盲审

- 自动赢家：`{quality.get('automatic_winner')}`；最终状态：`{quality.get('final_acceptance', {}).get('status')}`。
- test 图像与指标未用于调参：`{not bool(quality.get('test_images_or_metrics_accessed'))}`。
- 详细三 seed 均值、标准差、最差结果与分类别门槛见 `quality_decision.json` 和 `quality_summary.csv`。

## 限制

- val 只有一个 `leakage_group_id`，独立性有限。
- 没有融合 ground truth，所有画质指标均是代理指标。
- G0 在本轮续训只有 seed 42，赢家才进行 seed 42/43/44 复核。
- 物理 batch 32 在真实单步预检中触及 unsafe 显存阈值，无损失曲线、不参与排名。
"""


def _portable_html(
    root: Path,
    completion: dict[str, Any],
    refinement: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    selected = str(completion["selected_group"])
    final_status = str(completion["status"])
    improvement = quality.get("improvement_vs_G0") or {}
    headline = {
        "Validation loss reduction": _number(
            improvement.get("validation_loss_reduction_percent")
        ),
        "Roughness reduction": _number(
            improvement.get("roughness_reduction_percent")
        ),
        "Throughput increase": _number(improvement.get("throughput_increase_percent")),
        "Epoch time reduction": _number(
            improvement.get("epoch_time_reduction_percent")
        ),
    }
    ranking = refinement["ranking"]
    baseline = next(row for row in ranking if row["group"] == BASELINE_GROUP)
    curve_leader = min(
        (row for row in ranking if row["group"] != BASELINE_GROUP),
        key=lambda row: float(row["combined_score"]),
    )
    candidate_name = str(curve_leader["group"])
    candidate = quality.get("candidates", {}).get(candidate_name, {})
    candidate_checks = candidate.get("mean_checks", {})
    failed_checks = [name for name, passed in candidate_checks.items() if not passed]
    curve_loss_gain = (
        (float(baseline["final3_val_loss"]) - float(curve_leader["final3_val_loss"]))
        / float(baseline["final3_val_loss"])
        * 100.0
    )
    curve_roughness_gain = (
        (float(baseline["roughness"]) - float(curve_leader["roughness"]))
        / float(baseline["roughness"])
        * 100.0
    )
    metric_cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value:+.3f}%</strong></div>'
        for label, value in headline.items()
    )
    ranking_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['group']))}</td>"
        f"<td>{int(row['batch_size'])}</td>"
        f"<td>{float(row['learning_rate']):.0e}</td>"
        f"<td>{float(row['final3_val_loss']):.6f}</td>"
        f"<td>{float(row['roughness']):.6f}</td>"
        f"<td>{float(row['combined_score']):.4f}</td>"
        f"<td>{html.escape(str(row['eligible_curve']))}</td>"
        "</tr>"
        for row in ranking
    )
    figures = []
    for name, caption in (
        ("all_groups_total_loss.png", "All groups: raw training and validation loss"),
        ("all_groups_small_multiples.png", "Per-group loss curves and G8 unsafe status"),
        ("component_loss_grid.png", "Validation loss components"),
        ("best_vs_baseline.png", "Best curve candidate versus retained G0"),
        ("memory_throughput_comparison.png", "GPU memory safety and throughput"),
    ):
        path = root / "figures" / name
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(
            f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    conclusion = (
        "No non-baseline candidate passed every automatic curve and quality guardrail; "
        "G0 is retained and the registered final improvement is 0%."
        if selected == BASELINE_GROUP
        else f"{selected} passed all automatic and visual-review gates."
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Batch Size 与学习率稳定性优化实验报告</title>
<style>
:root {{ color-scheme: light; --ink:#0f172a; --muted:#475569; --line:#cbd5e1; --accent:#2563eb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,"Segoe UI",Arial,sans-serif; color:var(--ink); background:#f8fafc; }}
main {{ max-width:1180px; margin:auto; padding:32px 24px 64px; }} h1 {{ margin:0 0 8px; }} h2 {{ margin-top:36px; }}
.lede {{ color:var(--muted); line-height:1.6; }} .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0; }}
.card {{ background:white; border:1px solid var(--line); border-radius:12px; padding:16px; display:grid; gap:8px; }} .card span {{ color:var(--muted); }} .card strong {{ font-size:1.45rem; }}
.callout {{ background:#fff7ed; border-left:5px solid #ea580c; padding:16px 18px; line-height:1.65; }}
table {{ width:100%; border-collapse:collapse; background:white; font-variant-numeric:tabular-nums; }} th,td {{ border-bottom:1px solid var(--line); padding:10px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
figure {{ margin:26px 0; background:white; border:1px solid var(--line); border-radius:12px; padding:12px; }} img {{ width:100%; height:auto; display:block; }} figcaption {{ color:var(--muted); padding:8px 4px 2px; }}
code {{ background:#e2e8f0; padding:2px 5px; border-radius:4px; }}
</style></head><body><main>
<h1>Batch Size 与学习率稳定性优化实验报告</h1>
<p class="lede">Final status: <code>{html.escape(final_status)}</code>; selected group: <code>{html.escape(selected)}</code>. Train/val only; test was not accessed.</p>
<h2>最终结论</h2><p>{html.escape(conclusion)}</p><div class="cards">{metric_cards}</div>
<div class="callout"><strong>曲线最优候选不等于最终可接受参数。</strong> {html.escape(candidate_name)} 在 seed 42 的 15-epoch 曲线比较中，验证损失降低 {curve_loss_gain:.3f}%，粗糙度降低 {curve_roughness_gain:.3f}%；但三 seed 质量复核失败，未通过：<code>{html.escape(', '.join(failed_checks) or 'seed pass-count gate')}</code>。因此默认配置保持不变。这些数值不是准确率提升。</div>
<h2>15-epoch 曲线排名</h2><table><thead><tr><th>Group</th><th>Batch</th><th>LR</th><th>Q</th><th>R</th><th>Score</th><th>Curve gate</th></tr></thead><tbody>{ranking_rows}</tbody></table>
<h2>图表</h2>{''.join(figures)}
<h2>限制</h2><ul><li>没有融合 ground truth，质量指标均为代理。</li><li>验证集仅含一个 leakage group，独立性有限。</li><li>G0 只有 seed 42；候选 G1/G3 使用 seed 42/43/44。</li><li>物理 batch 32 在真实预检中达到 unsafe 显存阈值，未进行长训练。</li></ul>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    required = {
        "completion": root / "EXPERIMENT_COMPLETE.json",
        "preflight": root / "preflight.json",
        "screening": root / "screening_decision.json",
        "refinement": root / "refinement_decision.json",
        "quality": root / "quality_decision.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise DataContractError(f"report inputs missing: {missing}")
    values = {name: read_json(path) for name, path in required.items()}
    artifact = _artifact(root=root, **values)
    write_json(root / "artifact.json", artifact)
    (root / "REPORT.md").write_text(
        _markdown(values["completion"], values["refinement"], values["quality"]),
        encoding="utf-8",
    )
    (root / "report.html").write_text(
        _portable_html(
            root,
            values["completion"],
            values["refinement"],
            values["quality"],
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(root / "artifact.json"),
                "markdown": str(root / "REPORT.md"),
                "html": str(root / "report.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
