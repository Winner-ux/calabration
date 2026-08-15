"""Plot loss-weight ablation curves using one fixed W0 reference loss."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from scripts.loss_weight_experiment import BASELINE_RUNS, latest_completed_run


REFERENCE_WEIGHTS = {
    "val_intensity_loss": 1.0,
    "val_gradient_loss": 10.0,
    "val_ssim_loss": 5.0,
    "val_edge_loss": 2.0,
}
COLORS = {
    "W0": "#334155",
    "W1": "#2563EB",
    "W2": "#EA580C",
    "W3": "#CA8A04",
    "W4": "#7C3AED",
    "W5": "#0F766E",
}
MARKERS = {"W0": "o", "W1": "s", "W2": "D", "W3": "^", "W4": "v", "W5": "P"}


def _configure_font() -> None:
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#334155",
            "text.color": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def _read_reference_curve(run_dir: Path, epochs: int) -> list[float]:
    rows: dict[int, float] = {}
    with (run_dir / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            rows[epoch] = sum(float(raw[name]) * weight for name, weight in REFERENCE_WEIGHTS.items())
    expected = list(range(epochs))
    if not all(epoch in rows for epoch in expected):
        raise ValueError(f"metrics.csv does not cover epoch 0-{epochs - 1}: {run_dir}")
    return [rows[epoch] for epoch in expected]


def _style_axis(ax: plt.Axes, epochs: int) -> None:
    ax.set_xlabel("epoch（从 1 开始显示）")
    ax.set_ylabel("W0 参考验证损失（越低越好）")
    ax.set_xticks(range(1, epochs + 1))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("runs/loss_weight_ablation"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/loss_weight_ablation/analysis/loss_weight_curves.png"),
    )
    args = parser.parse_args()
    _configure_font()

    fig, (ax_screen, ax_final) = plt.subplots(1, 2, figsize=(16, 7.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.17, wspace=0.12)
    fig.suptitle("损失权重消融：统一参考损失曲线", y=0.965, fontsize=19, fontweight="bold")
    fig.text(
        0.5,
        0.905,
        "所有曲线均按 W0 = 1·intensity + 10·gradient + 5·SSIM + 2·edge 重算；不同候选自身目标损失未作横向比较",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    for group in ("W0", "W1", "W2", "W3", "W4", "W5"):
        run_dir = BASELINE_RUNS[42] if group == "W0" else latest_completed_run(args.ablation_root, group, 42, 5)
        values = _read_reference_curve(run_dir, 5)
        ax_screen.plot(
            range(1, 6),
            values,
            color=COLORS[group],
            marker=MARKERS[group],
            linewidth=2.0 if group in {"W0", "W2"} else 1.6,
            markersize=5.5,
            label=group,
        )
    ax_screen.set_title("筛选阶段 · seed 42 · 前 5 epoch", loc="left", fontsize=13, fontweight="bold")
    _style_axis(ax_screen, 5)
    ax_screen.legend(frameon=False, ncol=3, loc="upper right")

    line_styles = {42: "-", 43: "--", 44: ":"}
    for group in ("W0", "W1", "W2"):
        for seed in (42, 43, 44):
            run_dir = BASELINE_RUNS[seed] if group == "W0" else latest_completed_run(args.ablation_root, group, seed, 10)
            values = _read_reference_curve(run_dir, 10)
            ax_final.plot(
                range(1, 11),
                values,
                color=COLORS[group],
                linestyle=line_styles[seed],
                marker=MARKERS[group] if seed == 42 else None,
                linewidth=2.2 if group == "W2" else 1.7,
                alpha=0.95 if group == "W2" else 0.8,
                label=f"{group} · seed {seed}",
            )
    ax_final.set_title("复验阶段 · 三种子 · 10 epoch", loc="left", fontsize=13, fontweight="bold")
    _style_axis(ax_final, 10)
    ax_final.legend(frameon=False, fontsize=8.4, ncol=3, loc="upper right")

    fig.text(
        0.07,
        0.045,
        "数据：正式 val（600 对图像）；模型：固定 epoch 5/10 比较；test 未参与调参。曲线仅用于收敛审计，最终选择依据独立质量指标与盲评。",
        fontsize=9,
        color="#64748B",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
