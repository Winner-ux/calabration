"""绘制校准数据初训的 loss 曲线与稳定口径 checkpoint 对比。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _read_metrics(paths: list[Path]) -> list[dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                epoch = int(raw["epoch"])
                rows[epoch] = {
                    "epoch": epoch,
                    "train_loss": float(raw["train_loss"]),
                    "val_loss": float(raw["val_loss"]),
                }
    expected = list(range(10))
    if sorted(rows) != expected:
        raise ValueError(f"训练日志必须覆盖 epoch 0-9，实际={sorted(rows)}")
    return [rows[epoch] for epoch in expected]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_metrics(args.metrics)
    _configure_font()

    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    val = [row["val_loss"] for row in rows]
    blue = "#2563EB"
    orange = "#EA580C"
    gold = "#CA8A04"
    grey = "#64748B"

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.08, 1.0, 0.055))
    ax_train = fig.add_subplot(grid[0, :])
    ax_val = fig.add_subplot(grid[1, 0])
    ax_stable = fig.add_subplot(grid[1, 1])
    ax_footer = fig.add_subplot(grid[2, :])
    ax_footer.axis("off")

    fig.suptitle("Fusion ResNet 初步训练损失曲线", fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.945,
        "校准 IR/RGB 数据：train=2,250，val=600；448×448 crop，batch=2；共 10 epoch",
        ha="center",
        fontsize=11,
        color="#475569",
    )

    phase_spans = (
        (-0.4, 1.5, "#FFF7ED", "阶段 A：初始 LR=1e-4"),
        (1.5, 5.5, "#FEFCE8", "阶段 B：FP32 loss，LR=2e-5"),
        (5.5, 9.4, "#EFF6FF", "阶段 C：稳定 SSIM + SDPA，LR=1e-5"),
    )
    for left, right, color, label in phase_spans:
        ax_train.axvspan(left, right, color=color, zorder=0)
        ax_train.text(
            (left + right) / 2,
            0.96,
            label,
            transform=ax_train.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#475569",
        )
    ax_train.plot(
        epochs,
        train,
        color=blue,
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="记录的 train loss",
    )
    for x, y in zip(epochs, train, strict=True):
        ax_train.annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax_train.set_title("训练损失随 epoch 变化", loc="left", fontsize=13, fontweight="bold")
    ax_train.set_ylabel("总损失（越低越好）")
    ax_train.set_xticks(epochs)
    ax_train.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax_train.legend(frameon=False, loc="upper right")

    ax_val.plot(
        epochs[:2],
        val[:2],
        color=grey,
        marker="X",
        linestyle="--",
        linewidth=1.6,
        label="epoch 0–1：旧 SSIM 口径",
    )
    ax_val.plot(
        epochs[2:6],
        val[2:6],
        color=gold,
        marker="o",
        linewidth=2.0,
        label="epoch 2–5：中间口径",
    )
    ax_val.plot(
        epochs[6:],
        val[6:],
        color=blue,
        marker="o",
        linewidth=2.3,
        label="epoch 6–9：最终稳定口径",
    )
    ax_val.axvline(5.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax_val.axhline(0, color="#CBD5E1", linewidth=1)
    ax_val.set_title("日志中的验证损失（分口径展示）", loc="left", fontsize=13, fontweight="bold")
    ax_val.set_xlabel("epoch")
    ax_val.set_ylabel("验证总损失")
    ax_val.set_xticks(epochs)
    ax_val.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax_val.legend(frameon=False, fontsize=9, loc="upper right")
    ax_val.text(
        0.02,
        0.03,
        "注意：epoch 1 的 -0.393 是数值不稳定造成的虚低值，不能作为最佳模型依据。",
        transform=ax_val.transAxes,
        fontsize=8.5,
        color=orange,
        bbox={"facecolor": "#FFF7ED", "edgecolor": "#FDBA74", "boxstyle": "round,pad=0.35"},
    )

    stable_epochs = [1, 5, 9]
    stable_values = [0.9158844073613485, 0.2451860769589742, 0.22315935467680295]
    bars = ax_stable.bar(
        [str(epoch) for epoch in stable_epochs],
        stable_values,
        color=["#CBD5E1", "#93C5FD", blue],
        edgecolor=["#64748B", "#2563EB", "#1D4ED8"],
        linewidth=1.1,
        width=0.58,
    )
    ax_stable.bar_label(bars, labels=[f"{value:.3f}" for value in stable_values], padding=4, fontsize=10)
    ax_stable.set_title("最终稳定损失下的 checkpoint 复评", loc="left", fontsize=13, fontweight="bold")
    ax_stable.set_xlabel("checkpoint epoch")
    ax_stable.set_ylabel("稳定验证损失（600 张）")
    ax_stable.set_ylim(0, 1.05)
    ax_stable.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    improvement = (stable_values[1] - stable_values[2]) / stable_values[1]
    ax_stable.text(
        0.98,
        0.93,
        f"epoch 5 → 9 改善 {improvement:.1%}\n最终选择 epoch 9",
        transform=ax_stable.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#1D4ED8",
        bbox={"facecolor": "#EFF6FF", "edgecolor": "#93C5FD", "boxstyle": "round,pad=0.4"},
    )

    ax_footer.text(
        0.0,
        0.05,
        "来源：三个训练 run 的 metrics.csv；稳定复评统一使用修复后的 FP32 SSIM/FusionLoss。测试集未参与 checkpoint 选择。",
        fontsize=9,
        color="#64748B",
        transform=ax_footer.transAxes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
