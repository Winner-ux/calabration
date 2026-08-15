"""Export clear W0-W5 image and loss comparisons with W2 highlighted."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from scripts.export_improved_fusion_outputs import _configure_matplotlib_font, _pil_font
from scripts.loss_weight_experiment import BASELINE_RUNS, WEIGHT_GROUPS, latest_completed_run


GROUPS = ("W0", "W1", "W2", "W3", "W4", "W5")
WEIGHT_TEXT = {group: "/".join(str(int(value)) for value in WEIGHT_GROUPS[group]["weights"]) for group in GROUPS}
LINE_STYLE = {
    "W0": {"color": "#334155", "marker": "o", "linestyle": "-", "linewidth": 2.1},
    "W1": {"color": "#2563EB", "marker": "s", "linestyle": "--", "linewidth": 2.0},
    "W2": {"color": "#EA580C", "marker": "D", "linestyle": "-", "linewidth": 3.0},
    "W3": {"color": "#64748B", "marker": "^", "linestyle": "-.", "linewidth": 1.6},
    "W4": {"color": "#94A3B8", "marker": "v", "linestyle": ":", "linewidth": 1.8},
    "W5": {"color": "#CBD5E1", "marker": "P", "linestyle": "--", "linewidth": 2.0},
}
REFERENCE_WEIGHTS = {
    "val_intensity_loss": 1.0,
    "val_gradient_loss": 10.0,
    "val_ssim_loss": 5.0,
    "val_edge_loss": 2.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _center_crop(path: Path, size: int = 448) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        return image.crop((left, top, left + size, top + size))


def _screening_sources(root: Path, filename: str) -> list[tuple[str, Image.Image]]:
    source_root = root / "blind_review" / "full_resolution" / "sources"
    items = [
        ("IR source", _center_crop(source_root / "ir" / filename)),
        ("VIS source", _center_crop(source_root / "vis" / filename)),
    ]
    for group in GROUPS:
        path = root / "visual_outputs" / "screening" / "center_crop" / group / "seed42" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            items.append((f"{group}  {WEIGHT_TEXT[group]}", source.convert("RGB").copy()))
    return items


def _draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    label: str,
    left: int,
    top: int,
    tile_size: int,
    label_height: int,
) -> None:
    resized = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas.paste(resized, (left, top + label_height))
    best = label.startswith("W2")
    color = "#EA580C" if best else "#CBD5E1"
    width = 7 if best else 2
    draw.rectangle(
        (left, top + label_height, left + tile_size, top + label_height + tile_size),
        outline=color,
        width=width,
    )
    title = f"{label}  ★ BEST" if best else label
    font = _pil_font(19 if tile_size <= 320 else 21, bold=best)
    bbox = draw.textbbox((0, 0), title, font=font)
    x = left + (tile_size - (bbox[2] - bbox[0])) // 2
    draw.text((x, top + 5), title, fill="#EA580C" if best else "#334155", font=font)


def _make_summary(root: Path, filenames: list[str], output: Path) -> None:
    tile_size, label_height, gap, margin, header = 300, 42, 14, 26, 125
    columns, rows = 8, len(filenames)
    width = margin * 2 + columns * tile_size + (columns - 1) * gap
    height = header + margin + rows * (label_height + tile_size) + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "全部损失权重融合图像对比", fill="#0F172A", font=_pil_font(36, bold=True))
    draw.text(
        (margin, 69),
        "统一 448×448 中心区域 · seed 42 · W2（3/10/3/2）以橙色高亮；W0=epoch 10，W1–W5=epoch 5",
        fill="#475569",
        font=_pil_font(21),
    )
    for row, filename in enumerate(filenames):
        items = _screening_sources(root, filename)
        tokens = Path(filename).stem.split("__")
        if len(tokens) >= 4:
            items[0] = (f"{tokens[2]} {tokens[-1]} · IR", items[0][1])
        top = header + margin + row * (label_height + tile_size + gap)
        for column, (label, image) in enumerate(items):
            left = margin + column * (tile_size + gap)
            _draw_tile(canvas, draw, image, label, left, top, tile_size, label_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _make_sample_sheet(root: Path, filename: str, output: Path) -> None:
    tile_size, label_height, gap, margin, header = 448, 50, 18, 28, 125
    columns, rows = 4, 2
    width = margin * 2 + columns * tile_size + (columns - 1) * gap
    height = header + margin + rows * (label_height + tile_size) + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    tokens = Path(filename).stem.split("__")
    sample_label = f"{tokens[2]} · frame {tokens[-1]}" if len(tokens) >= 4 else Path(filename).stem
    draw.text((margin, 18), "单样本全部权重对比", fill="#0F172A", font=_pil_font(36, bold=True))
    draw.text(
        (margin, 69),
        f"{sample_label} · 原始 448×448 像素展示 · W2 以橙色高亮",
        fill="#475569",
        font=_pil_font(22),
    )
    for index, (label, image) in enumerate(_screening_sources(root, filename)):
        row, column = divmod(index, columns)
        left = margin + column * (tile_size + gap)
        top = header + margin + row * (label_height + tile_size + gap)
        _draw_tile(canvas, draw, image, label, left, top, tile_size, label_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _read_reference_curve(run_dir: Path, epoch_count: int) -> list[float]:
    rows: dict[int, float] = {}
    with (run_dir / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            rows[epoch] = sum(float(raw[field]) * weight for field, weight in REFERENCE_WEIGHTS.items())
    expected = list(range(epoch_count))
    if not all(epoch in rows for epoch in expected):
        raise ValueError(f"missing epochs in {run_dir / 'metrics.csv'}")
    return [rows[epoch] for epoch in expected]


def _plot_reference_curves(root: Path, output: Path) -> None:
    _configure_matplotlib_font()
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.17)
    fig.suptitle("全部权重组的统一参考损失曲线", y=0.96, fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.90,
        "统一按 W0 = 1·intensity + 10·gradient + 5·SSIM + 2·edge 重算；W2 以橙色高亮",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    for group in GROUPS:
        if group == "W0":
            run_dir, epoch_count = BASELINE_RUNS[42], 10
        elif group in {"W1", "W2"}:
            run_dir, epoch_count = latest_completed_run(root, group, 42, 10), 10
        else:
            run_dir, epoch_count = latest_completed_run(root, group, 42, 5), 5
        epochs = list(range(1, epoch_count + 1))
        label = f"{group} ({WEIGHT_TEXT[group]})" + (" · BEST" if group == "W2" else "")
        ax.plot(epochs, _read_reference_curve(run_dir, epoch_count), label=label, **LINE_STYLE[group])
    ax.axvline(5, color="#64748B", linestyle="--", linewidth=1.2)
    ax.axvspan(5, 10.15, color="#EFF6FF", alpha=0.65, zorder=0)
    ax.text(5.08, 0.99, "W3–W5 在筛选后停止", transform=ax.get_xaxis_transform(), va="top", color="#475569")
    ax.set_xlabel("epoch（从 1 开始显示）")
    ax.set_ylabel("W0 参考验证损失（越低越好）")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.text(
        0.09,
        0.05,
        "说明：参考损失只用于收敛与同公式比较；W2 的最终选择依据独立画质综合评分和盲评，而不是参考损失最低。",
        fontsize=9.5,
        color="#64748B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _read_screening_rows(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            result[raw["group"]] = {
                name: float(raw[name])
                for name in ("intensity_loss", "gradient_loss", "ssim_loss", "edge_loss", "reference_loss")
            }
    if set(result) != set(GROUPS):
        raise ValueError(f"screening summary does not contain W0-W5: {path}")
    return result


def _plot_component_losses(root: Path, output: Path) -> None:
    _configure_matplotlib_font()
    rows = _read_screening_rows(root / "screening_summary.csv")
    fields = (
        ("intensity_loss", "Intensity 原始损失", 5),
        ("gradient_loss", "Gradient 原始损失", 6),
        ("ssim_loss", "SSIM 原始损失", 5),
        ("edge_loss", "Edge 原始损失", 5),
        ("reference_loss", "W0 参考总损失", 5),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.10, hspace=0.42, wspace=0.26)
    fig.suptitle("全部权重组的固定 checkpoint 损失对比", y=0.965, fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        "正式 val（600 对）· seed 42 · W0 固定 epoch 10，W1–W5 固定 epoch 5；W2 以橙色高亮",
        ha="center",
        fontsize=10.8,
        color="#475569",
    )
    colors = ["#2563EB", "#CBD5E1", "#EA580C", "#CBD5E1", "#CBD5E1", "#CBD5E1"]
    edges = ["#1D4ED8", "#64748B", "#C2410C", "#64748B", "#64748B", "#64748B"]
    for ax, (field, title, decimals) in zip(axes.flat, fields, strict=False):
        values = [rows[group][field] for group in GROUPS]
        bars = ax.barh(GROUPS, values, color=colors, edgecolor=edges, linewidth=1.1)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.24)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, labels=[f"{value:.{decimals}f}" for value in values], padding=4, fontsize=9)
        ax.get_yticklabels()[2].set_color("#EA580C")
        ax.get_yticklabels()[2].set_fontweight("bold")
    note_ax = axes.flat[-1]
    note_ax.axis("off")
    note_ax.set_title("权重矩阵", loc="left", fontsize=12.5, fontweight="bold")
    note_ax.text(0.02, 0.84, "组别     intensity / gradient / SSIM / edge", fontsize=10.5, color="#475569")
    for index, group in enumerate(GROUPS):
        note_ax.text(
            0.04,
            0.70 - index * 0.105,
            f"{group:<3}      {WEIGHT_TEXT[group]}",
            fontsize=11.5,
            fontweight="bold" if group == "W2" else "normal",
            color="#EA580C" if group == "W2" else "#334155",
            family="monospace",
        )
    fig.text(
        0.07,
        0.035,
        "不同候选自身 objective loss 未作横向比较；这里仅展示原始分项损失和统一 W0 参考总损失。",
        fontsize=9.5,
        color="#64748B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("runs/loss_weight_ablation"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/loss_weight_ablation/all_weight_comparison"),
    )
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    output_root = args.output_root.resolve()
    image_root = root / "visual_outputs" / "screening" / "center_crop" / "W0" / "seed42"
    filenames = sorted(path.name for path in image_root.glob("*.png"))
    if len(filenames) != 9:
        raise ValueError(f"expected 9 fixed samples, got {len(filenames)}: {image_root}")
    representatives = [name for name in filenames if name.endswith("__000101.png")]
    if len(representatives) != 3:
        raise ValueError(f"expected one frame 000101 per category, got {len(representatives)}")

    summary_path = output_root / "all_weights_image_overview.png"
    _make_summary(root, representatives, summary_path)
    sheet_paths: list[Path] = []
    for filename in filenames:
        destination = output_root / "image_sheets" / f"comparison__{filename}"
        _make_sample_sheet(root, filename, destination)
        sheet_paths.append(destination)
    curve_path = output_root / "all_weights_reference_loss_curves.png"
    component_path = output_root / "all_weights_component_losses.png"
    _plot_reference_curves(root, curve_path)
    _plot_component_losses(root, component_path)

    artifacts = [
        summary_path,
        *sheet_paths,
        curve_path,
        curve_path.with_suffix(".svg"),
        component_path,
        component_path.with_suffix(".svg"),
    ]
    manifest = {
        "best_group": "W2",
        "best_weights": WEIGHT_GROUPS["W2"]["weights"],
        "comparison_groups": {group: WEIGHT_GROUPS[group]["weights"] for group in GROUPS},
        "image_comparison": {
            "seed": 42,
            "crop": "center 448x448",
            "W0_checkpoint_epoch": 9,
            "W1_to_W5_checkpoint_epoch": 4,
            "fixed_sample_count": 9,
        },
        "loss_comparison": {
            "cross_group_objective_loss_compared": False,
            "reference_formula": "1*intensity + 10*gradient + 5*ssim + 2*edge",
            "sample_count": 600,
        },
        "test_images_or_metrics_accessed": False,
        "artifacts": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in artifacts],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
