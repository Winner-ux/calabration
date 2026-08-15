"""Export the selected W2 fusion images and reproducible loss figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from scripts.loss_weight_experiment import latest_completed_run, weights_dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise FileExistsError(f"refusing to overwrite a different file: {destination}")
        return
    shutil.copy2(source, destination)


def _configure_matplotlib_font() -> str:
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
    return selected


def _read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({name: float(value) for name, value in raw.items()})
    if [int(row["epoch"]) for row in rows] != list(range(10)):
        raise ValueError(f"expected exactly epoch 0-9: {path}")
    return rows


def _plot_w2_losses(rows: list[dict[str, float]], output: Path) -> None:
    _configure_matplotlib_font()
    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, (ax_total, ax_parts) = plt.subplots(1, 2, figsize=(15.5, 7.4))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.17, wspace=0.16)
    fig.suptitle("W2 损失函数曲线", y=0.96, fontsize=19, fontweight="bold")
    fig.text(
        0.5,
        0.90,
        "权重 3·intensity + 10·gradient + 3·SSIM + 2·edge；gray 模式，seed 42，固定 10 epoch",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    ax_total.plot(
        epochs,
        [row["train_loss"] for row in rows],
        color="#2563EB",
        marker="o",
        linewidth=2.2,
        label="train objective loss",
    )
    ax_total.plot(
        epochs,
        [row["val_loss"] for row in rows],
        color="#EA580C",
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="val objective loss",
    )
    ax_total.set_title("W2 自身加权目标", loc="left", fontsize=13, fontweight="bold")
    ax_total.set_ylabel("加权损失（仅用于 W2 内部收敛观察）")
    ax_total.legend(frameon=False, loc="upper right")

    components = (
        ("val_intensity_loss", "intensity", "#2563EB", "o", "-"),
        ("val_gradient_loss", "gradient", "#EA580C", "s", "--"),
        ("val_ssim_loss", "SSIM", "#CA8A04", "^", "-."),
        ("val_edge_loss", "edge", "#0F766E", "D", ":"),
    )
    for field, label, color, marker, style in components:
        ax_parts.plot(
            epochs,
            [row[field] for row in rows],
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=2.0,
            markersize=5,
            label=label,
        )
    ax_parts.set_title("验证集原始分项损失（未乘权重）", loc="left", fontsize=13, fontweight="bold")
    ax_parts.set_ylabel("原始分项损失")
    ax_parts.legend(frameon=False, ncol=2, loc="upper right")

    for ax in (ax_total, ax_parts):
        ax.set_xlabel("epoch（从 1 开始显示）")
        ax.set_xticks(epochs)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.075,
        0.045,
        "数据：正式 train/val；曲线对应固定 seed 42 的 W2 训练记录。跨权重组比较请使用 loss_weight_comparison.png。",
        fontsize=9,
        color="#64748B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("msyhbd.ttc", "msyh.ttc") if bold else ("msyh.ttc", "msyhbd.ttc")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _make_preview(images: list[Path], output: Path) -> None:
    columns = 3
    tile_width, tile_height = 540, 330
    header_height, label_height, gap, margin = 105, 38, 16, 24
    rows = (len(images) + columns - 1) // columns
    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = header_height + margin + rows * (tile_height + label_height) + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    title_font = _pil_font(34, bold=True)
    subtitle_font = _pil_font(20)
    label_font = _pil_font(18)
    draw.text((margin, 20), "W2 改进后融合图像", fill="#0F172A", font=title_font)
    draw.text((margin, 66), "intensity / gradient / SSIM / edge = 3 / 10 / 3 / 2 · seed 42 · 全分辨率输出缩略预览", fill="#475569", font=subtitle_font)

    for index, path in enumerate(images):
        row, column = divmod(index, columns)
        left = margin + column * (tile_width + gap)
        top = header_height + margin + row * (tile_height + label_height + gap)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = left + (tile_width - image.width) // 2
            y = top + (tile_height - image.height) // 2
            canvas.paste(image, (x, y))
        draw.rectangle((left, top, left + tile_width, top + tile_height), outline="#CBD5E1", width=2)
        tokens = path.stem.split("__")
        category = tokens[2] if len(tokens) > 2 else "sample"
        frame = tokens[-1] if tokens else str(index + 1)
        draw.text((left + 8, top + tile_height + 7), f"{category} · frame {frame}", fill="#334155", font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("runs/loss_weight_ablation"))
    parser.add_argument("--group", choices=("W2",), default="W2")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/loss_weight_ablation/final_outputs"),
    )
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    output_root = args.output_root.resolve()
    source_dir = root / "blind_review" / "full_resolution" / args.group / f"seed{args.seed}"
    source_images = sorted(source_dir.glob("*.png"))
    if len(source_images) != 9:
        raise ValueError(f"expected 9 fixed full-resolution images, got {len(source_images)}: {source_dir}")

    fused_dir = output_root / "fused_images"
    copied_images: list[Path] = []
    for source in source_images:
        destination = fused_dir / source.name
        _copy_verified(source, destination)
        copied_images.append(destination)

    preview_path = output_root / "fusion_preview_grid.png"
    _make_preview(copied_images, preview_path)
    run_dir = latest_completed_run(root, args.group, args.seed, 10)
    loss_path = output_root / "w2_loss_curves.png"
    _plot_w2_losses(_read_metrics(run_dir / "metrics.csv"), loss_path)
    comparison_source = root / "analysis" / "loss_weight_curves.png"
    comparison_destination = output_root / "loss_weight_comparison.png"
    _copy_verified(comparison_source, comparison_destination)
    comparison_svg = comparison_source.with_suffix(".svg")
    _copy_verified(comparison_svg, comparison_destination.with_suffix(".svg"))

    checkpoint = run_dir / "last.pt"
    manifest = {
        "selected_group": args.group,
        "weights": weights_dict(args.group),
        "seed": args.seed,
        "selection_policy": "fixed preregistered representative seed; seed was not post-hoc optimized",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_split": "val",
        "test_images_or_metrics_accessed": False,
        "fusion_images": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in copied_images
        ],
        "figures": [
            {"path": str(preview_path.resolve()), "sha256": _sha256(preview_path)},
            {"path": str(loss_path.resolve()), "sha256": _sha256(loss_path)},
            {"path": str(loss_path.with_suffix('.svg').resolve()), "sha256": _sha256(loss_path.with_suffix('.svg'))},
            {"path": str(comparison_destination.resolve()), "sha256": _sha256(comparison_destination)},
            {"path": str(comparison_destination.with_suffix('.svg').resolve()), "sha256": _sha256(comparison_destination.with_suffix('.svg'))},
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
