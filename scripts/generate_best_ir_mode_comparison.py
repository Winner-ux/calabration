"""Generate matched full-resolution val comparisons for gray and learned_gray."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from data_pipeline.schema import (
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    parse_bool,
    read_csv_rows,
    resolve_data_path,
)
from main.inference_utils import (
    infer_pair_tensor,
    load_inference_pair,
    load_model_checkpoint,
    save_fused_png,
)
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion


MODES = ("gray", "learned_gray")


def _load_models(
    checkpoints: dict[str, Path], device: torch.device
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    models: dict[str, torch.nn.Module] = {}
    audits: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        model = ResNetFusion(
            Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode=mode
        )
        audits[mode] = load_model_checkpoint(model, checkpoints[mode])
        models[mode] = model.to(device).eval()
    return models, audits


def _select_ids(
    selection_path: Path, val_ids: list[str], count: int
) -> list[str]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = [str(row["sample_id"]) for row in selection["samples"]]
    selected_set = set(selected)
    selected.extend(sample_id for sample_id in val_ids if sample_id not in selected_set)
    selected = selected[:count]
    if len(selected) != count or len(set(selected)) != count:
        raise DataContractError(f"无法从 val 选择 {count} 个唯一样本")
    if not set(selected).issubset(set(val_ids)):
        raise DataContractError("固定视觉样本包含非 val sample_id")
    return selected


def _save_contact_sheet(
    output: Path, sample_rows: list[dict[str, Any]], image_root: Path
) -> None:
    thumb_width = 320
    thumb_height = 180
    label_height = 26
    columns = ("VIS", "IR gray", "gray best", "learned_gray best")
    sheet = Image.new(
        "RGB",
        (thumb_width * len(columns), (thumb_height + label_height) * len(sample_rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(sample_rows):
        sample_id = str(row["sample_id"])
        paths = (
            image_root / "sources" / "vis" / f"{sample_id}.png",
            image_root / "sources" / "ir_gray" / f"{sample_id}.png",
            image_root / "gray" / f"{sample_id}.png",
            image_root / "learned_gray" / f"{sample_id}.png",
        )
        for column_index, (label, path) in enumerate(zip(columns, paths, strict=True)):
            with Image.open(path) as image:
                preview = image.convert("RGB")
                preview.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb_width, thumb_height), "black")
            canvas.paste(
                preview,
                ((thumb_width - preview.width) // 2, (thumb_height - preview.height) // 2),
            )
            left = column_index * thumb_width
            top = row_index * (thumb_height + label_height)
            sheet.paste(canvas, (left, top))
            caption = label
            if column_index == 0:
                caption = f"{row_index + 1:02d} {row['category']} | {label}"
            draw.text((left + 4, top + thumb_height + 5), caption, fill="black")
    sheet.save(output, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gray-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    config = load_dataset_config(config_path)
    manifests = data_root / str(config["paths"]["manifests"])
    split = (
        data_root
        / str(config["paths"]["splits"])
        / args.split_version
        / "val.csv"
    )
    val_ids = [row["sample_id"] for row in read_csv_rows(split, ["sample_id"])]
    selected_ids = _select_ids(args.selection.resolve(), val_ids, args.count)
    rows_by_id = {
        row["sample_id"]: row
        for row in read_csv_rows(manifests / "samples.csv", SAMPLE_FIELDS)
    }
    selected_rows = [rows_by_id[sample_id] for sample_id in selected_ids]
    for row in selected_rows:
        if not parse_bool(row["usable"]):
            raise DataContractError(f"val 样本 usable=false: {row['sample_id']}")

    checkpoints = {
        "gray": args.gray_checkpoint.resolve(),
        "learned_gray": args.learned_checkpoint.resolve(),
    }
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and not args.no_amp
    models, checkpoint_audits = _load_models(checkpoints, device)
    inference_config = config.get("inference", {})
    factor = int(config["preprocessing"]["model_downsample_factor"])
    pad_mode = str(config["preprocessing"].get("pad_mode", "reflect"))
    tile_size = int(inference_config.get("tile_size", config["preprocessing"]["crop_size"]))
    overlap = int(inference_config.get("overlap", 64))
    max_full_tokens = int(inference_config.get("max_full_tokens", 12_544))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root.resolve() / f"best-gray-vs-learned-gray-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    image_root = run_dir / "images"
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        sample_id = str(row["sample_id"])
        vis_path = resolve_data_path(data_root, row["rgb_path"])
        ir_path = resolve_data_path(data_root, row["ir_path"])
        vis, ir = load_inference_pair(vis_path, ir_path, config)
        save_fused_png(vis, image_root / "sources" / "vis" / f"{sample_id}.png")
        save_fused_png(
            ir.expand(-1, 3, -1, -1),
            image_root / "sources" / "ir_gray" / f"{sample_id}.png",
        )
        mode_results: dict[str, Any] = {}
        for mode in MODES:
            fused, details = infer_pair_tensor(
                models[mode],
                vis,
                ir,
                device=device,
                requested_mode="auto",
                tile_size=tile_size,
                overlap=overlap,
                factor=factor,
                pad_mode=pad_mode,
                max_full_tokens=max_full_tokens,
                amp_enabled=amp_enabled,
            )
            output_path = image_root / mode / f"{sample_id}.png"
            save_fused_png(fused, output_path)
            mode_results[mode] = {"output": str(output_path), **details}
        result = {
            "index": index,
            "sample_id": sample_id,
            "category": row["category"],
            "leakage_group_id": row["leakage_group_id"],
            "vis_path": str(vis_path),
            "ir_path": str(ir_path),
            "modes": mode_results,
        }
        results.append(result)
        print(json.dumps({"completed": index, "sample_id": sample_id}, ensure_ascii=False))

    _save_contact_sheet(run_dir / "comparison_contact_sheet.png", selected_rows, image_root)
    metadata = {
        "scope": "val-only; held-out test images and metrics not accessed",
        "split_version": args.split_version,
        "sample_count": len(results),
        "checkpoints": {
            mode: {"path": str(checkpoints[mode]), **checkpoint_audits[mode]}
            for mode in MODES
        },
        "device": str(device),
        "amp": amp_enabled,
        "results": results,
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
