"""在正式 test split 上评分，并导出多样化的全分辨率融合结果。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data_pipeline.manifest import phash_distance
from data_pipeline.schema import (
    SAMPLE_FIELDS,
    DataContractError,
    canonical_rows_hash,
    load_dataset_config,
    read_csv_rows,
)
from data_pipeline.transforms import build_paired_transform
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion

from .dataset import PairedFusionDataset
from .inference import collect_split_input_pairs
from .inference_utils import (
    infer_pair_tensor,
    load_inference_pair,
    load_model_checkpoint,
    resolve_checkpoint_ir_mode,
    save_fused_png,
    sha256_file,
)
from .loss import FusionLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tensor_phash(tensor: torch.Tensor, hash_size: int = 8, factor: int = 4) -> str:
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    gray = (
        image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
    )[None, None]
    size = hash_size * factor
    pixels = (
        torch_functional.interpolate(
            gray, size=(size, size), mode="bicubic", align_corners=False
        )[0, 0]
        .numpy()
        .astype(np.float64)
    )
    indices = np.arange(size, dtype=np.float64)
    basis = np.cos(
        np.pi * (2 * indices[:, None] + 1) * indices[None, :] / (2 * size)
    )
    basis[:, 0] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / size)
    low = (basis.T @ pixels @ basis)[:hash_size, :hash_size]
    median = float(np.median(low.reshape(-1)[1:]))
    value = 0
    for bit in (low > median).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{(hash_size * hash_size + 3) // 4}x}"


def _make_run_directory(root: Path) -> Path:
    run_dir = root / f"best20-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _score_test_split(
    model: torch.nn.Module,
    criterion: FusionLoss,
    dataset: PairedFusionDataset,
    rows_by_id: dict[str, dict[str, str]],
    pair_by_id: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    workers: int,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    scores: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            sample_id = batch["sample_id"][0]
            vis = batch["vis"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused = model(vis, ir)
            loss, components = criterion(fused.float(), ir.float(), vis.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"测试评分出现非有限 loss: {sample_id}")
            pair = pair_by_id[sample_id]
            manifest_row = rows_by_id[sample_id]
            scores.append(
                {
                    "sample_id": sample_id,
                    "loss": float(loss),
                    **{name: float(value) for name, value in components.items()},
                    "fused_phash": _tensor_phash(fused),
                    "category": pair.category,
                    "leakage_group_id": pair.leakage_group_id,
                    "clip_key": pair.relative_path.parent.as_posix(),
                    "frame_sequence": int(manifest_row["frame_sequence"]),
                    "source_frame_index_0_based": int(
                        manifest_row.get("source_frame_index_0_based") or manifest_row["frame_sequence"]
                    ),
                }
            )
            if index % 100 == 0:
                print(json.dumps({"scored": index, "total": len(dataset)}, ensure_ascii=False))
    return scores


def _is_diverse(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    require_phash: bool,
    minimum_frame_gap: int = 5,
) -> bool:
    for existing in selected:
        if existing["clip_key"] != candidate["clip_key"]:
            continue
        if (
            abs(
                existing["source_frame_index_0_based"]
                - candidate["source_frame_index_0_based"]
            )
            < minimum_frame_gap
        ):
            return False
        if require_phash and phash_distance(
            existing["fused_phash"], candidate["fused_phash"]
        ) <= 5:
            return False
    return True


def _take_best(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    count: int,
    reason: str,
) -> None:
    for strictness, label in ((True, "phash_and_frame_gap"), (False, "frame_gap")):
        for candidate in candidates:
            if len([row for row in selected if row.get("selection_group") == reason]) >= count:
                return
            if candidate in selected or not _is_diverse(
                candidate, selected, require_phash=strictness
            ):
                continue
            candidate["selection_group"] = reason
            candidate["diversity_rule"] = label
            selected.append(candidate)
    for candidate in candidates:
        if len([row for row in selected if row.get("selection_group") == reason]) >= count:
            return
        if candidate not in selected:
            candidate["selection_group"] = reason
            candidate["diversity_rule"] = "score_only_fallback"
            selected.append(candidate)


def select_diverse_best(scores: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        by_clip[score["clip_key"]].append(score)
    if top_k < len(by_clip) * 3:
        raise DataContractError("top_k 不足以满足每个测试 clip 至少 3 张")
    selected: list[dict[str, Any]] = []
    for clip_key in sorted(by_clip):
        candidates = sorted(by_clip[clip_key], key=lambda row: (row["loss"], row["sample_id"]))
        _take_best(candidates, selected, 3, f"clip_quota:{clip_key}")
    remaining = top_k - len(selected)
    if remaining:
        _take_best(
            sorted(scores, key=lambda row: (row["loss"], row["sample_id"])),
            selected,
            remaining,
            "global_bonus",
        )
    if len(selected) != top_k:
        raise DataContractError(f"无法选择 {top_k} 张多样化结果，实际={len(selected)}")
    selected.sort(key=lambda row: (row["loss"], row["sample_id"]))
    return selected


def _write_contact_sheet(outputs: list[dict[str, Any]], path: Path) -> None:
    thumb = (320, 180)
    label_height = 34
    columns = 4
    rows = (len(outputs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(outputs):
        with Image.open(row["output_path"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail(thumb, Image.Resampling.LANCZOS)
            left = (index % columns) * thumb[0]
            top = (index // columns) * (thumb[1] + label_height)
            sheet.paste(preview, (left, top))
        draw.text(
            (left + 4, top + thumb[1] + 2),
            f"#{index + 1:02d} {row['category']} loss={row['loss']:.4f}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def run_selection(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    config_path = (args.config or data_root / "dataset.yaml").resolve()
    config = load_dataset_config(config_path)
    pairs, split_source = collect_split_input_pairs(
        data_root, config, args.split_version, "test"
    )
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    manifest_path = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    manifest_rows = read_csv_rows(manifest_path, SAMPLE_FIELDS)
    rows_by_id = {row["sample_id"]: row for row in manifest_rows}
    split_dir = data_root / str(config["paths"]["splits"]) / args.split_version
    dataset = PairedFusionDataset(
        data_root=data_root,
        samples_manifest=manifest_path,
        split_manifest=split_dir / "test.csv",
        paired_transform=build_paired_transform(config, "test"),
        config_path=config_path,
        base_seed=args.seed,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ir_mode = resolve_checkpoint_ir_mode(
        args.checkpoint, getattr(args, "ir_mode", "auto")
    )
    model = ResNetFusion(
        Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode=ir_mode
    )
    checkpoint_metadata = load_model_checkpoint(model, args.checkpoint)
    model.to(device)
    criterion = FusionLoss().to(device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scores = _score_test_split(
        model,
        criterion,
        dataset,
        rows_by_id,
        pair_by_id,
        device,
        amp_enabled,
        args.num_workers,
    )
    selected = select_diverse_best(scores, args.top_k)
    run_dir = _make_run_directory(args.output_root.resolve())
    fused_dir = run_dir / "fused"
    outputs: list[dict[str, Any]] = []
    factor = int(config["preprocessing"]["model_downsample_factor"])
    for rank, row in enumerate(selected, start=1):
        pair = pair_by_id[row["sample_id"]]
        vis, ir = load_inference_pair(pair.vis_path, pair.ir_path, config)
        fused, details = infer_pair_tensor(
            model,
            vis,
            ir,
            device=device,
            requested_mode="sliding",
            tile_size=args.tile_size,
            overlap=args.overlap,
            factor=factor,
            amp_enabled=amp_enabled,
        )
        output_path = fused_dir / f"{rank:02d}__{row['sample_id']}.png"
        save_fused_png(fused, output_path)
        output = {
            **row,
            "rank": rank,
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "output_width": details["output_width"],
            "output_height": details["output_height"],
        }
        outputs.append(output)
        print(json.dumps({"exported": rank, "sample_id": row["sample_id"]}, ensure_ascii=False))
    ranking_fields = [
        "rank", "sample_id", "category", "clip_key", "loss", "intensity_loss",
        "gradient_loss", "ssim_loss", "edge_loss", "selection_group",
        "diversity_rule", "source_frame_index_0_based", "output_width",
        "output_height", "output_sha256", "output_path",
    ]
    with (run_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outputs)
    _write_contact_sheet(outputs, run_dir / "contact_sheet.png")
    metadata = {
        "result_status": "heldout_test_best20_proxy",
        "quality_claim": "FusionLoss-based objective proxy without ground truth",
        "pair_count_scored": len(scores),
        "pair_count_exported": len(outputs),
        "selection_policy": "3 per test clip plus global best, pHash/frame diversity",
        "checkpoint": checkpoint_metadata,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "ir_mode": ir_mode,
        "dataset_manifest_sha256": canonical_rows_hash(manifest_rows, SAMPLE_FIELDS),
        "config_sha256": sha256_file(config_path),
        "split_source": split_source,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "device": str(device),
        "amp": amp_enabled,
        "outputs": outputs,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ir-mode", choices=("auto", "gray", "learned_gray"), default="auto")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    run_dir = run_selection(build_parser().parse_args())
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
