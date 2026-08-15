"""固定 val 样本并汇总 gray/learned_gray 三随机种子消融；不读取 test 图像。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data_pipeline.schema import SAMPLE_FIELDS, DataContractError, load_dataset_config, read_csv_rows
from data_pipeline.transforms import build_paired_transform
from main.dataset import PairedFusionDataset
from main.fusion_quality import fusion_quality_metrics
from main.inference_utils import load_model_checkpoint, save_fused_png, sha256_file
from main.loss import FusionLoss, SSIMLoss
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("gray", "learned_gray")
SEEDS = (42, 43, 44)
METRIC_FIELDS = (
    "loss",
    "intensity_loss",
    "gradient_loss",
    "ssim_loss",
    "edge_loss",
    "ir_ssim",
    "vis_ssim",
    "entropy",
    "standard_deviation",
    "average_gradient",
    "spatial_frequency",
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fixed_selection(
    data_root: Path,
    config: dict[str, Any],
    split_version: str,
    output_path: Path,
) -> dict[str, Any]:
    manifest = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    split = data_root / str(config["paths"]["splits"]) / split_version / "val.csv"
    val_ids = [row["sample_id"] for row in read_csv_rows(split, ["sample_id"])]
    rows_by_id = {row["sample_id"]: row for row in read_csv_rows(manifest, SAMPLE_FIELDS)}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for sample_id in val_ids:
        row = rows_by_id[sample_id]
        grouped[(row["leakage_group_id"], row["category"])].append(row)
    selected: list[dict[str, Any]] = []
    for (group_id, category), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["frame_sequence"]), row["sample_id"]))
        for quantile, label in ((0.25, "q25"), (0.50, "q50"), (0.75, "q75")):
            index = round((len(rows) - 1) * quantile)
            row = rows[index]
            selected.append(
                {
                    "sample_id": row["sample_id"],
                    "leakage_group_id": group_id,
                    "category": category,
                    "temporal_position": label,
                    "frame_sequence": int(row["frame_sequence"]),
                }
            )
    payload = {
        "policy": "val-only; q25/q50/q75 for each leakage_group_id/category",
        "split": "val",
        "split_version": split_version,
        "val_manifest_sha256": sha256_file(split),
        "sample_count": len(selected),
        "samples": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise DataContractError(f"固定视觉样本文件已存在但内容不一致: {output_path}")
        return existing
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _completed_runs(root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for config_path in root.rglob("run_config.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("smoke_only"):
            continue
        mode = str(payload.get("ir_mode", "gray"))
        seed = int(payload.get("seed", -1))
        run_dir = config_path.parent
        if mode in MODES and seed in SEEDS and (run_dir / "best.pt").is_file():
            metrics_path = run_dir / "metrics.csv"
            if metrics_path.is_file() and len(read_csv_rows(metrics_path, ["epoch"])) == 10:
                found[(mode, seed)].append(run_dir)
    duplicates = {key: paths for key, paths in found.items() if len(paths) != 1}
    missing = [key for key in ((m, s) for m in MODES for s in SEEDS) if key not in found]
    if duplicates or missing:
        raise DataContractError(
            f"消融 run 必须每个 mode/seed 恰好一个；missing={missing}, duplicates={duplicates}"
        )
    return {key: paths[0] for key, paths in found.items()}


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    outputs: list[dict[str, Any]] = []
    for values, group_rows in sorted(grouped.items()):
        output: dict[str, Any] = dict(zip(keys, values, strict=True))
        output["sample_count"] = len(group_rows)
        for metric in METRIC_FIELDS:
            output[metric] = statistics.fmean(float(row[metric]) for row in group_rows)
        outputs.append(output)
    return outputs


def _evaluate_run(
    run_dir: Path,
    mode: str,
    seed: int,
    dataset: PairedFusionDataset,
    selected_ids: set[str],
    visual_root: Path,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = ResNetFusion(
        Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode=mode
    )
    checkpoint = load_model_checkpoint(model, run_dir / "best.pt")
    model.to(device).eval()
    criterion = FusionLoss().to(device)
    ssim = SSIMLoss().to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    source_root = visual_root / "sources"
    output_root = visual_root / mode / f"seed{seed}"
    with torch.inference_mode():
        for batch in loader:
            sample_id = batch["sample_id"][0]
            vis = batch["vis"].to(device)
            ir = batch["ir"].to(device)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused = model(vis, ir)
            loss, components = criterion(fused.float(), ir.float(), vis.float())
            quality = fusion_quality_metrics(fused, ir, vis, ssim=ssim)
            row = {
                "sample_id": sample_id,
                "leakage_group_id": batch["leakage_group_id"][0],
                "category": batch["category"][0],
                "mode": mode,
                "seed": seed,
                "loss": float(loss),
                **{name: float(value) for name, value in components.items()},
                **quality,
            }
            rows.append(row)
            if sample_id in selected_ids:
                save_fused_png(fused, output_root / f"{sample_id}.png")
                if not (source_root / "vis" / f"{sample_id}.png").exists():
                    save_fused_png(vis, source_root / "vis" / f"{sample_id}.png")
                    save_fused_png(
                        ir.expand(-1, 3, -1, -1),
                        source_root / "ir" / f"{sample_id}.png",
                    )
    fields = [
        "sample_id", "leakage_group_id", "category", "mode", "seed", *METRIC_FIELDS
    ]
    _write_csv(run_dir / "val_quality.csv", rows, fields)
    group_rows = _aggregate(rows, ("leakage_group_id", "category"))
    _write_csv(
        run_dir / "val_group_quality.csv",
        group_rows,
        ["leakage_group_id", "category", "sample_count", *METRIC_FIELDS],
    )
    summary = {
        "mode": mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint_sha256": checkpoint["sha256"],
        "sample_count": len(rows),
    }
    for metric in METRIC_FIELDS:
        summary[metric] = statistics.fmean(float(row[metric]) for row in rows)
    return rows, summary


def _contact_sheets(
    visual_root: Path,
    selection: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    mappings: dict[str, Any] = {}
    thumb = 224
    label_height = 24
    columns = 4
    labels = ("VIS", "IR", "Model X", "Model Y")
    for seed in SEEDS:
        order = MODES if seed % 2 == 0 else tuple(reversed(MODES))
        mappings[str(seed)] = {"Model X": order[0], "Model Y": order[1]}
        rows = selection["samples"]
        sheet = Image.new(
            "RGB",
            (columns * thumb, len(rows) * (thumb + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, row in enumerate(rows):
            sample_id = row["sample_id"]
            paths = (
                visual_root / "sources" / "vis" / f"{sample_id}.png",
                visual_root / "sources" / "ir" / f"{sample_id}.png",
                visual_root / order[0] / f"seed{seed}" / f"{sample_id}.png",
                visual_root / order[1] / f"seed{seed}" / f"{sample_id}.png",
            )
            for column, (label, path) in enumerate(zip(labels, paths, strict=True)):
                with Image.open(path) as image:
                    preview = image.convert("RGB")
                    preview.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb, thumb), "black")
                canvas.paste(
                    preview,
                    ((thumb - preview.width) // 2, (thumb - preview.height) // 2),
                )
                left = column * thumb
                top = row_index * (thumb + label_height)
                sheet.paste(canvas, (left, top))
                caption = label if column else f"VIS {row['category']} {row['temporal_position']}"
                draw.text((left + 3, top + thumb + 3), caption, fill="black")
        path = output_root / f"blind_val_seed{seed}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, format="PNG")
    mapping_path = output_root / "blind_mapping.json"
    mapping_path.write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return mappings


def _decision(summaries: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["mode"], int(row["seed"])): row for row in summaries}
    paired = []
    for seed in SEEDS:
        baseline = by_key[("gray", seed)]
        candidate = by_key[("learned_gray", seed)]
        paired.append(
            {
                "seed": seed,
                "gray_val_loss": baseline["loss"],
                "learned_gray_val_loss": candidate["loss"],
                "loss_delta_candidate_minus_gray": candidate["loss"] - baseline["loss"],
            }
        )
    means = {
        mode: {
            metric: statistics.fmean(
                float(by_key[(mode, seed)][metric]) for seed in SEEDS
            )
            for metric in METRIC_FIELDS
        }
        for mode in MODES
    }
    baseline_vis_sd = statistics.stdev(
        float(by_key[("gray", seed)]["vis_ssim"]) for seed in SEEDS
    )
    candidate_wins = sum(row["loss_delta_candidate_minus_gray"] < 0 for row in paired)
    groups_by_key = {
        (
            str(row["mode"]),
            int(row["seed"]),
            str(row["leakage_group_id"]),
            str(row["category"]),
        ): row
        for row in group_rows
    }
    group_names = sorted(
        {
            (str(row["leakage_group_id"]), str(row["category"]))
            for row in group_rows
        }
    )
    paired_group_deltas: list[dict[str, Any]] = []
    for leakage_group_id, category in group_names:
        for seed in SEEDS:
            baseline = groups_by_key[("gray", seed, leakage_group_id, category)]
            candidate = groups_by_key[("learned_gray", seed, leakage_group_id, category)]
            paired_group_deltas.append(
                {
                    "leakage_group_id": leakage_group_id,
                    "category": category,
                    "seed": seed,
                    "loss_delta_candidate_minus_gray": (
                        float(candidate["loss"]) - float(baseline["loss"])
                    ),
                    "ir_ssim_delta_candidate_minus_gray": (
                        float(candidate["ir_ssim"]) - float(baseline["ir_ssim"])
                    ),
                    "vis_ssim_delta_candidate_minus_gray": (
                        float(candidate["vis_ssim"]) - float(baseline["vis_ssim"])
                    ),
                }
            )
    worst_group = max(
        paired_group_deltas,
        key=lambda row: float(row["loss_delta_candidate_minus_gray"]),
        default=None,
    )
    automatic = {
        "candidate_wins_at_least_two_seeds": candidate_wins >= 2,
        "candidate_mean_loss_better": means["learned_gray"]["loss"] < means["gray"]["loss"],
        "candidate_mean_ir_ssim_better": means["learned_gray"]["ir_ssim"] > means["gray"]["ir_ssim"],
        "vis_ssim_guardrail": (
            means["learned_gray"]["vis_ssim"]
            >= means["gray"]["vis_ssim"] - baseline_vis_sd
        ),
    }
    return {
        "paired_seed_results": paired,
        "three_seed_means": means,
        "gray_vis_ssim_across_seed_sd": baseline_vis_sd,
        "paired_group_deltas": paired_group_deltas,
        "worst_group_by_loss_delta": worst_group,
        "automatic_checks": automatic,
        "automatic_checks_pass": all(automatic.values()),
        "visual_review": "pending_manual_blind_review",
        "final_acceptance": "pending_manual_blind_review",
        "limitations": [
            "validation contains only one leakage_group_id, so worst-experiment evidence is not diverse",
            "all quality metrics are proxies without fusion ground truth",
            "test images and test metrics were not loaded by this evaluator",
            "the training entrypoint reads test.csv only for formal split integrity validation/hash",
        ],
        "group_row_count": len(group_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    root = args.ablation_root.resolve()
    config = load_dataset_config(config_path)
    selection_path = root / "fixed_val_visual_samples.json"
    selection = _fixed_selection(data_root, config, args.split_version, selection_path)
    print(json.dumps({"fixed_selection": str(selection_path), "count": selection["sample_count"]}))
    if args.prepare_only:
        return 0

    runs = _completed_runs(root)
    manifest = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    split = data_root / str(config["paths"]["splits"]) / args.split_version / "val.csv"
    dataset = PairedFusionDataset(
        data_root=data_root,
        samples_manifest=manifest,
        split_manifest=split,
        paired_transform=build_paired_transform(config, "val"),
        config_path=config_path,
        base_seed=0,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and not args.no_amp
    selected_ids = {row["sample_id"] for row in selection["samples"]}
    visual_root = root / "visual_outputs"
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode in MODES:
        for seed in SEEDS:
            rows, summary = _evaluate_run(
                runs[(mode, seed)], mode, seed, dataset, selected_ids,
                visual_root, device, amp_enabled,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(json.dumps({"evaluated": mode, "seed": seed, "samples": len(rows)}))
    summary_fields = [
        "mode", "seed", "run_dir", "checkpoint_sha256", "sample_count", *METRIC_FIELDS
    ]
    _write_csv(root / "three_seed_summary.csv", summaries, summary_fields)
    group_rows = _aggregate(all_rows, ("mode", "seed", "leakage_group_id", "category"))
    _write_csv(
        root / "all_group_quality.csv",
        group_rows,
        ["mode", "seed", "leakage_group_id", "category", "sample_count", *METRIC_FIELDS],
    )
    _contact_sheets(visual_root, selection, root / "blind_review")
    decision = _decision(summaries, group_rows)
    (root / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
