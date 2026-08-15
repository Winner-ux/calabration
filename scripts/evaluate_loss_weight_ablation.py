"""统一评估损失权重消融；只读取 val 图像，不读取 held-out test 图像。"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data_pipeline.schema import (
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    parse_bool,
    read_csv_rows,
    resolve_data_path,
)
from data_pipeline.transforms import build_paired_transform, build_rgb_transform
from main.dataset import PairedFusionDataset
from main.fusion_quality import fusion_quality_metrics
from main.inference_utils import (
    infer_pair_tensor,
    load_inference_pair,
    load_model_checkpoint,
    save_fused_png,
    sha256_file,
)
from main.loss import FusionLoss, SSIMLoss
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion
from scripts.loss_weight_experiment import (
    BASELINE_RUNS,
    COMPONENT_FIELDS,
    QUALITY_FIELDS,
    SEEDS,
    WEIGHT_GROUPS,
    composite_ratio,
    latest_completed_run,
    weights_dict,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROW_FIELDS = (
    "objective_loss",
    "reference_loss",
    *COMPONENT_FIELDS,
    *QUALITY_FIELDS,
)


def _mean_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = list(rows)
    return {
        field: statistics.fmean(float(row[field]) for row in values)
        for field in ROW_FIELDS
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    outputs: list[dict[str, Any]] = []
    for values, group_rows in sorted(grouped.items()):
        output: dict[str, Any] = dict(zip(keys, values, strict=True))
        output["sample_count"] = len(group_rows)
        output.update(_mean_rows(group_rows))
        outputs.append(output)
    return outputs


def _fixed_selection(
    data_root: Path, config: dict[str, Any], split_version: str, output: Path
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
            row = rows[round((len(rows) - 1) * quantile)]
            selected.append({
                "sample_id": row["sample_id"],
                "leakage_group_id": group_id,
                "category": category,
                "temporal_position": label,
                "frame_sequence": int(row["frame_sequence"]),
            })
    payload = {
        "policy": "val-only; q25/q50/q75 for each leakage_group_id/category",
        "split": "val",
        "split_version": split_version,
        "val_manifest_sha256": sha256_file(split),
        "sample_count": len(selected),
        "samples": selected,
    }
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != payload:
        raise DataContractError(f"固定视觉样本已存在但内容不一致: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _run_dirs(root: Path, phase: str, groups: list[str]) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    seeds = (42,) if phase == "screening" else SEEDS
    found.update({("W0", seed): PROJECT_ROOT / BASELINE_RUNS[seed] for seed in seeds})
    epochs = 5 if phase == "screening" else 10
    for group in groups:
        for seed in seeds:
            found[(group, seed)] = latest_completed_run(root, group, seed, epochs)
    return found


def _criterion(group: str, device: torch.device) -> FusionLoss:
    weights = weights_dict(group)
    return FusionLoss(
        weights["intensity"], weights["gradient"], weights["ssim"], weights["edge"]
    ).to(device)


def _evaluate_run(
    run_dir: Path,
    group: str,
    seed: int,
    dataset: PairedFusionDataset,
    selected_ids: set[str],
    visual_root: Path,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode="gray")
    checkpoint_path = run_dir / "last.pt"
    audit = load_model_checkpoint(model, checkpoint_path)
    checkpoint_epoch = int(audit.get("epoch", -1))
    model.to(device).eval()
    criterion = _criterion(group, device)
    reference_weights = weights_dict("W0")
    ssim = SSIMLoss().to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            sample_id = str(batch["sample_id"][0])
            vis = batch["vis"].to(device)
            ir = batch["ir"].to(device)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused = model(vis, ir)
            objective, components = criterion(fused.float(), ir.float(), vis.float())
            reference = sum(
                reference_weights[name.removesuffix("_loss")] * float(value)
                for name, value in components.items()
            )
            quality = fusion_quality_metrics(fused, ir, vis, ssim=ssim)
            denominator = quality["ir_ssim"] + quality["vis_ssim"]
            quality["h_ssim"] = (
                2.0 * quality["ir_ssim"] * quality["vis_ssim"] / denominator
                if denominator > 0 else 0.0
            )
            row = {
                "sample_id": sample_id,
                "leakage_group_id": str(batch["leakage_group_id"][0]),
                "category": str(batch["category"][0]),
                "group": group,
                "seed": seed,
                "objective_loss": float(objective),
                "reference_loss": reference,
                **{name: float(value) for name, value in components.items()},
                **quality,
            }
            rows.append(row)
            if sample_id in selected_ids:
                save_fused_png(fused, visual_root / "center_crop" / group / f"seed{seed}" / f"{sample_id}.png")
    summary = {
        "group": group,
        "label": WEIGHT_GROUPS[group]["label"],
        "seed": seed,
        "weights": json.dumps(weights_dict(group), sort_keys=True),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": audit["sha256"],
        "checkpoint_epoch": checkpoint_epoch,
        "sample_count": len(rows),
        **_mean_rows(rows),
    }
    return rows, summary


def _score_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(str(row["group"]), int(row["seed"])): row for row in summaries}
    outputs: list[dict[str, Any]] = []
    for row in summaries:
        baseline = indexed[("W0", int(row["seed"]))]
        scored = dict(row)
        scored["score_ratio"] = composite_ratio(row, baseline)
        scored["score_delta_percent"] = (scored["score_ratio"] - 1.0) * 100.0
        outputs.append(scored)
    return outputs


def _screening_decision(scored: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        (row for row in scored if row["group"] != "W0"),
        key=lambda row: (-float(row["score_ratio"]), str(row["group"])),
    )
    return {
        "phase": "screening",
        "selection_checkpoint": "fixed epoch 4 (five completed epochs)",
        "baseline_checkpoint": "fixed epoch 9 existing W0",
        "ranking": [
            {
                "rank": index + 1,
                "group": row["group"],
                "score_ratio": row["score_ratio"],
                "score_delta_percent": row["score_delta_percent"],
            }
            for index, row in enumerate(candidates)
        ],
        "shortlist": [str(row["group"]) for row in candidates[:2]],
        "cross_group_weighted_loss_compared": False,
        "test_images_or_metrics_accessed": False,
    }


def _final_decision(
    scored: list[dict[str, Any]], group_rows: list[dict[str, Any]], groups: list[str]
) -> dict[str, Any]:
    by_key = {(str(row["group"]), int(row["seed"])): row for row in scored}
    group_index = {
        (str(row["group"]), int(row["seed"]), str(row["category"])): row
        for row in group_rows
    }
    results: dict[str, Any] = {}
    for group in groups:
        paired = [float(by_key[(group, seed)]["score_ratio"]) for seed in SEEDS]
        mean_candidate = {
            field: statistics.fmean(float(by_key[(group, seed)][field]) for seed in SEEDS)
            for field in QUALITY_FIELDS
        }
        mean_baseline = {
            field: statistics.fmean(float(by_key[("W0", seed)][field]) for seed in SEEDS)
            for field in QUALITY_FIELDS
        }
        category_ratios: dict[str, float] = {}
        for category in ("health", "health_sick", "sick"):
            candidate_category = {
                field: statistics.fmean(
                    float(group_index[(group, seed, category)][field]) for seed in SEEDS
                ) for field in QUALITY_FIELDS
            }
            baseline_category = {
                field: statistics.fmean(
                    float(group_index[("W0", seed, category)][field]) for seed in SEEDS
                ) for field in QUALITY_FIELDS
            }
            category_ratios[category] = composite_ratio(candidate_category, baseline_category)
        checks = {
            "mean_score_gain_at_least_0_5_percent": statistics.fmean(paired) >= 1.005,
            "wins_at_least_two_seeds": sum(value > 1.0 for value in paired) >= 2,
            "h_ssim_guardrail": mean_candidate["h_ssim"] / mean_baseline["h_ssim"] >= 0.995,
            "ir_ssim_guardrail": mean_candidate["ir_ssim"] / mean_baseline["ir_ssim"] >= 0.99,
            "vis_ssim_guardrail": mean_candidate["vis_ssim"] / mean_baseline["vis_ssim"] >= 0.99,
            "standard_deviation_guardrail": mean_candidate["standard_deviation"] / mean_baseline["standard_deviation"] >= 0.95,
            "average_gradient_guardrail": mean_candidate["average_gradient"] / mean_baseline["average_gradient"] >= 0.95,
            "spatial_frequency_guardrail": mean_candidate["spatial_frequency"] / mean_baseline["spatial_frequency"] >= 0.95,
            "all_categories_guardrail": min(category_ratios.values()) >= 0.99,
        }
        results[group] = {
            "paired_seed_score_ratios": dict(zip(map(str, SEEDS), paired, strict=True)),
            "mean_score_ratio": statistics.fmean(paired),
            "mean_score_delta_percent": (statistics.fmean(paired) - 1.0) * 100.0,
            "mean_quality": mean_candidate,
            "baseline_mean_quality": mean_baseline,
            "category_score_ratios": category_ratios,
            "automatic_checks": checks,
            "automatic_checks_pass": all(checks.values()),
        }
    eligible = [group for group in groups if results[group]["automatic_checks_pass"]]
    automatic_winner = max(eligible, key=lambda name: results[name]["mean_score_ratio"], default=None)
    return {
        "phase": "final",
        "selection_checkpoint": "fixed epoch 9 (ten completed epochs)",
        "candidates": results,
        "automatic_winner": automatic_winner,
        "visual_review": {"status": "pending_blind_review"},
        "final_acceptance": {
            "status": "pending_blind_review" if automatic_winner else "retain_W0",
            "selected_group": None,
            "reason": (
                "等待全分辨率盲评" if automatic_winner
                else "没有候选通过预注册的全部自动接受标准"
            ),
        },
        "cross_group_weighted_loss_compared": False,
        "test_images_or_metrics_accessed": False,
        "limitations": [
            "validation contains only one leakage_group_id",
            "all metrics are proxies without fusion ground truth",
            "entropy and sharpness proxies can be inflated by noise, so blind artifact review is mandatory",
        ],
    }


def _full_resolution_visuals(
    root: Path,
    groups: list[str],
    run_dirs: dict[tuple[str, int], Path],
    selection: dict[str, Any],
    data_root: Path,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> None:
    manifest = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    rows_by_id = {row["sample_id"]: row for row in read_csv_rows(manifest, SAMPLE_FIELDS)}
    selected_rows = [rows_by_id[str(item["sample_id"])] for item in selection["samples"]]
    visual_root = root / "blind_review" / "full_resolution"
    inference = config.get("inference", {})
    factor = int(config["preprocessing"]["model_downsample_factor"])
    pad_mode = str(config["preprocessing"].get("pad_mode", "reflect"))
    tile_size = int(inference.get("tile_size", config["preprocessing"]["crop_size"]))
    overlap = int(inference.get("overlap", 64))
    max_full_tokens = int(inference.get("max_full_tokens", 12_544))
    mappings: dict[str, Any] = {}
    inference_rows: list[dict[str, Any]] = []
    all_groups = ["W0", *groups]
    for seed in SEEDS:
        order = list(all_groups)
        random.Random(20260814 + seed).shuffle(order)
        labels = [f"Model {chr(88 + index)}" for index in range(len(order))]
        mappings[str(seed)] = dict(zip(labels, order, strict=True))
        models: dict[str, torch.nn.Module] = {}
        for group in all_groups:
            model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode="gray")
            load_model_checkpoint(model, run_dirs[(group, seed)] / "last.pt")
            models[group] = model.to(device).eval()
        for row in selected_rows:
            if not parse_bool(row["usable"]):
                raise DataContractError(f"val 样本 usable=false: {row['sample_id']}")
            sample_id = str(row["sample_id"])
            vis_path = resolve_data_path(data_root, row["rgb_path"])
            ir_path = resolve_data_path(data_root, row["ir_path"])
            vis, ir = load_inference_pair(vis_path, ir_path, config)
            save_fused_png(vis, visual_root / "sources" / "vis" / f"{sample_id}.png")
            save_fused_png(ir.expand(-1, 3, -1, -1), visual_root / "sources" / "ir" / f"{sample_id}.png")
            for group in all_groups:
                fused, details = infer_pair_tensor(
                    models[group], vis, ir, device=device, requested_mode="auto",
                    tile_size=tile_size, overlap=overlap, factor=factor,
                    pad_mode=pad_mode, max_full_tokens=max_full_tokens,
                    amp_enabled=amp_enabled,
                )
                output = visual_root / group / f"seed{seed}" / f"{sample_id}.png"
                save_fused_png(fused, output)
                inference_rows.append({
                    "seed": seed, "group": group, "sample_id": sample_id,
                    "category": row["category"], "output": str(output), **details,
                })
        _contact_sheet(visual_root, selected_rows, seed, labels, mappings[str(seed)])
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (root / "blind_review" / "blind_mapping.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "blind_review" / "full_resolution_inference.json").write_text(
        json.dumps({"scope": "val-only", "rows": inference_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _contact_sheet(
    root: Path,
    rows: list[dict[str, str]],
    seed: int,
    labels: list[str],
    mapping: dict[str, str],
) -> None:
    width, height, label_height = 256, 144, 24
    columns = ["VIS", "IR", *labels]
    sheet = Image.new("RGB", (width * len(columns), (height + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        paths = [
            root / "sources" / "vis" / f"{sample_id}.png",
            root / "sources" / "ir" / f"{sample_id}.png",
            *(root / mapping[label] / f"seed{seed}" / f"{sample_id}.png" for label in labels),
        ]
        for column, (label, path) in enumerate(zip(columns, paths, strict=True)):
            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "black")
            canvas.paste(preview, ((width - preview.width) // 2, (height - preview.height) // 2))
            left, top = column * width, row_index * (height + label_height)
            sheet.paste(canvas, (left, top))
            caption = label if column else f"{row['category']} {label}"
            draw.text((left + 3, top + height + 3), caption, fill="black")
    output = root.parent / f"blind_val_seed{seed}.png"
    sheet.save(output, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("screening", "final"), required=True)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--full-resolution", action="store_true")
    args = parser.parse_args()

    root = args.ablation_root.resolve()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    config = load_dataset_config(config_path)
    groups = list(args.groups or (["W1", "W2", "W3", "W4", "W5"] if args.phase == "screening" else []))
    if not groups or any(group not in WEIGHT_GROUPS or group == "W0" for group in groups):
        raise DataContractError(f"非法候选组: {groups}")
    selection = _fixed_selection(data_root, config, args.split_version, root / "fixed_val_visual_samples.json")
    selected_ids = {str(row["sample_id"]) for row in selection["samples"]}
    manifest = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    split = data_root / str(config["paths"]["splits"]) / args.split_version / "val.csv"
    dataset = PairedFusionDataset(
        data_root=data_root,
        samples_manifest=manifest,
        split_manifest=split,
        paired_transform=build_paired_transform(config, "val"),
        rgb_transform=build_rgb_transform(config, "val"),
        config_path=config_path,
        base_seed=0,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda" and not args.no_amp
    run_dirs = _run_dirs(root, args.phase, groups)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seeds = (42,) if args.phase == "screening" else SEEDS
    for group in ["W0", *groups]:
        for seed in seeds:
            rows, summary = _evaluate_run(
                run_dirs[(group, seed)], group, seed, dataset, selected_ids,
                root / "visual_outputs" / args.phase, device, amp_enabled,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(json.dumps({"evaluated": group, "seed": seed, "samples": len(rows)}))
    scored = _score_summaries(summaries)
    summary_fields = [
        "group", "label", "seed", "weights", "run_dir", "checkpoint",
        "checkpoint_sha256", "checkpoint_epoch", "sample_count", *ROW_FIELDS,
        "score_ratio", "score_delta_percent",
    ]
    write_csv(root / f"{args.phase}_summary.csv", scored, summary_fields)
    group_rows = _aggregate(all_rows, ("group", "seed", "leakage_group_id", "category"))
    write_csv(
        root / f"{args.phase}_group_quality.csv", group_rows,
        ["group", "seed", "leakage_group_id", "category", "sample_count", *ROW_FIELDS],
    )
    if args.phase == "screening":
        decision = _screening_decision(scored)
        output = root / "screening_decision.json"
    else:
        decision = _final_decision(scored, group_rows, groups)
        output = root / "decision.json"
        if args.full_resolution:
            _full_resolution_visuals(
                root, groups, run_dirs, selection, data_root, config, device, amp_enabled
            )
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, ValueError, FloatingPointError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
