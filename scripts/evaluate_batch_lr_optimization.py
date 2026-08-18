"""Evaluate batch/LR candidates on val only and apply quality guardrails."""

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

from data_pipeline.schema import SAMPLE_FIELDS, DataContractError, load_dataset_config, read_csv_rows
from data_pipeline.transforms import build_paired_transform, build_rgb_transform
from main.dataset import PairedFusionDataset
from main.fusion_quality import fusion_quality_metrics
from main.inference_utils import load_model_checkpoint, save_fused_png, sha256_file
from main.loss import FusionLoss, SSIMLoss
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion
from scripts.batch_lr_experiment import (
    BASELINE_GROUP,
    COMPONENT_FIELDS,
    GROUPS,
    QUALITY_FIELDS,
    SEEDS,
    W2_LOSS_WEIGHTS,
    improvement_percent,
    latest_completed_run,
    read_json,
    summarize_curve,
    write_csv,
    write_json,
)
from scripts.loss_weight_experiment import composite_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "runs" / "batch_lr_optimization"
ROW_FIELDS = ("objective_loss", *COMPONENT_FIELDS, *QUALITY_FIELDS)


def _mean_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        raise DataContractError("不能聚合空质量结果")
    return {
        field: statistics.fmean(float(row[field]) for row in values)
        for field in ROW_FIELDS
    }


def _aggregate(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
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
    rows_by_id = {
        row["sample_id"]: row for row in read_csv_rows(manifest, SAMPLE_FIELDS)
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for sample_id in val_ids:
        row = rows_by_id[sample_id]
        grouped[(row["leakage_group_id"], row["category"])].append(row)
    selected: list[dict[str, Any]] = []
    for (group_id, category), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["frame_sequence"]), row["sample_id"]))
        for quantile, label in ((0.25, "q25"), (0.50, "q50"), (0.75, "q75")):
            row = rows[round((len(rows) - 1) * quantile)]
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
    if output.is_file() and read_json(output) != payload:
        raise DataContractError(f"固定视觉样本已存在但内容不一致: {output}")
    write_json(output, payload)
    return payload


def _available_runs(
    root: Path, shortlist: list[str], confirmed: str, init_sha256: str
) -> dict[tuple[str, int], Path]:
    runs: dict[tuple[str, int], Path] = {}
    for group in [BASELINE_GROUP, *shortlist]:
        runs[(group, 42)] = latest_completed_run(
            root,
            group,
            42,
            15,
            initialization_sha256=init_sha256,
        )
    for group in shortlist:
        for seed in (43, 44):
            try:
                runs[(group, seed)] = latest_completed_run(
                    root,
                    group,
                    seed,
                    15,
                    initialization_sha256=init_sha256,
                )
            except DataContractError:
                if group == confirmed:
                    raise
    return runs


def _criterion(device: torch.device) -> FusionLoss:
    return FusionLoss(
        lambda_intensity=W2_LOSS_WEIGHTS["intensity"],
        lambda_gradient=W2_LOSS_WEIGHTS["gradient"],
        lambda_ssim=W2_LOSS_WEIGHTS["ssim"],
        lambda_edge=W2_LOSS_WEIGHTS["edge"],
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
    checkpoint = load_model_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    criterion = _criterion(device)
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
            quality = fusion_quality_metrics(fused, ir, vis, ssim=ssim)
            denominator = quality["ir_ssim"] + quality["vis_ssim"]
            quality["h_ssim"] = (
                2.0 * quality["ir_ssim"] * quality["vis_ssim"] / denominator
                if denominator > 0
                else 0.0
            )
            row = {
                "sample_id": sample_id,
                "leakage_group_id": str(batch["leakage_group_id"][0]),
                "category": str(batch["category"][0]),
                "group": group,
                "seed": seed,
                "objective_loss": float(objective),
                **{name: float(value) for name, value in components.items()},
                **quality,
            }
            rows.append(row)
            if sample_id in selected_ids:
                output = visual_root / group / f"seed{seed}" / f"{sample_id}.png"
                save_fused_png(fused, output)
                if group == BASELINE_GROUP and seed == 42:
                    save_fused_png(vis, visual_root / "sources" / "vis" / f"{sample_id}.png")
                    save_fused_png(
                        ir.expand(-1, 3, -1, -1),
                        visual_root / "sources" / "ir" / f"{sample_id}.png",
                    )
    summary = {
        "group": group,
        "seed": seed,
        "batch_size": GROUPS[group].batch_size,
        "learning_rate": GROUPS[group].learning_rate,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "sample_count": len(rows),
        **_mean_rows(rows),
    }
    del model, criterion, ssim
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def _quality_checks(
    candidate: dict[str, float],
    baseline: dict[str, float],
    category_ratios: dict[str, float],
) -> dict[str, bool]:
    return {
        "h_ssim_guardrail": candidate["h_ssim"] / baseline["h_ssim"] >= 0.995,
        "ir_ssim_guardrail": candidate["ir_ssim"] / baseline["ir_ssim"] >= 0.99,
        "vis_ssim_guardrail": candidate["vis_ssim"] / baseline["vis_ssim"] >= 0.99,
        "standard_deviation_guardrail": candidate["standard_deviation"]
        / baseline["standard_deviation"]
        >= 0.95,
        "average_gradient_guardrail": candidate["average_gradient"]
        / baseline["average_gradient"]
        >= 0.95,
        "spatial_frequency_guardrail": candidate["spatial_frequency"]
        / baseline["spatial_frequency"]
        >= 0.95,
        "all_categories_guardrail": min(category_ratios.values()) >= 0.99,
    }


def _decision(
    summaries: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    shortlist: list[str],
    refinement: dict[str, Any],
    confirmation_curves: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {(str(row["group"]), int(row["seed"])): row for row in summaries}
    baseline = by_key[(BASELINE_GROUP, 42)]
    baseline_quality = {field: float(baseline[field]) for field in QUALITY_FIELDS}
    category_index = {
        (str(row["group"]), int(row["seed"]), str(row["category"])): row
        for row in group_rows
    }
    baseline_categories = {
        category: {
            field: float(category_index[(BASELINE_GROUP, 42, category)][field])
            for field in QUALITY_FIELDS
        }
        for category in ("health", "health_sick", "sick")
    }
    curve_by_group = {
        str(row["group"]): row for row in refinement.get("ranking", [])
    }
    confirmation_curve_by_key = {
        (str(row["group"]), int(row["seed"])): row
        for row in confirmation_curves
    }
    candidate_results: dict[str, Any] = {}
    for group in shortlist:
        seeds = sorted(seed for candidate, seed in by_key if candidate == group)
        seed_results: dict[str, Any] = {}
        for seed in seeds:
            summary = by_key[(group, seed)]
            candidate_quality = {field: float(summary[field]) for field in QUALITY_FIELDS}
            category_ratios: dict[str, float] = {}
            for category in ("health", "health_sick", "sick"):
                row = category_index[(group, seed, category)]
                candidate_category = {
                    field: float(row[field]) for field in QUALITY_FIELDS
                }
                category_ratios[category] = composite_ratio(
                    candidate_category, baseline_categories[category]
                )
            checks = _quality_checks(candidate_quality, baseline_quality, category_ratios)
            seed_results[str(seed)] = {
                "objective_loss": float(summary["objective_loss"]),
                "quality": candidate_quality,
                "quality_ratios_to_G0": {
                    field: candidate_quality[field] / baseline_quality[field]
                    for field in QUALITY_FIELDS
                },
                "category_score_ratios": category_ratios,
                "checks": checks,
                "checks_pass": all(checks.values()),
            }
        mean_quality = {
            field: statistics.fmean(
                float(seed_results[str(seed)]["quality"][field]) for seed in seeds
            )
            for field in QUALITY_FIELDS
        }
        std_quality = {
            field: (
                statistics.stdev(
                    float(seed_results[str(seed)]["quality"][field])
                    for seed in seeds
                )
                if len(seeds) > 1
                else 0.0
            )
            for field in QUALITY_FIELDS
        }
        worst_quality = {
            field: min(
                float(seed_results[str(seed)]["quality"][field]) for seed in seeds
            )
            for field in QUALITY_FIELDS
        }
        objective_values = [
            float(seed_results[str(seed)]["objective_loss"]) for seed in seeds
        ]
        curve_values = [
            confirmation_curve_by_key[(group, seed)]
            for seed in seeds
            if (group, seed) in confirmation_curve_by_key
        ]
        curve_statistics: dict[str, Any] = {}
        for field in (
            "final3_val_loss",
            "roughness",
            "median_samples_per_second",
            "median_epoch_seconds",
        ):
            values = [float(row[field]) for row in curve_values if row.get(field) is not None]
            if values:
                higher_is_worse = field in {"final3_val_loss", "roughness", "median_epoch_seconds"}
                curve_statistics[field] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "worst": max(values) if higher_is_worse else min(values),
                }
        mean_category_ratios = {
            category: statistics.fmean(
                float(seed_results[str(seed)]["category_score_ratios"][category])
                for seed in seeds
            )
            for category in ("health", "health_sick", "sick")
        }
        mean_checks = _quality_checks(
            mean_quality, baseline_quality, mean_category_ratios
        )
        seed_pass_count = sum(
            bool(seed_results[str(seed)]["checks_pass"]) for seed in seeds
        )
        curve = curve_by_group[group]
        automatic_pass = (
            len(seeds) == len(SEEDS)
            and seed_pass_count >= 2
            and all(mean_checks.values())
            and bool(curve["eligible_curve"])
        )
        candidate_results[group] = {
            "available_seeds": seeds,
            "seed_results": seed_results,
            "seed_pass_count": seed_pass_count,
            "mean_quality": mean_quality,
            "std_quality": std_quality,
            "worst_quality": worst_quality,
            "objective_loss_statistics": {
                "mean": statistics.fmean(objective_values),
                "std": statistics.stdev(objective_values) if len(objective_values) > 1 else 0.0,
                "worst": max(objective_values),
            },
            "curve_statistics": curve_statistics,
            "mean_quality_ratios_to_G0": {
                field: mean_quality[field] / baseline_quality[field]
                for field in QUALITY_FIELDS
            },
            "mean_category_score_ratios": mean_category_ratios,
            "mean_checks": mean_checks,
            "curve": curve,
            "automatic_checks_pass": automatic_pass,
        }
    selection_order = [
        str(row["group"])
        for row in refinement["ranking"]
        if row["group"] in shortlist
    ]
    automatic_winner = next(
        (
            group
            for group in selection_order
            if candidate_results[group]["automatic_checks_pass"]
        ),
        None,
    )
    pending_fallback = next(
        (
            group
            for group in selection_order
            if len(candidate_results[group]["available_seeds"]) < len(SEEDS)
        ),
        None,
    )
    baseline_curve = curve_by_group[BASELINE_GROUP]
    improvement: dict[str, Any] | None = None
    if automatic_winner:
        winner_curve = curve_by_group[automatic_winner]
        improvement = {
            "validation_loss_reduction_percent": improvement_percent(
                float(baseline_curve["final3_val_loss"]),
                float(winner_curve["final3_val_loss"]),
            ),
            "roughness_reduction_percent": improvement_percent(
                float(baseline_curve["roughness"]),
                float(winner_curve["roughness"]),
            ),
            "throughput_increase_percent": (
                -improvement_percent(
                    float(baseline_curve["median_samples_per_second"]),
                    float(winner_curve["median_samples_per_second"]),
                )
                if baseline_curve.get("median_samples_per_second")
                and winner_curve.get("median_samples_per_second")
                else None
            ),
            "epoch_time_reduction_percent": improvement_percent(
                float(baseline_curve["median_epoch_seconds"]),
                float(winner_curve["median_epoch_seconds"]),
            ),
        }
    return {
        "phase": "quality_confirmation",
        "baseline_group": BASELINE_GROUP,
        "baseline_quality": baseline_quality,
        "candidates": candidate_results,
        "automatic_winner": automatic_winner,
        "fallback_confirmation_required": (
            pending_fallback if automatic_winner is None else None
        ),
        "improvement_vs_G0": improvement,
        "visual_review": {
            "status": "pending" if automatic_winner else "not_applicable"
        },
        "final_acceptance": {
            "status": "pending_visual_review" if automatic_winner else "retain_G0",
            "selected_group": automatic_winner,
        },
        "test_images_or_metrics_accessed": False,
        "limitations": [
            "validation contains only one leakage_group_id",
            "quality metrics are proxies because no fusion ground truth exists",
            "G0 has seed42 only in this continuation experiment",
        ],
    }


def _blind_sheet(
    root: Path,
    selection: dict[str, Any],
    winner: str,
) -> None:
    order = [BASELINE_GROUP, winner]
    random.Random(20260816).shuffle(order)
    labels = ("Model X", "Model Y")
    mapping = dict(zip(labels, order, strict=True))
    write_json(root / "blind_review" / "blind_mapping.json", mapping)
    visual_root = root / "visual_outputs"
    width, height, label_height = 320, 320, 28
    columns = ("VIS", "IR", *labels)
    sheet = Image.new(
        "RGB",
        (width * len(columns), (height + label_height) * len(selection["samples"])),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(selection["samples"]):
        sample_id = str(row["sample_id"])
        paths = [
            visual_root / "sources" / "vis" / f"{sample_id}.png",
            visual_root / "sources" / "ir" / f"{sample_id}.png",
            *(
                visual_root / mapping[label] / "seed42" / f"{sample_id}.png"
                for label in labels
            ),
        ]
        for column, (label, path) in enumerate(zip(columns, paths, strict=True)):
            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "black")
            canvas.paste(
                preview,
                ((width - preview.width) // 2, (height - preview.height) // 2),
            )
            left = column * width
            top = row_index * (height + label_height)
            sheet.paste(canvas, (left, top))
            caption = label if column else f"{row['category']} {label}"
            draw.text((left + 4, top + height + 5), caption, fill="black")
    output = root / "blind_review" / "blind_val_seed42.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_ROOT / "dataset.yaml")
    parser.add_argument("--split-version", default="calibrated_v1")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    training_state = read_json(root / "training_complete.json")
    refinement = read_json(root / "refinement_decision.json")
    shortlist = [str(group) for group in training_state["shortlist"]]
    confirmed = str(training_state["confirmed_group"])
    init_sha256 = str(training_state["initialization_sha256"])
    config = load_dataset_config(args.config.resolve())
    selection = _fixed_selection(
        args.data_root.resolve(),
        config,
        args.split_version,
        root / "fixed_val_visual_samples.json",
    )
    selected_ids = {str(row["sample_id"]) for row in selection["samples"]}
    manifest = (
        args.data_root.resolve()
        / str(config["paths"]["manifests"])
        / "samples.csv"
    )
    split = (
        args.data_root.resolve()
        / str(config["paths"]["splits"])
        / args.split_version
        / "val.csv"
    )
    dataset = PairedFusionDataset(
        data_root=args.data_root.resolve(),
        samples_manifest=manifest,
        split_manifest=split,
        paired_transform=build_paired_transform(config, "val"),
        rgb_transform=build_rgb_transform(config, "val"),
        config_path=args.config.resolve(),
        base_seed=0,
    )
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    runs = _available_runs(root, shortlist, confirmed, init_sha256)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (group, seed), run_dir in sorted(runs.items()):
        rows, summary = _evaluate_run(
            run_dir,
            group,
            seed,
            dataset,
            selected_ids,
            root / "visual_outputs",
            device,
            amp_enabled,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps({"evaluated": group, "seed": seed, "samples": len(rows)}))
    write_csv(
        root / "quality_summary.csv",
        summaries,
        [
            "group",
            "seed",
            "batch_size",
            "learning_rate",
            "run_dir",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_epoch",
            "sample_count",
            *ROW_FIELDS,
        ],
    )
    group_rows = _aggregate(all_rows, ("group", "seed", "leakage_group_id", "category"))
    write_csv(
        root / "quality_by_category.csv",
        group_rows,
        [
            "group",
            "seed",
            "leakage_group_id",
            "category",
            "sample_count",
            *ROW_FIELDS,
        ],
    )
    confirmation_curves = [
        summarize_curve(group, seed, run_dir)
        for (group, seed), run_dir in sorted(runs.items())
    ]
    decision = _decision(
        summaries,
        group_rows,
        shortlist,
        refinement,
        confirmation_curves,
    )
    write_json(root / "quality_decision.json", decision)
    winner = decision.get("automatic_winner")
    if winner:
        _blind_sheet(root, selection, str(winner))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, ValueError, FloatingPointError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
