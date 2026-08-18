"""Batch size / learning-rate experiment definitions and scoring helpers."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from data_pipeline.schema import DataContractError


@dataclass(frozen=True)
class BatchLRSpec:
    group: str
    batch_size: int
    learning_rate: float
    label: str


GROUPS: dict[str, BatchLRSpec] = {
    "G0": BatchLRSpec("G0", 2, 1e-5, "baseline"),
    "G1": BatchLRSpec("G1", 4, 1e-5, "batch4_conservative"),
    "G2": BatchLRSpec("G2", 4, 2e-5, "batch4_faster"),
    "G3": BatchLRSpec("G3", 8, 5e-6, "batch8_low_lr"),
    "G4": BatchLRSpec("G4", 8, 1e-5, "batch8_conservative"),
    "G5": BatchLRSpec("G5", 8, 2e-5, "batch8_faster"),
    "G6": BatchLRSpec("G6", 16, 1e-5, "batch16_conservative"),
    "G7": BatchLRSpec("G7", 16, 2e-5, "batch16_faster"),
    "G8": BatchLRSpec("G8", 32, 2e-5, "physical_batch32_stress"),
}
SEEDS = (42, 43, 44)
BASELINE_GROUP = "G0"
W2_LOSS_WEIGHTS = {
    "intensity": 3.0,
    "gradient": 10.0,
    "ssim": 3.0,
    "edge": 2.0,
}
COMPONENT_FIELDS = ("intensity_loss", "gradient_loss", "ssim_loss", "edge_loss")
QUALITY_FIELDS = (
    "ir_ssim",
    "vis_ssim",
    "h_ssim",
    "entropy",
    "standard_deviation",
    "average_gradient",
    "spatial_frequency",
)


def candidate_matrix() -> dict[str, dict[str, Any]]:
    return {group: asdict(spec) for group, spec in GROUPS.items()}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataContractError(f"JSON 根节点必须是对象: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_metrics(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    key: float(value)
                    for key, value in raw.items()
                    if key is not None and value not in (None, "")
                }
            )
    return rows


def metric_count(run_dir: Path) -> int:
    return len(read_metrics(run_dir))


def _weights_match(config: dict[str, Any]) -> bool:
    raw = config.get("loss_weights", {})
    try:
        actual = {name: float(raw[name]) for name in W2_LOSS_WEIGHTS}
    except (KeyError, TypeError, ValueError):
        return False
    return actual == W2_LOSS_WEIGHTS


def matching_runs(
    output_root: Path,
    group: str,
    seed: int,
    *,
    initialization_sha256: str | None = None,
) -> list[Path]:
    spec = GROUPS[group]
    runs: list[Path] = []
    if not output_root.is_dir():
        return runs
    for config_path in output_root.glob("train-*/run_config.json"):
        config = read_json(config_path)
        initialization = config.get("initialization") or {}
        checks = (
            not bool(config.get("smoke_only", False)),
            int(config.get("seed", -1)) == seed,
            int(config.get("batch_size", -1)) == spec.batch_size,
            math.isclose(
                float(config.get("learning_rate", -1.0)),
                spec.learning_rate,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            str(config.get("ir_mode", "")) == "gray",
            _weights_match(config),
            initialization_sha256 is None
            or initialization.get("sha256") == initialization_sha256,
        )
        if all(checks):
            runs.append(config_path.parent)
    return sorted(runs, key=lambda path: (path.stat().st_mtime_ns, path.name))


def latest_completed_run(
    root: Path,
    group: str,
    seed: int,
    epochs: int,
    *,
    initialization_sha256: str | None = None,
) -> Path:
    runs = matching_runs(
        root / "training" / group / f"seed{seed}",
        group,
        seed,
        initialization_sha256=initialization_sha256,
    )
    complete = [
        run
        for run in runs
        if metric_count(run) >= epochs and (run / "last.pt").is_file()
    ]
    if not complete:
        raise DataContractError(
            f"缺少完整 run: group={group}, seed={seed}, epochs={epochs}"
        )
    return complete[-1]


def curve_roughness(values: Iterable[float]) -> float:
    series = [float(value) for value in values]
    if len(series) < 3 or not all(math.isfinite(value) for value in series):
        return float("inf")
    differences = [right - left for left, right in zip(series, series[1:])]
    trend = statistics.median(differences)
    mad = statistics.median(abs(value - trend) for value in differences)
    scale = statistics.median(abs(value) for value in series)
    return mad / max(scale, 1e-12)


def summarize_curve(group: str, seed: int, run_dir: Path) -> dict[str, Any]:
    rows = read_metrics(run_dir)
    if len(rows) < 3:
        raise DataContractError(f"曲线至少需要 3 个 epoch: {run_dir}")
    train = [row["train_loss"] for row in rows]
    validation = [row["val_loss"] for row in rows]
    q_value = statistics.fmean(validation[-3:])
    train_roughness = curve_roughness(train)
    val_roughness = curve_roughness(validation)
    roughness = 0.3 * train_roughness + 0.7 * val_roughness
    finite = all(math.isfinite(value) for value in (*train, *validation, roughness))
    throughput_values = [
        row["samples_per_second"] for row in rows if "samples_per_second" in row
    ]
    epoch_seconds = [row["epoch_seconds"] for row in rows]
    allocated = [
        row["peak_gpu_memory_bytes"]
        for row in rows
        if "peak_gpu_memory_bytes" in row
    ]
    reserved = [
        row["peak_gpu_memory_reserved_bytes"]
        for row in rows
        if "peak_gpu_memory_reserved_bytes" in row
    ]
    spec = GROUPS[group]
    return {
        "group": group,
        "seed": seed,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "epochs": len(rows),
        "run_dir": str(run_dir.resolve()),
        "final3_val_loss": q_value,
        "train_roughness": train_roughness,
        "val_roughness": val_roughness,
        "roughness": roughness,
        "finite": finite,
        "median_samples_per_second": (
            statistics.median(throughput_values) if throughput_values else None
        ),
        "median_epoch_seconds": statistics.median(epoch_seconds),
        "peak_gpu_memory_bytes": max(allocated) if allocated else None,
        "peak_gpu_memory_reserved_bytes": max(reserved) if reserved else None,
    }


def rank_curves(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = {str(row["group"]): row for row in summaries}
    if BASELINE_GROUP not in by_group:
        raise DataContractError("曲线排名缺少 G0 基线")
    baseline = by_group[BASELINE_GROUP]
    baseline_q = float(baseline["final3_val_loss"])
    baseline_r = max(float(baseline["roughness"]), 1e-12)
    ranked: list[dict[str, Any]] = []
    for group, row in by_group.items():
        q_ratio = float(row["final3_val_loss"]) / baseline_q
        r_ratio = float(row["roughness"]) / baseline_r
        quality_guardrail = q_ratio <= 1.01
        finite = bool(row["finite"])
        ranked.append(
            {
                **row,
                "quality_ratio_to_G0": q_ratio,
                "roughness_ratio_to_G0": r_ratio,
                "quality_guardrail_pass": quality_guardrail,
                "eligible_curve": finite and quality_guardrail,
                "combined_score": 0.5 * q_ratio + 0.5 * r_ratio,
            }
        )
    ranked.sort(
        key=lambda row: (
            not bool(row["eligible_curve"]),
            float(row["combined_score"]),
            float(row["final3_val_loss"]),
            int(row["batch_size"]),
            str(row["group"]),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    screening_pool = sorted(
        (
            row
            for row in ranked
            if row["group"] != BASELINE_GROUP and bool(row["finite"])
        ),
        key=lambda row: (
            float(row["combined_score"]),
            float(row["final3_val_loss"]),
            int(row["batch_size"]),
            str(row["group"]),
        ),
    )
    return {
        "baseline": baseline,
        "ranking": ranked,
        # The 5-epoch shortlist is diagnostic rather than a final acceptance
        # gate. Larger physical batches perform fewer optimizer steps per epoch,
        # so an initially under-trained but finite curve must remain eligible for
        # the planned 15-epoch refinement. Final selection still requires every
        # quality and stability guardrail.
        "shortlist": [str(row["group"]) for row in screening_pool[:2]],
    }


def improvement_percent(baseline: float, candidate: float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if not math.isfinite(baseline) or not math.isfinite(candidate) or baseline == 0:
        raise DataContractError(
            f"提升率要求有限且非零基线: baseline={baseline}, candidate={candidate}"
        )
    return (baseline - candidate) / baseline * 100.0


def memory_preflight_status(
    *, peak_allocated_bytes: int | None, total_bytes: int, failed: bool = False
) -> str:
    if failed or peak_allocated_bytes is None:
        return "unavailable"
    if total_bytes <= 0:
        raise DataContractError("GPU 总显存必须为正")
    return "safe" if peak_allocated_bytes / total_bytes <= 0.95 else "unsafe"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "BASELINE_GROUP",
    "BatchLRSpec",
    "COMPONENT_FIELDS",
    "GROUPS",
    "QUALITY_FIELDS",
    "SEEDS",
    "W2_LOSS_WEIGHTS",
    "candidate_matrix",
    "curve_roughness",
    "improvement_percent",
    "latest_completed_run",
    "matching_runs",
    "memory_preflight_status",
    "metric_count",
    "rank_curves",
    "read_json",
    "read_metrics",
    "summarize_curve",
    "write_csv",
    "write_json",
]
