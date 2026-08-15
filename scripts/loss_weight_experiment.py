"""损失权重消融的共享常量、发现与评分工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from data_pipeline.schema import DataContractError


WEIGHT_GROUPS: dict[str, dict[str, Any]] = {
    "W0": {"label": "baseline", "weights": (1.0, 10.0, 5.0, 2.0)},
    "W1": {"label": "balanced", "weights": (2.0, 11.0, 4.0, 1.0)},
    "W2": {"label": "intensity", "weights": (3.0, 10.0, 3.0, 2.0)},
    "W3": {"label": "detail", "weights": (1.0, 13.0, 3.0, 1.0)},
    "W4": {"label": "structure", "weights": (1.0, 8.0, 7.0, 2.0)},
    "W5": {"label": "no_edge", "weights": (1.0, 12.0, 5.0, 0.0)},
}
WEIGHT_NAMES = ("intensity", "gradient", "ssim", "edge")
SEEDS = (42, 43, 44)
QUALITY_FIELDS = (
    "ir_ssim",
    "vis_ssim",
    "h_ssim",
    "entropy",
    "standard_deviation",
    "average_gradient",
    "spatial_frequency",
)
COMPONENT_FIELDS = ("intensity_loss", "gradient_loss", "ssim_loss", "edge_loss")
BASELINE_RUNS = {
    42: Path("runs/ir_gray_ablation/gray/seed42/train-20260813-162703-894324"),
    43: Path("runs/ir_gray_ablation/gray/seed43/train-20260813-172749-018151"),
    44: Path("runs/ir_gray_ablation/gray/seed44/train-20260813-182814-637896"),
}


def weights_dict(group: str) -> dict[str, float]:
    return dict(zip(WEIGHT_NAMES, WEIGHT_GROUPS[group]["weights"], strict=True))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataContractError(f"JSON 根节点必须是对象: {path}")
    return value


def metric_count(run_dir: Path) -> int:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def matching_runs(output_root: Path, group: str, seed: int) -> list[Path]:
    expected = weights_dict(group)
    runs: list[Path] = []
    if not output_root.is_dir():
        return runs
    for config_path in output_root.glob("train-*/run_config.json"):
        config = read_json(config_path)
        raw = config.get("loss_weights", {})
        actual = {name: float(raw.get(name, -1)) for name in WEIGHT_NAMES}
        if int(config.get("seed", -1)) == seed and actual == expected:
            runs.append(config_path.parent)
    return sorted(runs, key=lambda path: (path.stat().st_mtime_ns, path.name))


def latest_completed_run(root: Path, group: str, seed: int, epochs: int) -> Path:
    runs = matching_runs(root / group / f"seed{seed}", group, seed)
    complete = [run for run in runs if metric_count(run) >= epochs and (run / "last.pt").is_file()]
    if not complete:
        raise DataContractError(f"缺少完整 run: group={group}, seed={seed}, epochs={epochs}")
    return complete[-1]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def composite_ratio(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    terms = (
        (0.40, "h_ssim"),
        (0.25, "standard_deviation"),
        (0.125, "average_gradient"),
        (0.125, "spatial_frequency"),
        (0.10, "entropy"),
    )
    score = 0.0
    for weight, name in terms:
        denominator = float(baseline[name])
        if denominator <= 0:
            raise DataContractError(f"基线指标必须为正: {name}={denominator}")
        score += weight * float(candidate[name]) / denominator
    return score


__all__ = [
    "BASELINE_RUNS",
    "COMPONENT_FIELDS",
    "QUALITY_FIELDS",
    "SEEDS",
    "WEIGHT_GROUPS",
    "WEIGHT_NAMES",
    "composite_ratio",
    "latest_completed_run",
    "matching_runs",
    "metric_count",
    "read_json",
    "weights_dict",
    "write_csv",
]
