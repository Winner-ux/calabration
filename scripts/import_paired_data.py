"""将外部配对结果规划/显式复制到项目唯一 paired 层。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest import sha256_file  # noqa: E402
from data_pipeline.schema import (  # noqa: E402
    DataContractError,
    load_dataset_config,
    normalize_category,
    normalize_numeric_id,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DataContractError(f"summary.json 根节点必须是对象: {path}")
    return value


def _source_reference(raw_path: str) -> str:
    """去掉机器盘符，仅保留 input/ 后的稳定来源标识。"""
    parts = list(Path(raw_path).parts)
    lowered = [part.lower() for part in parts]
    if "input" in lowered:
        index = lowered.index("input") + 1
        return PurePosixPath(*parts[index:]).as_posix()
    return Path(raw_path).name


def _normalized_summary(source_path: Path, summary: dict[str, Any], canonical_task: str) -> bytes:
    normalized = copy.deepcopy(summary)
    normalized["task"] = canonical_task
    normalized["source_summary_sha256"] = sha256_file(source_path)
    for modality in ("ir", "rgb"):
        source = normalized.get("sources", {}).get(modality)
        if isinstance(source, dict) and source.get("path"):
            source["path"] = _source_reference(str(source["path"]))
    return (json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes_no_overwrite(path: Path, payload: bytes) -> str:
    if path.exists():
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = hashlib.sha256(payload).hexdigest()
        if existing != expected:
            raise FileExistsError(f"目标已存在且内容不同，拒绝覆盖: {path}")
        return "identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return "copied"


def _copy_no_overwrite(source: Path, target: Path) -> str:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise FileExistsError(f"目标已存在且内容不同，拒绝覆盖: {target}")
        return "identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "copied"


def discover_import_plan(source_root: Path, data_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for day_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        day = normalize_numeric_id(day_dir.name, "day")
        for experiment_dir in sorted(path for path in day_dir.iterdir() if path.is_dir()):
            experiment = normalize_numeric_id(experiment_dir.name, "experiment")
            for category_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
                category, category_label = normalize_category(category_dir.name, config)
                for clip_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
                    clip = normalize_numeric_id(clip_dir.name, "clip")
                    ir_files = {path.name: path for path in (clip_dir / "ir").glob("*.png")}
                    rgb_files = {path.name: path for path in (clip_dir / "rgb").glob("*.png")}
                    if set(ir_files) != set(rgb_files):
                        raise DataContractError(
                            f"外部 clip 的 IR/RGB 文件名集合不一致: {clip_dir}"
                        )
                    if not ir_files:
                        raise DataContractError(f"外部 clip 没有 PNG 对: {clip_dir}")
                    for required in ("frames.csv", "summary.json", "sync_preview.png"):
                        if not (clip_dir / required).is_file():
                            raise DataContractError(f"外部 clip 缺少 {required}: {clip_dir}")
                    target = (
                        data_root
                        / str(config["paths"]["paired"])
                        / day
                        / experiment
                        / category
                        / clip
                    )
                    ir_modes = set()
                    rgb_modes = set()
                    for path in ir_files.values():
                        with Image.open(path) as image:
                            ir_modes.add(image.mode)
                    for path in rgb_files.values():
                        with Image.open(path) as image:
                            rgb_modes.add(image.mode)
                    plans.append(
                        {
                            "source": clip_dir,
                            "target": target,
                            "study_day": day,
                            "experiment_id": experiment,
                            "category": category,
                            "category_label": category_label,
                            "clip_id": clip,
                            "pair_count": len(ir_files),
                            "ir_modes": sorted(ir_modes),
                            "rgb_modes": sorted(rgb_modes),
                            "ir_files": ir_files,
                            "rgb_files": rgb_files,
                        }
                    )
    if not plans:
        raise DataContractError(f"外部目录中未发现 clip: {source_root}")
    return plans


def execute_plan(plans: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"copied": 0, "identical": 0}
    for plan in plans:
        source: Path = plan["source"]
        target: Path = plan["target"]
        for modality in ("ir", "rgb"):
            for filename, source_file in sorted(plan[f"{modality}_files"].items()):
                status = _copy_no_overwrite(source_file, target / modality / filename)
                counts[status] += 1
        for filename in ("frames.csv", "sync_preview.png"):
            status = _copy_no_overwrite(source / filename, target / filename)
            counts[status] += 1
        summary_path = source / "summary.json"
        summary = _load_json(summary_path)
        canonical_task = f"experiment/{plan['study_day']}/{plan['experiment_id']}/{plan['category']}/{plan['clip_id']}"
        payload = _normalized_summary(summary_path, summary, canonical_task)
        status = _write_bytes_no_overwrite(target / "summary.json", payload)
        counts[status] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data" / "dataset.yaml")
    parser.add_argument(
        "--execute-copy",
        action="store_true",
        help="显式授权复制；缺省为只读 dry-run",
    )
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    plans = discover_import_plan(args.source_root.resolve(), args.data_root.resolve(), config)
    report = {
        "mode": "copy" if args.execute_copy else "dry-run",
        "clip_count": len(plans),
        "pair_count": sum(plan["pair_count"] for plan in plans),
        "clips": [
            {
                "source": str(plan["source"]),
                "target": str(plan["target"]),
                "category_label": plan["category_label"],
                "category": plan["category"],
                "pair_count": plan["pair_count"],
                "ir_modes": plan["ir_modes"],
                "rgb_modes": plan["rgb_modes"],
            }
            for plan in plans
        ],
    }
    category_clip_counts = Counter(plan["category"] for plan in plans)
    experiment_categories: dict[str, Counter[str]] = defaultdict(Counter)
    for plan in plans:
        experiment_key = f"{plan['study_day']}__{plan['experiment_id']}"
        experiment_categories[experiment_key][plan["category"]] += 1
    required = set(config["splitting"]["required_categories"])
    expected_clips = int(config["splitting"]["clips_per_category"])
    complete_experiments = [
        experiment_key
        for experiment_key, counts in sorted(experiment_categories.items())
        if set(counts) == required
        and all(counts[category] == expected_clips for category in required)
    ]
    minimum = int(config["splitting"]["minimum_complete_groups"])
    report["complete_experiment_count"] = len(complete_experiments)
    report["complete_experiments"] = complete_experiments
    report["formal_split_ready"] = False
    report["formal_split_blockers"] = [
        "manual group_assignments.csv is required before manifest construction",
        f"at least {minimum} complete independent leakage groups are required; complete experiment candidates={len(complete_experiments)}, observed clip counts={dict(category_clip_counts)}",
        "registration evidence is currently sync_preview only; edge audit runs after explicit import",
    ]
    if args.execute_copy:
        report["copy_result"] = execute_plan(plans)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileExistsError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
