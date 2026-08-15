"""将外部配对结果规划/显式复制到项目唯一 paired 层。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
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

from data_pipeline.manifest import sha256_file, write_csv_atomic  # noqa: E402
from data_pipeline.schema import (  # noqa: E402
    GROUP_ASSIGNMENT_FIELDS,
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


def _normalized_summary(
    source_path: Path,
    summary: dict[str, Any],
    canonical_task: str,
    calibration_report: Path | None,
) -> bytes:
    normalized = copy.deepcopy(summary)
    normalized["task"] = canonical_task
    normalized["source_summary_sha256"] = sha256_file(source_path)
    for modality in ("ir", "rgb"):
        source = normalized.get("sources", {}).get(modality)
        if isinstance(source, dict) and source.get("path"):
            source["path"] = _source_reference(str(source["path"]))
    if calibration_report is not None:
        normalized["calibration"] = {
            "application_report": "application_report.yaml",
            "application_report_sha256": sha256_file(calibration_report),
            "pixel_source": "calibrated_output",
        }
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


def _normalized_frames_payload(path: Path) -> bytes:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataContractError(f"frames.csv 缺少表头: {path}")
        alias = "source_frame_index_0_based"
        legacy_alias = "source_frame_0_based"
        if alias not in reader.fieldnames and legacy_alias not in reader.fieldnames:
            raise DataContractError(
                f"frames.csv 缺少 {alias}/{legacy_alias}: {path}"
            )
        fields = [alias if name == legacy_alias else name for name in reader.fieldnames]
        if len(fields) != len(set(fields)):
            raise DataContractError(f"frames.csv 规范化后字段重复: {path}")
        rows = list(reader)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        if alias not in normalized:
            normalized[alias] = normalized.pop(legacy_alias)
        writer.writerow({field: normalized.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def _write_normalized_frames(source: Path, target: Path) -> str:
    payload = _normalized_frames_payload(source)
    expected = hashlib.sha256(payload).hexdigest()
    if target.exists():
        existing = sha256_file(target)
        if existing == expected:
            return "identical"
        if existing != sha256_file(source):
            raise FileExistsError(
                f"目标 frames.csv 既非规范化内容也非原始副本，拒绝替换: {target}"
            )
        temp = target.with_name(target.name + ".tmp")
        try:
            temp.write_bytes(payload)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return "normalized"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return "copied"


def _copy_no_overwrite(source: Path, target: Path) -> str:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise FileExistsError(f"目标已存在且内容不同，拒绝覆盖: {target}")
        return "identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "copied"


def _hardlink_no_overwrite(source: Path, target: Path) -> str:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise FileExistsError(f"目标已存在且内容不同，拒绝覆盖: {target}")
        return "identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as exc:
        raise OSError(f"无法建立 NTFS 硬链接 {source} -> {target}: {exc}") from exc
    if sha256_file(source) != sha256_file(target):
        target.unlink(missing_ok=True)
        raise DataContractError(f"硬链接建立后哈希不一致: {target}")
    return "linked"


def discover_import_plan(
    source_root: Path,
    data_root: Path,
    config: dict[str, Any],
    *,
    metadata_root: Path | None = None,
    complete_only: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    metadata_root = metadata_root or source_root
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
                    relative_clip = clip_dir.relative_to(source_root)
                    metadata_dir = metadata_root / relative_clip
                    for required in ("frames.csv", "summary.json", "sync_preview.png"):
                        if not (metadata_dir / required).is_file():
                            raise DataContractError(
                                f"元数据 clip 缺少 {required}: {metadata_dir}"
                            )
                    calibration_report_path = clip_dir / "application_report.yaml"
                    if metadata_root != source_root and not calibration_report_path.is_file():
                        raise DataContractError(
                            f"校准输出缺少 application_report.yaml: {clip_dir}"
                        )
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
                            "metadata": metadata_dir,
                            "calibration_report": (
                                calibration_report_path
                                if calibration_report_path.is_file()
                                else None
                            ),
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
    required = set(config["splitting"]["required_categories"])
    expected_clips = int(config["splitting"]["clips_per_category"])
    experiment_categories: dict[str, Counter[str]] = defaultdict(Counter)
    for plan in plans:
        experiment_key = f"{plan['study_day']}__{plan['experiment_id']}"
        experiment_categories[experiment_key][plan["category"]] += 1
    incomplete = [
        key
        for key, counts in sorted(experiment_categories.items())
        if set(counts) != required
        or any(counts[category] != expected_clips for category in required)
    ]
    if complete_only and incomplete:
        excluded = set(incomplete)
        plans = [
            plan
            for plan in plans
            if f"{plan['study_day']}__{plan['experiment_id']}" not in excluded
        ]
    return plans, incomplete


def execute_plan(plans: list[dict[str, Any]], image_transfer: str) -> dict[str, int]:
    counts = {"copied": 0, "linked": 0, "normalized": 0, "identical": 0}
    for plan in plans:
        source: Path = plan["source"]
        target: Path = plan["target"]
        for modality in ("ir", "rgb"):
            for filename, source_file in sorted(plan[f"{modality}_files"].items()):
                transfer = _hardlink_no_overwrite if image_transfer == "hardlink" else _copy_no_overwrite
                status = transfer(source_file, target / modality / filename)
                counts[status] += 1
        metadata: Path = plan["metadata"]
        status = _write_normalized_frames(metadata / "frames.csv", target / "frames.csv")
        counts[status] += 1
        status = _copy_no_overwrite(
            metadata / "sync_preview.png", target / "sync_preview.png"
        )
        counts[status] += 1
        calibration_report: Path | None = plan["calibration_report"]
        if calibration_report is not None:
            status = _copy_no_overwrite(calibration_report, target / "application_report.yaml")
            counts[status] += 1
        summary_path = metadata / "summary.json"
        summary = _load_json(summary_path)
        canonical_task = f"experiment/{plan['study_day']}/{plan['experiment_id']}/{plan['category']}/{plan['clip_id']}"
        payload = _normalized_summary(summary_path, summary, canonical_task, calibration_report)
        status = _write_bytes_no_overwrite(target / "summary.json", payload)
        counts[status] += 1
    return counts


def _experiment_group_rows(plans: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for plan in sorted(
        plans,
        key=lambda value: (
            value["study_day"], value["experiment_id"], value["category"], value["clip_id"]
        ),
    ):
        row = {field: "" for field in GROUP_ASSIGNMENT_FIELDS}
        row.update(
            {
                "study_day": plan["study_day"],
                "experiment_id": plan["experiment_id"],
                "category": plan["category"],
                "clip_id": plan["clip_id"],
                "leakage_group_id": f"{plan['study_day']}_{plan['experiment_id']}",
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="可选的 frames.csv/summary.json/sync_preview.png 根目录",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data" / "dataset.yaml")
    parser.add_argument(
        "--execute-copy",
        action="store_true",
        help="兼容旧接口：显式执行 copy 导入；缺省为只读 dry-run",
    )
    parser.add_argument("--execute-import", action="store_true", help="显式执行所选导入模式")
    parser.add_argument(
        "--image-transfer",
        choices=("copy", "hardlink"),
        default="copy",
        help="图像进入正式 paired 层的方式",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="只导入满足完整 experiment 类别契约的 clip",
    )
    parser.add_argument(
        "--group-by-experiment",
        action="store_true",
        help="按 experiment 写入独立 leakage_group_id",
    )
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    execute = bool(args.execute_import or args.execute_copy)
    if args.execute_copy and args.image_transfer != "copy":
        raise DataContractError("--execute-copy 只能与 --image-transfer copy 一起使用")
    plans, incomplete = discover_import_plan(
        args.source_root.resolve(),
        args.data_root.resolve(),
        config,
        metadata_root=args.metadata_root.resolve() if args.metadata_root else None,
        complete_only=args.complete_only,
    )
    report = {
        "mode": args.image_transfer if execute else "dry-run",
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
    report["formal_split_ready"] = bool(
        len(complete_experiments) >= minimum and args.group_by_experiment
    )
    report["formal_split_blockers"] = []
    if not args.group_by_experiment:
        report["formal_split_blockers"].append(
            "group_assignments.csv is required before manifest construction"
        )
    if len(complete_experiments) < minimum:
        report["formal_split_blockers"].append(
            f"at least {minimum} complete independent leakage groups are required; "
            f"complete experiment candidates={len(complete_experiments)}, "
            f"observed clip counts={dict(category_clip_counts)}"
        )
    report["incomplete_experiments"] = incomplete
    report["excluded_incomplete_experiments"] = incomplete if args.complete_only else []
    if execute:
        args.data_root.mkdir(parents=True, exist_ok=True)
        _copy_no_overwrite(args.config.resolve(), args.data_root.resolve() / "dataset.yaml")
        report["import_result"] = execute_plan(plans, args.image_transfer)
        if args.group_by_experiment:
            assignment_path = args.data_root.resolve() / str(config["paths"]["manifests"]) / "group_assignments.csv"
            assignment_rows = _experiment_group_rows(plans)
            if assignment_path.exists():
                with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    existing_rows = list(csv.DictReader(handle))
                if existing_rows != assignment_rows:
                    raise FileExistsError(
                        f"已有 group_assignments.csv 与计划不一致，拒绝覆盖: {assignment_path}"
                    )
            else:
                write_csv_atomic(
                    assignment_path,
                    assignment_rows,
                    GROUP_ASSIGNMENT_FIELDS,
                )
            report["group_assignments"] = str(assignment_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileExistsError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
