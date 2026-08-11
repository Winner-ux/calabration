"""配对图像发现、稳定 manifest 和内容指纹。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .schema import (
    GROUP_ASSIGNMENT_FIELDS,
    GROUP_FIELDS,
    SAMPLE_FIELDS,
    DataContractError,
    ensure_relative_posix,
    make_experiment_key,
    make_sample_id,
    normalize_category,
    normalize_numeric_id,
    read_csv_rows,
)


def validate_experiment_contract(
    samples: Iterable[Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    require_usable_clips: bool = False,
) -> dict[str, dict[str, Any]]:
    """校验每个实验的类别、clip 和 group 契约并返回实验摘要。"""
    required = set(config["splitting"]["required_categories"])
    expected_clips = int(config["splitting"]["clips_per_category"])
    experiments: dict[str, dict[str, Any]] = {}
    for row in samples:
        day = row.get("study_day", "")
        experiment = row.get("experiment_id", "")
        key = make_experiment_key(day, experiment)
        summary = experiments.setdefault(
            key,
            {
                "experiment_key": key,
                "study_day": normalize_numeric_id(day, "day"),
                "experiment_id": normalize_numeric_id(experiment, "experiment"),
                "categories": defaultdict(set),
                "usable_clips": defaultdict(set),
                "leakage_group_ids": set(),
                "sample_ids": [],
            },
        )
        category = row.get("category", "")
        clip = row.get("clip_id", "")
        summary["categories"][category].add(clip)
        if row.get("usable") == "true":
            summary["usable_clips"][category].add(clip)
        group_id = row.get("leakage_group_id", "").strip()
        if group_id:
            summary["leakage_group_ids"].add(group_id)
        summary["sample_ids"].append(row.get("sample_id", ""))

    for key, summary in sorted(experiments.items()):
        actual_categories = set(summary["categories"])
        missing = sorted(required - actual_categories)
        unexpected = sorted(actual_categories - required)
        if missing or unexpected:
            raise DataContractError(
                f"实验 {key} 类别不完整: missing={missing}, unexpected={unexpected}"
            )
        invalid_clip_counts = {
            category: len(summary["categories"][category])
            for category in sorted(required)
            if len(summary["categories"][category]) != expected_clips
        }
        if invalid_clip_counts:
            raise DataContractError(
                f"实验 {key} 每类必须恰好 {expected_clips} 个 clip: {invalid_clip_counts}"
            )
        if len(summary["leakage_group_ids"]) != 1:
            raise DataContractError(
                f"实验 {key} 必须映射到一个 leakage_group_id: "
                f"{sorted(summary['leakage_group_ids'])}"
            )
        if require_usable_clips:
            unusable = [
                category
                for category in sorted(required)
                if len(summary["usable_clips"][category]) != expected_clips
            ]
            if unusable:
                raise DataContractError(f"实验 {key} 包含无可用样本的 clip: {unusable}")
        summary["leakage_group_id"] = next(iter(summary["leakage_group_ids"]))
    return experiments


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    """使用 NumPy DCT 生成稳定 64-bit pHash，不依赖 imagehash。"""
    size = hash_size * high_frequency_factor
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    indices = np.arange(size, dtype=np.float64)
    basis = np.cos(np.pi * (2 * indices[:, None] + 1) * indices[None, :] / (2 * size))
    basis[:, 0] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / size)
    dct = basis.T @ pixels @ basis
    low = dct[:hash_size, :hash_size]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def phash_distance(first: str, second: str) -> int:
    if len(first) != len(second):
        raise DataContractError("pHash 长度不一致")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"JSON 根节点必须是对象: {path}")
    return value


def _assignment_key(day: str, experiment: str, category: str, clip: str, config: Mapping[str, Any]) -> tuple[str, str, str, str]:
    category_slug, _ = normalize_category(category, config)
    return (
        normalize_numeric_id(day, "day"),
        normalize_numeric_id(experiment, "experiment"),
        category_slug,
        normalize_numeric_id(clip, "clip"),
    )


def read_group_assignments(path: str | Path, config: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = read_csv_rows(path, GROUP_ASSIGNMENT_FIELDS)
    assignments: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        key = _assignment_key(
            row["study_day"], row["experiment_id"], row["category"], row["clip_id"], config
        )
        leakage_group_id = row["leakage_group_id"].strip()
        if not leakage_group_id:
            raise DataContractError(f"group_assignments.csv 第 {line_number} 行缺少 leakage_group_id")
        if key in assignments:
            raise DataContractError(f"group_assignments.csv 存在重复 clip 映射: {key}")
        normalized = {field: row.get(field, "").strip() for field in GROUP_ASSIGNMENT_FIELDS}
        normalized.update(
            {
                "study_day": key[0],
                "experiment_id": key[1],
                "category": key[2],
                "clip_id": key[3],
            }
        )
        assignments[key] = normalized
    return assignments


def discover_clip_directories(data_root: str | Path, config: Mapping[str, Any]) -> list[Path]:
    """仅供 manifest 构建器扫描受控的 paired 层。"""
    root = Path(data_root).resolve()
    paired = root / str(config["paths"]["paired"])
    if not paired.is_dir():
        raise DataContractError(f"正式图片目录不存在: {paired}")
    clips: list[Path] = []
    for day in sorted(path for path in paired.iterdir() if path.is_dir()):
        normalize_numeric_id(day.name, "day")
        for experiment in sorted(path for path in day.iterdir() if path.is_dir()):
            normalize_numeric_id(experiment.name, "experiment")
            for category in sorted(path for path in experiment.iterdir() if path.is_dir()):
                normalize_category(category.name, config)
                for clip in sorted(path for path in category.iterdir() if path.is_dir()):
                    normalize_numeric_id(clip.name, "clip")
                    clips.append(clip)
    if not clips:
        raise DataContractError(f"{paired} 中没有发现任何 clip")
    return clips


def _read_frame_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(
        path,
        ("sequence", "filename", "source_frame_index_0_based", "timestamp_seconds"),
    )
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = row["filename"].strip()
        if not filename or filename in result:
            raise DataContractError(f"frames.csv 文件名为空或重复: {path} -> {filename!r}")
        result[filename] = row
    return result


def _image_metadata(path: Path) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode, int(image.width), int(image.height)
    except (OSError, UnidentifiedImageError) as exc:
        raise DataContractError(f"图像无法读取 {path}: {exc}") from exc


def _relative(path: Path, root: Path) -> str:
    return ensure_relative_posix(path.resolve().relative_to(root.resolve()).as_posix())


def build_manifest_rows(
    data_root: str | Path,
    config: Mapping[str, Any],
    assignments_path: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """从唯一 paired 层生成 samples/groups 行；不写文件。"""
    root = Path(data_root).resolve()
    assignments = read_group_assignments(assignments_path, config)
    clips = discover_clip_directories(root, config)
    samples: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()
    group_sources: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"ir": [], "rgb": []}
    )
    hash_size = int(config["audit"].get("phash_size", 8))
    high_factor = int(config["audit"].get("phash_high_frequency_factor", 4))
    allowed_ir_modes = set(config["image_contract"]["ir_source_modes"])
    rgb_source_mode = str(config["image_contract"]["rgb_source_mode"])

    for clip_dir in clips:
        category_dir = clip_dir.parent
        experiment_dir = category_dir.parent
        day_dir = experiment_dir.parent
        day = normalize_numeric_id(day_dir.name, "day")
        experiment = normalize_numeric_id(experiment_dir.name, "experiment")
        category, category_label = normalize_category(category_dir.name, config)
        clip = normalize_numeric_id(clip_dir.name, "clip")
        key = (day, experiment, category, clip)
        assignment = assignments.get(key)
        if assignment is None:
            raise DataContractError(
                "clip 缺少人工 leakage group 映射: " + "/".join(key)
            )
        ir_dir = clip_dir / "ir"
        rgb_dir = clip_dir / "rgb"
        if not ir_dir.is_dir() or not rgb_dir.is_dir():
            raise DataContractError(f"clip 缺少 ir/ 或 rgb/ 目录: {clip_dir}")
        frames_path = clip_dir / "frames.csv"
        summary_path = clip_dir / "summary.json"
        frames = _read_frame_metadata(frames_path)
        summary = _load_json(summary_path)
        sources = summary.get("sources", {})
        for modality in ("ir", "rgb"):
            fingerprint = sources.get(modality, {}).get("fingerprint", {})
            if fingerprint:
                group_sources[assignment["leakage_group_id"]][modality].append(fingerprint)

        ir_files = {path.name: path for path in ir_dir.glob("*.png") if path.is_file()}
        rgb_files = {path.name: path for path in rgb_dir.glob("*.png") if path.is_file()}
        filenames = sorted(set(ir_files) | set(rgb_files) | set(frames))
        for filename in filenames:
            reasons: list[str] = []
            ir_path = ir_files.get(filename, ir_dir / filename)
            rgb_path = rgb_files.get(filename, rgb_dir / filename)
            if filename not in ir_files:
                reasons.append("missing_ir")
            if filename not in rgb_files:
                reasons.append("missing_rgb")
            frame = frames.get(filename)
            if frame is None:
                reasons.append("missing_frames_csv_record")
                try:
                    sequence = int(Path(filename).stem)
                except ValueError:
                    sequence = 0
                source_frame_index = ""
                timestamp_seconds = ""
            else:
                sequence = int(frame["sequence"])
                source_frame_index = frame["source_frame_index_0_based"].strip()
                timestamp_seconds = frame["timestamp_seconds"].strip()
            if Path(filename).stem != f"{sequence:06d}":
                reasons.append("filename_sequence_mismatch")
            sample_id = make_sample_id(day, experiment, category, clip, sequence)
            if sample_id in seen_sample_ids:
                raise DataContractError(f"sample_id 冲突: {sample_id}")
            seen_sample_ids.add(sample_id)

            ir_mode = rgb_mode = ""
            width = height = 0
            ir_sha = rgb_sha = ir_phash = rgb_phash = ""
            if ir_path.is_file():
                try:
                    ir_mode, ir_width, ir_height = _image_metadata(ir_path)
                    ir_sha = sha256_file(ir_path)
                    ir_phash = perceptual_hash(ir_path, hash_size, high_factor)
                    width, height = ir_width, ir_height
                    if ir_mode not in allowed_ir_modes:
                        reasons.append(f"unsupported_ir_mode:{ir_mode}")
                except DataContractError as exc:
                    reasons.append(f"unreadable_ir:{exc}")
            if rgb_path.is_file():
                try:
                    rgb_mode, rgb_width, rgb_height = _image_metadata(rgb_path)
                    rgb_sha = sha256_file(rgb_path)
                    rgb_phash = perceptual_hash(rgb_path, hash_size, high_factor)
                    if not width:
                        width, height = rgb_width, rgb_height
                    elif (width, height) != (rgb_width, rgb_height):
                        reasons.append("spatial_size_mismatch")
                    if rgb_mode != rgb_source_mode:
                        reasons.append(f"unsupported_rgb_mode:{rgb_mode}")
                except DataContractError as exc:
                    reasons.append(f"unreadable_rgb:{exc}")

            samples.append(
                {
                    "sample_id": sample_id,
                    "ir_path": _relative(ir_path, root),
                    "rgb_path": _relative(rgb_path, root),
                    "category": category,
                    "category_label": category_label,
                    "study_day": day,
                    "experiment_id": experiment,
                    "clip_id": clip,
                    "frame_sequence": str(sequence),
                    "source_frame_index": source_frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "subject_id": assignment.get("subject_id", ""),
                    "batch_id": assignment.get("batch_id", ""),
                    "scene_id": assignment.get("scene_id", ""),
                    "session_id": assignment.get("session_id", ""),
                    "lighting_condition": assignment.get("lighting_condition", ""),
                    "scene_condition": assignment.get("scene_condition", ""),
                    "leakage_group_id": assignment["leakage_group_id"],
                    "width": str(width),
                    "height": str(height),
                    "ir_mode": ir_mode,
                    "rgb_mode": rgb_mode,
                    "ir_sha256": ir_sha,
                    "rgb_sha256": rgb_sha,
                    "ir_phash": ir_phash,
                    "rgb_phash": rgb_phash,
                    "usable": "false" if reasons else "true",
                    "exclusion_reason": ";".join(reasons),
                }
            )

    samples.sort(key=lambda row: row["sample_id"])
    validate_experiment_contract(samples, config)
    groups = _build_group_rows(samples, group_sources)
    return samples, groups


def _unique_nonempty(rows: Iterable[Mapping[str, str]], field: str) -> list[str]:
    return sorted({row.get(field, "") for row in rows if row.get(field, "")})


def _single_or_error(rows: list[dict[str, str]], field: str, group_id: str) -> str:
    values = _unique_nonempty(rows, field)
    if len(values) > 1:
        raise DataContractError(f"组 {group_id} 的 {field} 不一致: {values}")
    return values[0] if values else ""


def _build_group_rows(
    samples: list[dict[str, str]],
    group_sources: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        group_id = sample["leakage_group_id"]
        if not group_id:
            raise DataContractError(f"样本缺少 leakage_group_id: {sample['sample_id']}")
        grouped[group_id].append(sample)
    result: list[dict[str, str]] = []
    for group_id, rows in sorted(grouped.items()):
        sources = group_sources.get(group_id, {"ir": [], "rgb": []})
        ir_sources = sorted({_stable_json(value) for value in sources.get("ir", [])})
        rgb_sources = sorted({_stable_json(value) for value in sources.get("rgb", [])})
        result.append(
            {
                "leakage_group_id": group_id,
                "subject_id": _single_or_error(rows, "subject_id", group_id),
                "batch_id": _single_or_error(rows, "batch_id", group_id),
                "scene_id": _single_or_error(rows, "scene_id", group_id),
                "session_id": _single_or_error(rows, "session_id", group_id),
                "lighting_condition": _single_or_error(rows, "lighting_condition", group_id),
                "scene_condition": _single_or_error(rows, "scene_condition", group_id),
                "day_ids": _stable_json(_unique_nonempty(rows, "study_day")),
                "experiment_ids": _stable_json(_unique_nonempty(rows, "experiment_id")),
                "clip_ids": _stable_json(
                    sorted(
                        {
                            f"{row['study_day']}/{row['experiment_id']}/{row['category']}/{row['clip_id']}"
                            for row in rows
                        }
                    )
                ),
                "categories": _stable_json(_unique_nonempty(rows, "category")),
                "pair_count": str(sum(row["usable"] == "true" for row in rows)),
                "ir_source_fingerprint": _stable_json(ir_sources),
                "rgb_source_fingerprint": _stable_json(rgb_sources),
            }
        )
    return result


def write_csv_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def write_manifest_files(
    samples_path: str | Path,
    groups_path: str | Path,
    samples: list[dict[str, str]],
    groups: list[dict[str, str]],
    *,
    overwrite: bool = False,
) -> None:
    if not overwrite and (Path(samples_path).exists() or Path(groups_path).exists()):
        raise FileExistsError("拒绝覆盖已有 manifest；如确需重建，请显式传入 --overwrite")
    write_csv_atomic(samples_path, samples, SAMPLE_FIELDS, overwrite=overwrite)
    write_csv_atomic(groups_path, groups, GROUP_FIELDS, overwrite=overwrite)
