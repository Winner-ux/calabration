"""数据 schema、稳定 ID 和配置校验。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class DataContractError(ValueError):
    """输入数据或配置违反项目数据契约。"""


SAMPLE_FIELDS = [
    "sample_id",
    "ir_path",
    "rgb_path",
    "category",
    "category_label",
    "study_day",
    "experiment_id",
    "clip_id",
    "frame_sequence",
    "source_frame_index",
    "timestamp_seconds",
    "subject_id",
    "batch_id",
    "scene_id",
    "session_id",
    "lighting_condition",
    "scene_condition",
    "leakage_group_id",
    "width",
    "height",
    "ir_mode",
    "rgb_mode",
    "ir_sha256",
    "rgb_sha256",
    "ir_phash",
    "rgb_phash",
    "usable",
    "exclusion_reason",
]

GROUP_FIELDS = [
    "leakage_group_id",
    "subject_id",
    "batch_id",
    "scene_id",
    "session_id",
    "lighting_condition",
    "scene_condition",
    "day_ids",
    "experiment_ids",
    "clip_ids",
    "categories",
    "pair_count",
    "ir_source_fingerprint",
    "rgb_source_fingerprint",
]

GROUP_ASSIGNMENT_FIELDS = [
    "study_day",
    "experiment_id",
    "category",
    "clip_id",
    "subject_id",
    "batch_id",
    "scene_id",
    "session_id",
    "lighting_condition",
    "scene_condition",
    "leakage_group_id",
]

GROUP_SPLIT_ASSIGNMENT_FIELDS = [
    "leakage_group_id",
    "outer_split",
    "inner_split",
]

EXPERIMENT_ASSIGNMENT_FIELDS = [
    "experiment_key",
    "study_day",
    "experiment_id",
    "leakage_group_id",
    "outer_split",
    "inner_split",
]

SPLIT_NAMES = ("train", "val", "test")
SPLIT_SCHEMA_VERSION = 2
_SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """读取并严格校验 dataset.yaml。"""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"无法读取配置 {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DataContractError(f"配置根节点必须是映射: {config_path}")
    if config.get("schema_version") != 1:
        raise DataContractError("当前实现只支持 schema_version=1")
    for section in ("paths", "categories", "image_contract", "preprocessing", "splitting", "audit"):
        if not isinstance(config.get(section), dict):
            raise DataContractError(f"配置缺少映射节点: {section}")
    mapping = config["categories"].get("labels_to_slugs")
    if not isinstance(mapping, dict) or not mapping:
        raise DataContractError("categories.labels_to_slugs 不能为空")
    slugs = list(mapping.values())
    if len(slugs) != len(set(slugs)):
        raise DataContractError("类别 slug 必须唯一")
    if any(not isinstance(slug, str) or not _SAFE_SLUG.fullmatch(slug) for slug in slugs):
        raise DataContractError("类别 slug 只能包含小写字母、数字和下划线")
    splitting = config["splitting"]
    outer = splitting.get("outer")
    inner = splitting.get("inner")
    if not isinstance(outer, dict) or set(outer) != {"train_pool", "test"}:
        raise DataContractError("splitting.outer 必须包含 train_pool/test")
    if not isinstance(inner, dict) or set(inner) != {"train", "val"}:
        raise DataContractError("splitting.inner 必须包含 train/val")
    for name, ratios in (("outer", outer), ("inner", inner)):
        values = [float(value) for value in ratios.values()]
        if any(value <= 0.0 or value >= 1.0 for value in values):
            raise DataContractError(f"splitting.{name} 的比例必须位于 (0,1)")
        if abs(sum(values) - 1.0) > 1e-9:
            raise DataContractError(f"splitting.{name} 的比例之和必须为 1")
    minimum = int(splitting.get("minimum_complete_groups", 0))
    if minimum < 3:
        raise DataContractError("minimum_complete_groups 至少为 3")
    if splitting.get("grouping_field") != "leakage_group_id":
        raise DataContractError("当前实现的 grouping_field 必须是 leakage_group_id")
    if splitting.get("experiment_key_fields") != ["study_day", "experiment_id"]:
        raise DataContractError("experiment_key_fields 必须是 [study_day, experiment_id]")
    required_categories = splitting.get("required_categories")
    if not isinstance(required_categories, list) or not required_categories:
        raise DataContractError("required_categories 必须是非空列表")
    if len(required_categories) != len(set(required_categories)):
        raise DataContractError("required_categories 不能重复")
    unknown_categories = sorted(set(required_categories) - set(slugs))
    if unknown_categories:
        raise DataContractError(f"required_categories 包含未知类别: {unknown_categories}")
    if int(splitting.get("clips_per_category", 0)) != 1:
        raise DataContractError("当前实验契约要求 clips_per_category=1")
    crop_size = int(config["preprocessing"].get("crop_size", 0))
    factor = int(config["preprocessing"].get("model_downsample_factor", 0))
    if crop_size <= 0 or factor <= 0 or crop_size % factor:
        raise DataContractError("crop_size 必须为正数且能被 model_downsample_factor 整除")
    contract = config["image_contract"]
    if contract.get("ir_conversion") != "bt601":
        raise DataContractError("schema v1 的 ir_conversion 必须是 bt601")
    weights = contract.get("bt601_weights")
    if not isinstance(weights, list) or len(weights) != 3:
        raise DataContractError("bt601_weights 必须包含三个权重")
    sampler = config.get("training", {}).get("sampler", "epoch_shuffle")
    if sampler != "epoch_shuffle":
        raise DataContractError("当前实现的 training.sampler 必须是 epoch_shuffle")
    return config


def normalize_numeric_id(value: str, prefix: str, width: int = 3) -> str:
    """将 day3、003 等输入规范化为 day003。"""
    text = str(value).strip().lower()
    match = re.fullmatch(rf"(?:{re.escape(prefix)})?(\d+)", text)
    if not match:
        raise DataContractError(f"非法 {prefix} ID: {value!r}")
    return f"{prefix}{int(match.group(1)):0{width}d}"


def normalize_category(value: str, config: Mapping[str, Any]) -> tuple[str, str]:
    """返回 (安全 slug, 原始 label)。"""
    text = str(value).strip()
    mapping = config["categories"]["labels_to_slugs"]
    if text in mapping:
        return str(mapping[text]), text
    reverse = {str(slug): str(label) for label, slug in mapping.items()}
    if text in reverse:
        return text, reverse[text]
    raise DataContractError(f"配置中不存在类别: {text!r}")


def make_sample_id(day: str, experiment: str, category: str, clip: str, frame_sequence: int) -> str:
    return f"{day}__{experiment}__{category}__{clip}__{int(frame_sequence):06d}"


def make_experiment_key(day: str, experiment: str) -> str:
    """由已规范化或可规范化的 day/experiment 生成全局实验键。"""
    return (
        f"{normalize_numeric_id(day, 'day')}__"
        f"{normalize_numeric_id(experiment, 'experiment')}"
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    raise DataContractError(f"无法解析布尔值: {value!r}")


def ensure_relative_posix(path: str) -> str:
    """拒绝绝对路径和目录穿越，返回 POSIX 相对路径。"""
    normalized = str(path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        raise DataContractError(f"manifest 路径不能是绝对路径: {path}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise DataContractError(f"manifest 路径包含非法片段: {path}")
    return candidate.as_posix()


def resolve_data_path(data_root: str | Path, relative_path: str) -> Path:
    """解析 manifest 路径，并保证结果仍在 data_root 内。"""
    root = Path(data_root).resolve()
    relative = ensure_relative_posix(relative_path)
    resolved = (root / Path(relative)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataContractError(f"路径逃逸 data_root: {relative_path}") from exc
    return resolved


def read_csv_rows(path: str | Path, required_fields: Iterable[str] | None = None) -> list[dict[str, str]]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = set(required_fields or ()) - set(fields)
            if missing:
                raise DataContractError(f"{csv_path} 缺少字段: {sorted(missing)}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise DataContractError(f"无法读取 CSV {csv_path}: {exc}") from exc


def canonical_rows_hash(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> str:
    """对稳定排序、固定字段的 manifest 内容计算 SHA-256。"""
    field_list = list(fields)
    normalized = [
        {field: str(row.get(field, "")) for field in field_list}
        for row in rows
    ]
    normalized.sort(key=lambda row: tuple(row[field] for field in field_list))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
