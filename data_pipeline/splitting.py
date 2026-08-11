"""确定性的两阶段 leakage-group 数据划分。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .manifest import validate_experiment_contract, write_csv_atomic
from .schema import (
    EXPERIMENT_ASSIGNMENT_FIELDS,
    GROUP_SPLIT_ASSIGNMENT_FIELDS,
    SAMPLE_FIELDS,
    SPLIT_NAMES,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    canonical_rows_hash,
    parse_bool,
)


OuterSplit = Literal["train_pool", "test"]
InnerSplit = Literal["train", "val", "test"]


@dataclass(frozen=True)
class GroupSplitAssignment:
    """一个不可拆分 leakage group 的外层和内层归属。"""

    leakage_group_id: str
    outer_split: OuterSplit
    inner_split: InnerSplit

    def __post_init__(self) -> None:
        if not self.leakage_group_id:
            raise DataContractError("GroupSplitAssignment 缺少 leakage_group_id")
        if self.outer_split not in {"train_pool", "test"}:
            raise DataContractError(f"非法 outer_split: {self.outer_split}")
        expected_inner = {"train", "val"} if self.outer_split == "train_pool" else {"test"}
        if self.inner_split not in expected_inner:
            raise DataContractError(
                f"outer_split={self.outer_split} 与 inner_split={self.inner_split} 不一致"
            )


def _integer_targets(count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """使用最大余数法计算两路 group 数目标，且两路至少各一个。"""
    if count < 2:
        raise DataContractError("两路划分至少需要 2 个独立组")
    names = list(ratios)
    if len(names) != 2:
        raise DataContractError("两阶段划分的每个阶段必须恰好包含两个目标")
    raw = {name: count * float(ratios[name]) for name in names}
    targets = {name: max(1, math.floor(raw[name])) for name in names}
    while sum(targets.values()) < count:
        selected = max(
            names,
            key=lambda name: (raw[name] - math.floor(raw[name]), float(ratios[name]), -names.index(name)),
        )
        targets[selected] += 1
        raw[selected] = math.floor(raw[selected])
    while sum(targets.values()) > count:
        candidates = [name for name in names if targets[name] > 1]
        if not candidates:
            raise DataContractError("无法计算合法的两路 group 数目标")
        selected = min(
            candidates,
            key=lambda name: (raw[name] - targets[name], float(ratios[name]), names.index(name)),
        )
        targets[selected] -= 1
    return targets


def _group_features(samples: Iterable[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for row in samples:
        if not parse_bool(row.get("usable", "false")):
            continue
        group_id = row.get("leakage_group_id", "").strip()
        if not group_id:
            raise DataContractError(f"usable 样本缺少 leakage_group_id: {row.get('sample_id')}")
        experiment_key = f"{row['study_day']}__{row['experiment_id']}"
        feature = features.setdefault(
            group_id,
            {
                "categories": set(),
                "category_pairs": Counter(),
                "pair_count": 0,
                "experiments": set(),
                "study_day": set(),
                "lighting_condition": set(),
                "scene_condition": set(),
            },
        )
        feature["categories"].add(row["category"])
        feature["category_pairs"][row["category"]] += 1
        feature["pair_count"] += 1
        feature["experiments"].add(experiment_key)
        for field in ("study_day", "lighting_condition", "scene_condition"):
            if row.get(field):
                feature[field].add(row[field])
    return features


def _stage_score(
    assignments: Mapping[str, str],
    features: Mapping[str, Mapping[str, Any]],
    names: tuple[str, str],
    ratios: Mapping[str, float],
) -> float:
    total_pairs = sum(int(feature["pair_count"]) for feature in features.values())
    total_experiments = sum(len(feature["experiments"]) for feature in features.values())
    categories = sorted(
        {category for feature in features.values() for category in feature["categories"]}
    )
    pair_counts = Counter()
    experiment_counts = Counter()
    category_pairs = {name: Counter() for name in names}
    metadata = {
        field: {name: Counter() for name in names}
        for field in ("study_day", "lighting_condition", "scene_condition")
    }
    for group_id, name in assignments.items():
        feature = features[group_id]
        pair_counts[name] += int(feature["pair_count"])
        experiment_counts[name] += len(feature["experiments"])
        category_pairs[name].update(feature["category_pairs"])
        for field in metadata:
            for value in feature[field]:
                metadata[field][name][value] += 1

    score = 0.0
    for name in names:
        ratio = float(ratios[name])
        pair_target = total_pairs * ratio
        experiment_target = total_experiments * ratio
        score += 4.0 * ((pair_counts[name] - pair_target) / max(1.0, pair_target)) ** 2
        score += 3.0 * (
            (experiment_counts[name] - experiment_target) / max(1.0, experiment_target)
        ) ** 2
        for category in categories:
            total = sum(
                int(feature["category_pairs"].get(category, 0))
                for feature in features.values()
            )
            target = total * ratio
            score += 2.0 * (
                (category_pairs[name][category] - target) / max(1.0, target)
            ) ** 2
    for field in metadata:
        values = sorted({value for feature in features.values() for value in feature[field]})
        for value in values:
            total = sum(value in feature[field] for feature in features.values())
            for name in names:
                target = total * float(ratios[name])
                score += 0.2 * (
                    (metadata[field][name][value] - target) / max(1.0, target)
                ) ** 2
    return score


def _assign_stage(
    features: Mapping[str, Mapping[str, Any]],
    ratios: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    names = tuple(ratios)
    if len(names) != 2:
        raise DataContractError("每个 split stage 必须恰好包含两个目标")
    targets = _integer_targets(len(features), ratios)
    best: dict[str, str] | None = None
    best_score = float("inf")
    for trial in range(256):
        rng = random.Random(seed * 1009 + trial)
        order = list(features)
        rng.shuffle(order)
        order.sort(
            key=lambda group_id: (
                -int(features[group_id]["pair_count"]),
                -len(features[group_id]["experiments"]),
            )
        )
        candidate: dict[str, str] = {}
        counts = Counter()
        for group_id in order:
            choices: list[tuple[float, str]] = []
            for name in names:
                if counts[name] >= targets[name]:
                    continue
                proposed = dict(candidate)
                proposed[group_id] = name
                choices.append((_stage_score(proposed, features, names, ratios), name))
            if not choices:
                raise DataContractError("split stage 无可用目标")
            minimum_score = min(value for value, _ in choices)
            tied = sorted(name for value, name in choices if abs(value - minimum_score) < 1e-12)
            selected = rng.choice(tied)
            candidate[group_id] = selected
            counts[selected] += 1
        if dict(counts) != targets:
            continue
        score = _stage_score(candidate, features, names, ratios)
        signature = tuple(sorted(candidate.items()))
        best_signature = tuple(sorted(best.items())) if best else ()
        if score < best_score - 1e-12 or (
            abs(score - best_score) < 1e-12 and signature < best_signature
        ):
            best = candidate
            best_score = score
    if best is None:
        raise DataContractError("无法生成满足 group 数目标的确定性 split stage")
    return dict(sorted(best.items()))


def generate_nested_group_assignments(
    samples: list[dict[str, str]],
    groups: list[dict[str, str]],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, GroupSplitAssignment]:
    """先生成 train_pool/test，再在 train_pool 内生成 train/val。"""
    validate_experiment_contract(samples, config, require_usable_clips=True)
    features = _group_features(samples)
    known_groups = {row["leakage_group_id"] for row in groups}
    missing_groups = sorted(set(features) - known_groups)
    if missing_groups:
        raise DataContractError(f"samples 引用了 groups.csv 中不存在的组: {missing_groups}")
    minimum = int(config["splitting"]["minimum_complete_groups"])
    if len(features) < minimum:
        raise DataContractError(
            f"独立组不足，不能生成正式 split；至少需要 {minimum} 个完整组，实际 {len(features)}"
        )

    outer_ratios = {
        "train_pool": float(config["splitting"]["outer"]["train_pool"]),
        "test": float(config["splitting"]["outer"]["test"]),
    }
    outer = _assign_stage(features, outer_ratios, seed=seed)
    train_pool_features = {
        group_id: features[group_id]
        for group_id, split in outer.items()
        if split == "train_pool"
    }
    inner_ratios = {
        "train": float(config["splitting"]["inner"]["train"]),
        "val": float(config["splitting"]["inner"]["val"]),
    }
    inner = _assign_stage(train_pool_features, inner_ratios, seed=seed * 7919 + 17)
    result: dict[str, GroupSplitAssignment] = {}
    for group_id in sorted(features):
        if outer[group_id] == "test":
            result[group_id] = GroupSplitAssignment(group_id, "test", "test")
        else:
            result[group_id] = GroupSplitAssignment(group_id, "train_pool", inner[group_id])
    return result


def split_sample_ids(
    samples: Iterable[Mapping[str, str]],
    assignments: Mapping[str, GroupSplitAssignment],
) -> dict[str, list[str]]:
    result = {"train_pool": [], **{name: [] for name in SPLIT_NAMES}}
    for row in samples:
        if not parse_bool(row.get("usable", "false")):
            continue
        group_id = row["leakage_group_id"]
        if group_id not in assignments:
            raise DataContractError(f"样本组未分配: {group_id}")
        assignment = assignments[group_id]
        sample_id = row["sample_id"]
        if assignment.outer_split == "train_pool":
            result["train_pool"].append(sample_id)
        result[assignment.inner_split].append(sample_id)
    for values in result.values():
        values.sort()
    return result


def _experiment_assignment_rows(
    experiments: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, GroupSplitAssignment],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for experiment_key, summary in sorted(experiments.items()):
        group_id = str(summary["leakage_group_id"])
        assignment = assignments[group_id]
        rows.append(
            {
                "experiment_key": experiment_key,
                "study_day": str(summary["study_day"]),
                "experiment_id": str(summary["experiment_id"]),
                "leakage_group_id": group_id,
                "outer_split": assignment.outer_split,
                "inner_split": assignment.inner_split,
            }
        )
    return rows


def basic_split_audit(
    samples: list[dict[str, str]],
    assignments: Mapping[str, GroupSplitAssignment],
    split_ids: Mapping[str, list[str]],
    experiments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    id_to_row = {row["sample_id"]: row for row in samples}
    failures: list[str] = []
    sample_intersections: dict[str, list[str]] = {}
    group_intersections: dict[str, list[str]] = {}
    experiment_intersections: dict[str, list[str]] = {}
    experiment_by_sample = {
        sample_id: f"{row['study_day']}__{row['experiment_id']}"
        for sample_id, row in id_to_row.items()
    }
    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1 :]:
            key = f"{first}__{second}"
            first_ids = set(split_ids[first])
            second_ids = set(split_ids[second])
            sample_intersections[key] = sorted(first_ids & second_ids)
            first_groups = {id_to_row[sample_id]["leakage_group_id"] for sample_id in first_ids}
            second_groups = {id_to_row[sample_id]["leakage_group_id"] for sample_id in second_ids}
            group_intersections[key] = sorted(first_groups & second_groups)
            first_experiments = {experiment_by_sample[sample_id] for sample_id in first_ids}
            second_experiments = {experiment_by_sample[sample_id] for sample_id in second_ids}
            experiment_intersections[key] = sorted(first_experiments & second_experiments)
    train_pool = set(split_ids["train_pool"])
    train_val_union = set(split_ids["train"]) | set(split_ids["val"])
    if train_pool != train_val_union:
        failures.append("train_pool != train union val")
    if train_pool & set(split_ids["test"]):
        failures.append("test samples appear in train_pool")
    for label, intersections in (
        ("sample", sample_intersections),
        ("group", group_intersections),
        ("experiment", experiment_intersections),
    ):
        failures.extend(
            f"{label} overlap {key}: {values}"
            for key, values in intersections.items()
            if values
        )

    category_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLIT_NAMES:
        pair_counts = Counter(id_to_row[sample_id]["category"] for sample_id in split_ids[split])
        category_groups: dict[str, set[str]] = defaultdict(set)
        for sample_id in split_ids[split]:
            row = id_to_row[sample_id]
            category_groups[row["category"]].add(row["leakage_group_id"])
        category_counts[split] = {
            category: {"pairs": pair_counts[category], "groups": len(category_groups[category])}
            for category in sorted(set(pair_counts) | set(category_groups))
        }
    return {
        "formal": True,
        "audit_level": "basic",
        "fatal_count": len(failures),
        "failures": failures,
        "sample_counts": {name: len(split_ids[name]) for name in ("train_pool", *SPLIT_NAMES)},
        "group_counts": dict(sorted(Counter(a.inner_split for a in assignments.values()).items())),
        "experiment_count": len(experiments),
        "category_counts": category_counts,
        "sample_intersections": sample_intersections,
        "group_intersections": group_intersections,
        "experiment_intersections": experiment_intersections,
        "train_pool_equals_train_union_val": train_pool == train_val_union,
        "test_outside_train_pool": not bool(train_pool & set(split_ids["test"])),
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def create_split_version(
    data_root: str | Path,
    version: str,
    samples: list[dict[str, str]],
    groups: list[dict[str, str]],
    config: Mapping[str, Any],
    seed: int,
) -> Path:
    if not version or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in version
    ):
        raise DataContractError(f"非法 split version: {version!r}")
    root = Path(data_root).resolve()
    splits_root = root / str(config["paths"]["splits"])
    target = splits_root / version
    if target.exists():
        raise FileExistsError(f"split 版本已存在，拒绝覆盖: {target}")

    experiments = validate_experiment_contract(samples, config, require_usable_clips=True)
    assignments = generate_nested_group_assignments(samples, groups, config, seed)
    split_ids = split_sample_ids(samples, assignments)
    audit = basic_split_audit(samples, assignments, split_ids, experiments)
    if audit["fatal_count"]:
        raise DataContractError(f"split 内部审计失败: {audit['failures']}")
    manifest_hash = canonical_rows_hash(samples, SAMPLE_FIELDS)
    split_config = {
        "schema_version": int(config["schema_version"]),
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "split_version": version,
        "formal": True,
        "random_seed": int(seed),
        "outer_ratios": config["splitting"]["outer"],
        "inner_ratios": config["splitting"]["inner"],
        "effective_target_ratios": {
            "train": float(config["splitting"]["outer"]["train_pool"])
            * float(config["splitting"]["inner"]["train"]),
            "val": float(config["splitting"]["outer"]["train_pool"])
            * float(config["splitting"]["inner"]["val"]),
            "test": float(config["splitting"]["outer"]["test"]),
        },
        "grouping_field": config["splitting"]["grouping_field"],
        "experiment_key_fields": config["splitting"]["experiment_key_fields"],
        "required_categories": config["splitting"]["required_categories"],
        "dataset_manifest_sha256": manifest_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": config["splitting"]["algorithm"],
    }
    audit.update(
        {
            "schema_version": int(config["schema_version"]),
            "split_schema_version": SPLIT_SCHEMA_VERSION,
            "split_version": version,
            "dataset_manifest_sha256": manifest_hash,
        }
    )

    group_assignment_rows = [
        {
            "leakage_group_id": group_id,
            "outer_split": assignment.outer_split,
            "inner_split": assignment.inner_split,
        }
        for group_id, assignment in sorted(assignments.items())
    ]
    experiment_assignment_rows = _experiment_assignment_rows(experiments, assignments)
    splits_root.mkdir(parents=True, exist_ok=True)
    temp = splits_root / f".{version}.tmp-{hashlib.sha256(str(seed).encode()).hexdigest()[:8]}"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=False)
    try:
        write_csv_atomic(
            temp / "group_assignments.csv",
            group_assignment_rows,
            GROUP_SPLIT_ASSIGNMENT_FIELDS,
        )
        write_csv_atomic(
            temp / "experiment_assignments.csv",
            experiment_assignment_rows,
            EXPERIMENT_ASSIGNMENT_FIELDS,
        )
        for split in ("train_pool", *SPLIT_NAMES):
            write_csv_atomic(
                temp / f"{split}.csv",
                ({"sample_id": sample_id} for sample_id in split_ids[split]),
                ["sample_id"],
            )
        write_json_atomic(temp / "split_config.json", split_config)
        write_json_atomic(temp / "audit.json", audit)
        os.replace(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


__all__ = [
    "GroupSplitAssignment",
    "basic_split_audit",
    "create_split_version",
    "generate_nested_group_assignments",
    "split_sample_ids",
    "write_json_atomic",
]
