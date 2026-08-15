"""数据、split、重复候选和配准质量审计。"""

from __future__ import annotations

import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .manifest import phash_distance, validate_experiment_contract, write_csv_atomic
from .schema import (
    EXPERIMENT_ASSIGNMENT_FIELDS,
    GROUP_FIELDS,
    GROUP_SPLIT_ASSIGNMENT_FIELDS,
    SAMPLE_FIELDS,
    SPLIT_NAMES,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    canonical_rows_hash,
    parse_bool,
    read_csv_rows,
    resolve_data_path,
)
from .splitting import write_json_atomic


REGISTRATION_FIELDS = [
    "sample_id",
    "leakage_group_id",
    "edge_correlation",
    "review_status",
    "preview_path",
    "notes",
]

NEAR_DUPLICATE_FIELDS = [
    "sample_id_a",
    "sample_id_b",
    "modality",
    "phash_distance",
    "group_id_a",
    "group_id_b",
    "split_a",
    "split_b",
    "cross_split",
]


def _append_issue(container: list[dict[str, str]], code: str, message: str) -> None:
    container.append({"code": code, "message": message})


def _load_split_ids(split_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train_pool", *SPLIT_NAMES):
        rows = read_csv_rows(split_dir / f"{split}.csv", ["sample_id"])
        ids = [row["sample_id"].strip() for row in rows]
        if not ids or any(not sample_id for sample_id in ids):
            raise DataContractError(f"{split}.csv 为空或包含空 sample_id")
        if len(ids) != len(set(ids)):
            raise DataContractError(f"{split}.csv 内存在重复 sample_id")
        result[split] = ids
    return result


def _split_lookup(split_ids: Mapping[str, Iterable[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split, values in split_ids.items():
        for sample_id in values:
            if sample_id in lookup:
                raise DataContractError(
                    f"sample_id 同时出现在 {lookup[sample_id]} 和 {split}: {sample_id}"
                )
            lookup[sample_id] = split
    return lookup


def _edge_correlation(ir_path: Path, rgb_path: Path, resize_width: int) -> float:
    with Image.open(ir_path) as ir_image, Image.open(rgb_path) as rgb_image:
        if ir_image.size != rgb_image.size:
            return float("nan")
        width, height = ir_image.size
        target_height = max(1, round(height * resize_width / width))
        target = (resize_width, target_height)
        ir = np.asarray(
            ir_image.convert("L").resize(target, Image.Resampling.BILINEAR), dtype=np.float32
        )
        rgb = np.asarray(
            rgb_image.convert("L").resize(target, Image.Resampling.BILINEAR), dtype=np.float32
        )
    ir_dy, ir_dx = np.gradient(ir)
    rgb_dy, rgb_dx = np.gradient(rgb)
    ir_edge = np.sqrt(ir_dx * ir_dx + ir_dy * ir_dy).reshape(-1)
    rgb_edge = np.sqrt(rgb_dx * rgb_dx + rgb_dy * rgb_dy).reshape(-1)
    if float(ir_edge.std()) < 1e-8 or float(rgb_edge.std()) < 1e-8:
        return float("nan")
    return float(np.corrcoef(ir_edge, rgb_edge)[0, 1])


def _preview_path_for_sample(sample: Mapping[str, str]) -> str:
    ir_path = Path(sample["ir_path"])
    return (ir_path.parent.parent / "sync_preview.png").as_posix()


def _build_registration_rows(
    data_root: Path,
    samples: Iterable[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    width = int(config["audit"].get("registration_resize_width", 256))
    threshold = float(config["audit"].get("registration_review_threshold", 0.1))
    for sample in samples:
        if not parse_bool(sample.get("usable", "false")):
            continue
        ir_path = resolve_data_path(data_root, sample["ir_path"])
        rgb_path = resolve_data_path(data_root, sample["rgb_path"])
        preview = _preview_path_for_sample(sample)
        preview_exists = resolve_data_path(data_root, preview).is_file()
        try:
            correlation = _edge_correlation(ir_path, rgb_path, width)
            finite = np.isfinite(correlation)
        except (OSError, UnidentifiedImageError):
            correlation = float("nan")
            finite = False
        review = "pass_information_only" if finite and correlation >= threshold else "manual_review"
        notes = []
        if not preview_exists:
            notes.append("missing_sync_preview")
        if not finite:
            notes.append("edge_correlation_unavailable")
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "leakage_group_id": sample["leakage_group_id"],
                "edge_correlation": f"{correlation:.8f}" if finite else "",
                "review_status": review,
                "preview_path": preview,
                "notes": ";".join(notes),
            }
        )
    return rows


def _build_near_duplicate_rows(
    samples: list[dict[str, str]],
    split_lookup: Mapping[str, str],
    threshold: int,
    *,
    top_k_per_sample: int = 5,
    stats: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """保留每个样本最接近的跨 split pHash 候选，避免审计 CSV 无界增长。"""
    usable = [row for row in samples if parse_bool(row.get("usable", "false"))]
    if top_k_per_sample <= 0:
        raise DataContractError("near_duplicate_top_k_per_sample 必须为正")
    heaps: dict[tuple[str, str], list[tuple[int, int, tuple[str, ...], dict[str, str]]]] = defaultdict(list)
    comparisons = 0
    threshold_matches = 0
    sequence = 0

    def retain(modality: str, sample_id: str, row: dict[str, str], distance: int) -> None:
        nonlocal sequence
        sequence += 1
        row_key = (row["modality"], row["sample_id_a"], row["sample_id_b"])
        entry = (-distance, -sequence, row_key, row)
        heap = heaps[(modality, sample_id)]
        if len(heap) < top_k_per_sample:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    for modality in ("ir", "rgb"):
        field = f"{modality}_phash"
        for index, first in enumerate(usable):
            if not first.get(field):
                continue
            split_a = split_lookup.get(first["sample_id"], "")
            if not split_a:
                continue
            for second in usable[index + 1 :]:
                if not second.get(field):
                    continue
                split_b = split_lookup.get(second["sample_id"], "")
                if not split_b or split_a == split_b:
                    continue
                comparisons += 1
                distance = phash_distance(first[field], second[field])
                if distance > threshold:
                    continue
                threshold_matches += 1
                row = {
                    "sample_id_a": first["sample_id"],
                    "sample_id_b": second["sample_id"],
                    "modality": modality,
                    "phash_distance": str(distance),
                    "group_id_a": first["leakage_group_id"],
                    "group_id_b": second["leakage_group_id"],
                    "split_a": split_a,
                    "split_b": split_b,
                    "cross_split": "true",
                }
                retain(modality, first["sample_id"], row, distance)
                retain(modality, second["sample_id"], row, distance)
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for heap in heaps.values():
        for _, _, row_key, row in heap:
            selected[row_key] = row
    result = list(selected.values())
    result.sort(key=lambda row: (int(row["phash_distance"]), row["modality"], row["sample_id_a"], row["sample_id_b"]))
    if stats is not None:
        stats.update(
            {
                "cross_split_comparisons": comparisons,
                "threshold_matches_before_limit": threshold_matches,
                "retained_candidates": len(result),
                "top_k_per_sample": top_k_per_sample,
            }
        )
    return result


def audit_dataset(
    data_root: str | Path,
    samples: list[dict[str, str]],
    groups: list[dict[str, str]],
    config: Mapping[str, Any],
    split_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """执行只读审计并返回 audit、registration、near-duplicate 行。"""
    root = Path(data_root).resolve()
    fatals: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    sample_ids = [row.get("sample_id", "") for row in samples]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicate_ids:
        _append_issue(fatals, "duplicate_sample_id", str(duplicate_ids))
    group_ids = {row.get("leakage_group_id", "") for row in groups}
    if "" in group_ids:
        _append_issue(fatals, "empty_group_id", "groups.csv 包含空 leakage_group_id")
    id_to_row = {row["sample_id"]: row for row in samples if row.get("sample_id")}
    experiments: dict[str, dict[str, Any]] = {}
    try:
        experiments = validate_experiment_contract(samples, config, require_usable_clips=True)
    except DataContractError as exc:
        _append_issue(fatals, "invalid_experiment_contract", str(exc))

    for row in samples:
        if not row.get("leakage_group_id"):
            _append_issue(fatals, "sample_missing_group", row.get("sample_id", "<unknown>"))
        elif row["leakage_group_id"] not in group_ids:
            _append_issue(fatals, "unknown_group", row["sample_id"])
        if "sync_preview" in row.get("ir_path", "") or "sync_preview" in row.get("rgb_path", ""):
            _append_issue(fatals, "preview_in_samples", row["sample_id"])
        if not parse_bool(row.get("usable", "false")):
            _append_issue(warnings, "excluded_sample", f"{row.get('sample_id')}: {row.get('exclusion_reason')}")
            continue
        try:
            ir_path = resolve_data_path(root, row["ir_path"])
            rgb_path = resolve_data_path(root, row["rgb_path"])
        except DataContractError as exc:
            _append_issue(fatals, "invalid_path", f"{row['sample_id']}: {exc}")
            continue
        if ir_path.stem != rgb_path.stem:
            _append_issue(fatals, "stem_mismatch", row["sample_id"])
        for modality, path in (("ir", ir_path), ("rgb", rgb_path)):
            if not path.is_file():
                _append_issue(fatals, f"missing_{modality}", f"{row['sample_id']}: {path}")
        if not ir_path.is_file() or not rgb_path.is_file():
            continue
        try:
            with Image.open(ir_path) as ir_image, Image.open(rgb_path) as rgb_image:
                ir_image.load()
                rgb_image.load()
                if ir_image.size != rgb_image.size:
                    _append_issue(fatals, "size_mismatch", row["sample_id"])
                if ir_image.mode not in config["image_contract"]["ir_source_modes"]:
                    _append_issue(fatals, "ir_mode_mismatch", f"{row['sample_id']}: {ir_image.mode}")
                if rgb_image.mode != config["image_contract"]["rgb_source_mode"]:
                    _append_issue(fatals, "rgb_mode_mismatch", f"{row['sample_id']}: {rgb_image.mode}")
        except (OSError, UnidentifiedImageError) as exc:
            _append_issue(fatals, "unreadable_image", f"{row['sample_id']}: {exc}")

    split_ids: dict[str, list[str]] = {}
    split_lookup: dict[str, str] = {}
    split_config: dict[str, Any] | None = None
    if split_dir is not None:
        split_path = Path(split_dir)
        try:
            split_ids = _load_split_ids(split_path)
            split_lookup = _split_lookup({name: split_ids[name] for name in SPLIT_NAMES})
        except DataContractError as exc:
            _append_issue(fatals, "invalid_split_files", str(exc))
            split_ids = {}
        config_path = split_path / "split_config.json"
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                split_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            _append_issue(fatals, "invalid_split_config", str(exc))
        if split_config:
            expected_hash = canonical_rows_hash(samples, SAMPLE_FIELDS)
            if split_config.get("dataset_manifest_sha256") != expected_hash:
                _append_issue(fatals, "manifest_hash_mismatch", expected_hash)
            if int(split_config.get("split_schema_version", -1)) != SPLIT_SCHEMA_VERSION:
                _append_issue(
                    fatals,
                    "split_schema_version_mismatch",
                    f"expected={SPLIT_SCHEMA_VERSION}, actual={split_config.get('split_schema_version')}",
                )
            if split_config.get("algorithm") != config["splitting"]["algorithm"]:
                _append_issue(
                    fatals,
                    "split_algorithm_mismatch",
                    str(split_config.get("algorithm")),
                )
        for split, ids in split_ids.items():
            unknown = sorted(set(ids) - set(id_to_row))
            if unknown:
                _append_issue(fatals, "unknown_split_samples", f"{split}: {unknown}")
            excluded = sorted(
                sample_id
                for sample_id in ids
                if sample_id in id_to_row and not parse_bool(id_to_row[sample_id]["usable"])
            )
            if excluded:
                _append_issue(fatals, "excluded_samples_in_split", f"{split}: {excluded}")
        train_pool = set(split_ids.get("train_pool", []))
        train_val_union = set(split_ids.get("train", [])) | set(split_ids.get("val", []))
        if train_pool != train_val_union:
            _append_issue(fatals, "train_pool_mismatch", "train_pool 必须等于 train ∪ val")
        if train_pool & set(split_ids.get("test", [])):
            _append_issue(fatals, "test_in_train_pool", "test 样本出现在 train_pool")

        group_by_split: dict[str, set[str]] = {}
        experiment_by_split: dict[str, set[str]] = {}
        for split in SPLIT_NAMES:
            ids = split_ids.get(split, [])
            group_by_split[split] = {
                id_to_row[sample_id]["leakage_group_id"]
                for sample_id in ids
                if sample_id in id_to_row
            }
            experiment_by_split[split] = {
                f"{id_to_row[sample_id]['study_day']}__{id_to_row[sample_id]['experiment_id']}"
                for sample_id in ids
                if sample_id in id_to_row
            }
        for index, first in enumerate(SPLIT_NAMES):
            for second in SPLIT_NAMES[index + 1 :]:
                overlap = sorted(group_by_split.get(first, set()) & group_by_split.get(second, set()))
                if overlap:
                    _append_issue(fatals, "cross_split_group_overlap", f"{first}/{second}: {overlap}")
                experiment_overlap = sorted(
                    experiment_by_split.get(first, set())
                    & experiment_by_split.get(second, set())
                )
                if experiment_overlap:
                    _append_issue(
                        fatals,
                        "cross_split_experiment_overlap",
                        f"{first}/{second}: {experiment_overlap}",
                    )

        try:
            group_assignment_rows = read_csv_rows(
                split_path / "group_assignments.csv", GROUP_SPLIT_ASSIGNMENT_FIELDS
            )
            experiment_assignment_rows = read_csv_rows(
                split_path / "experiment_assignments.csv", EXPERIMENT_ASSIGNMENT_FIELDS
            )
            assignment_groups = {
                row["leakage_group_id"]: row for row in group_assignment_rows
            }
            if len(assignment_groups) != len(group_assignment_rows):
                _append_issue(fatals, "duplicate_split_group_assignment", "group assignment 重复")
            if set(assignment_groups) != set(group_ids):
                _append_issue(
                    fatals,
                    "split_group_assignment_coverage",
                    f"expected={sorted(group_ids)}, actual={sorted(assignment_groups)}",
                )
            assignment_experiments = {
                row["experiment_key"]: row for row in experiment_assignment_rows
            }
            if len(assignment_experiments) != len(experiment_assignment_rows):
                _append_issue(fatals, "duplicate_experiment_assignment", "experiment assignment 重复")
            if experiments and set(assignment_experiments) != set(experiments):
                _append_issue(
                    fatals,
                    "experiment_assignment_coverage",
                    f"expected={sorted(experiments)}, actual={sorted(assignment_experiments)}",
                )
            for experiment_key, summary in experiments.items():
                row = assignment_experiments.get(experiment_key)
                if row and row["leakage_group_id"] != summary["leakage_group_id"]:
                    _append_issue(
                        fatals,
                        "experiment_assignment_group_mismatch",
                        experiment_key,
                    )
        except DataContractError as exc:
            _append_issue(fatals, "invalid_assignment_files", str(exc))

        group_rows = {row["leakage_group_id"]: row for row in groups}
        for modality in ("ir", "rgb"):
            fingerprint_splits: dict[str, set[str]] = defaultdict(set)
            field = f"{modality}_source_fingerprint"
            for split, group_set in group_by_split.items():
                for group_id in group_set:
                    raw = group_rows.get(group_id, {}).get(field, "[]")
                    try:
                        fingerprints = json.loads(raw)
                    except json.JSONDecodeError:
                        _append_issue(fatals, "invalid_fingerprint_json", f"{group_id}: {field}")
                        continue
                    for fingerprint in fingerprints:
                        fingerprint_splits[str(fingerprint)].add(split)
            duplicated = {
                fingerprint: sorted(splits)
                for fingerprint, splits in fingerprint_splits.items()
                if len(splits) > 1
            }
            if duplicated:
                _append_issue(fatals, "cross_split_source_fingerprint", f"{modality}: {duplicated}")

    for modality in ("ir", "rgb"):
        field = f"{modality}_sha256"
        hash_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in samples:
            if row.get(field):
                hash_rows[row[field]].append(row)
        for digest, rows in hash_rows.items():
            groups_for_hash = {row["leakage_group_id"] for row in rows}
            if len(rows) > 1 and len(groups_for_hash) > 1:
                message = f"{modality} {digest}: {[row['sample_id'] for row in rows]}"
                splits = {split_lookup.get(row["sample_id"], "") for row in rows} - {""}
                if len(splits) > 1:
                    _append_issue(fatals, "cross_split_exact_duplicate", message)
                else:
                    _append_issue(warnings, "cross_group_exact_duplicate", message)

    category_groups: dict[str, set[str]] = defaultdict(set)
    category_pairs = Counter()
    for row in samples:
        if parse_bool(row.get("usable", "false")):
            category_groups[row["category"]].add(row["leakage_group_id"])
            category_pairs[row["category"]] += 1
    minimum = int(config["splitting"].get("minimum_complete_groups", 6))
    usable_group_count = len(
        {
            row["leakage_group_id"]
            for row in samples
            if parse_bool(row.get("usable", "false")) and row.get("leakage_group_id")
        }
    )
    insufficient = usable_group_count < minimum
    if insufficient:
        _append_issue(
            warnings,
            "insufficient_groups_for_formal_split",
            f"至少需要 {minimum} 个完整独立组，实际 {usable_group_count}",
        )
    preprocessing = config["preprocessing"]
    deterministic = (
        preprocessing.get("validation", {}).get("random") is False
        and preprocessing.get("test", {}).get("random") is False
    )
    if not deterministic:
        _append_issue(fatals, "nondeterministic_evaluation_preprocessing", "val/test random 必须为 false")

    registration_rows = _build_registration_rows(root, samples, config)
    threshold = int(config["audit"].get("near_duplicate_hamming_threshold", 5))
    duplicate_stats: dict[str, int] = {}
    top_k = int(config["audit"].get("near_duplicate_top_k_per_sample", 5))
    duplicate_rows = _build_near_duplicate_rows(
        samples,
        split_lookup,
        threshold,
        top_k_per_sample=top_k,
        stats=duplicate_stats,
    )
    split_category_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, ids in split_ids.items():
        pair_counts = Counter(id_to_row[sample_id]["category"] for sample_id in ids if sample_id in id_to_row)
        groups_by_category: dict[str, set[str]] = defaultdict(set)
        for sample_id in ids:
            if sample_id in id_to_row:
                row = id_to_row[sample_id]
                groups_by_category[row["category"]].add(row["leakage_group_id"])
        split_category_counts[split] = {
            category: {"pairs": pair_counts[category], "groups": len(groups_by_category[category])}
            for category in sorted(set(pair_counts) | set(groups_by_category))
        }
    if split_ids:
        all_categories = sorted(category_groups)
        for split in SPLIT_NAMES:
            for category in all_categories:
                values = split_category_counts.get(split, {}).get(
                    category, {"pairs": 0, "groups": 0}
                )
                if values["groups"] < 1:
                    _append_issue(
                        fatals,
                        "missing_category_group_in_split",
                        f"{split} 缺少类别 {category} 的独立组",
                    )
    audit = {
        "schema_version": int(config["schema_version"]),
        "audit_level": "full",
        "formal_ready": not fatals and not insufficient,
        "fatal_count": len(fatals),
        "warning_count": len(warnings),
        "fatals": fatals,
        "warnings": warnings,
        "sample_count": len(samples),
        "usable_sample_count": sum(parse_bool(row.get("usable", "false")) for row in samples),
        "group_count": len(groups),
        "experiment_count": len(experiments),
        "category_pair_counts": dict(sorted(category_pairs.items())),
        "category_group_counts": {
            category: len(values) for category, values in sorted(category_groups.items())
        },
        "split_category_counts": split_category_counts,
        "dataset_manifest_sha256": canonical_rows_hash(samples, SAMPLE_FIELDS),
        "evaluation_preprocessing_deterministic": deterministic,
        "near_duplicate_candidate_count": len(duplicate_rows),
        "near_duplicate_scope": "cross_split_top_k_per_sample",
        "near_duplicate_stats": duplicate_stats,
        "registration_manual_review_count": sum(
            row["review_status"] == "manual_review" for row in registration_rows
        ),
    }
    if split_config:
        audit["split_version"] = split_config.get("split_version")
        audit["split_schema_version"] = split_config.get("split_schema_version")
        audit["formal"] = bool(split_config.get("formal"))
    return audit, registration_rows, duplicate_rows


def write_audit_outputs(
    data_root: str | Path,
    audit: Mapping[str, Any],
    registration_rows: list[dict[str, str]],
    duplicate_rows: list[dict[str, str]],
    config: Mapping[str, Any],
    *,
    split_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    root = Path(data_root).resolve()
    quality = root / str(config["paths"]["quality"])
    quality.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        quality / "registration.csv",
        registration_rows,
        REGISTRATION_FIELDS,
        overwrite=overwrite,
    )
    write_csv_atomic(
        quality / "near_duplicates.csv",
        duplicate_rows,
        NEAR_DUPLICATE_FIELDS,
        overwrite=overwrite,
    )
    audit_path = Path(split_dir) / "audit.json" if split_dir else quality / "audit.json"
    write_json_atomic(audit_path, audit, overwrite=overwrite)
