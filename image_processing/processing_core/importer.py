from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .discovery import (
    normalize_clip,
    normalize_day,
    normalize_experiment,
    validate_experiment_pair,
)
from .manifest import load_yaml, read_task_pairs, save_yaml, validate_task
from .models import CATEGORIES, PairSpec
from .paths import ProjectPaths
from .video import fingerprint, probe_video, sha256_file


class ImportConflict(RuntimeError):
    pass


def _relative_video_path(pair: PairSpec, modality: str, suffix: str) -> Path:
    filename = f"{normalize_clip(pair.clip_id)}{suffix.lower()}"
    if pair.job_type == "calibration":
        return Path(modality) / filename
    if pair.category not in CATEGORIES:
        raise ValueError("实验类别无效")
    return Path(pair.category) / modality / filename


def _task_dir(paths: ProjectPaths, pair: PairSpec) -> Path:
    return (
        paths.input_root
        / pair.job_type
        / normalize_day(pair.day)
        / normalize_experiment(pair.experiment)
    )


def _existing_pairs(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    data = load_yaml(manifest_path)
    if int(data.get("schema_version", 1)) == 2:
        return list(data.get("pairs") or [])
    job_type = data.get("job_type")
    if job_type not in {"calibration", "experiment"}:
        return []
    migrated: list[dict[str, Any]] = []
    for pair in read_task_pairs(manifest_path.parent, job_type):
        migrated.append(
            {
                "clip_id": normalize_clip(pair.clip_id),
                "category": pair.category,
                "paths": {
                    "ir": pair.ir_path.relative_to(manifest_path.parent).as_posix(),
                    "rgb": pair.rgb_path.relative_to(manifest_path.parent).as_posix(),
                },
                "sources": {
                    "ir": str(pair.ir_path.resolve()),
                    "rgb": str(pair.rgb_path.resolve()),
                },
                "fingerprints": {
                    "ir": fingerprint(pair.ir_path),
                    "rgb": fingerprint(pair.rgb_path),
                },
                "user_confirmed": True,
                "confirmed_at": "migrated_from_schema_v1",
                "warnings": ["从 schema v1 兼容迁移；原始来源路径不可用"],
            }
        )
    return migrated


def import_confirmed_pairs(paths: ProjectPaths, pairs: list[PairSpec]) -> list[str]:
    if not pairs:
        raise ValueError("没有可导入的视频对")
    paths.ensure_system_dirs()
    messages: list[str] = []
    grouped: dict[Path, list[PairSpec]] = {}
    for pair in pairs:
        if not pair.confirmed:
            raise ValueError(f"{pair.unit_name} 尚未确认")
        if pair.job_type == "experiment":
            errors = validate_experiment_pair(pair)
            if errors:
                raise ValueError(
                    f"{pair.unit_name} 不符合实验成片要求：\n" + "\n".join(errors)
                )
        else:
            pair.ir_metadata = pair.ir_metadata or probe_video(pair.ir_path)
            pair.rgb_metadata = pair.rgb_metadata or probe_video(pair.rgb_path)
        grouped.setdefault(_task_dir(paths, pair), []).append(pair)
    for task_dir, task_pairs in grouped.items():
        stage = paths.system_root / "staging" / f"import_{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        manifest_path = task_dir / "job.yaml"
        ready_path = task_dir / "READY"
        original_manifest = (
            manifest_path.read_bytes() if manifest_path.exists() else None
        )
        original_ready = ready_path.read_bytes() if ready_path.exists() else None
        records = _existing_pairs(manifest_path)
        record_keys = {(r.get("category"), str(r.get("clip_id"))) for r in records}
        planned: list[tuple[Path, Path, Path]] = []
        committed: list[Path] = []
        try:
            for pair in task_pairs:
                ir_rel = _relative_video_path(pair, "ir", pair.ir_path.suffix)
                rgb_rel = _relative_video_path(pair, "rgb", pair.rgb_path.suffix)
                ir_dest, rgb_dest = task_dir / ir_rel, task_dir / rgb_rel
                source_fingerprints = {
                    "ir": fingerprint(pair.ir_path),
                    "rgb": fingerprint(pair.rgb_path),
                }
                for source, destination, relative, modality in (
                    (pair.ir_path, ir_dest, ir_rel, "IR"),
                    (pair.rgb_path, rgb_dest, rgb_rel, "RGB"),
                ):
                    if destination.exists():
                        if (
                            sha256_file(destination)
                            != source_fingerprints[modality.lower()]["sha256"]
                        ):
                            raise ImportConflict(
                                f"目标文件已存在且内容不同：{destination}"
                            )
                        messages.append(f"已存在且一致，跳过复制：{destination}")
                        continue
                    staged = stage / relative
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, staged)
                    if (
                        sha256_file(staged)
                        != source_fingerprints[modality.lower()]["sha256"]
                    ):
                        raise ImportConflict(f"复制校验失败：{source}")
                    planned.append((staged, destination, relative))
                key = (pair.category, normalize_clip(pair.clip_id))
                record = {
                    "clip_id": normalize_clip(pair.clip_id),
                    "category": pair.category,
                    "paths": {"ir": ir_rel.as_posix(), "rgb": rgb_rel.as_posix()},
                    "sources": {
                        "ir": str(pair.ir_path.resolve()),
                        "rgb": str(pair.rgb_path.resolve()),
                    },
                    "fingerprints": source_fingerprints,
                    "user_confirmed": True,
                    "confirmed_at": datetime.now().isoformat(timespec="seconds"),
                    "warnings": list(pair.warnings),
                }
                if key in record_keys:
                    records = [
                        r
                        for r in records
                        if (r.get("category"), str(r.get("clip_id"))) != key
                    ]
                records.append(record)
                record_keys.add(key)
            for staged, destination, _ in planned:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)
                committed.append(destination)
            first = task_pairs[0]
            manifest = {
                "schema_version": 2,
                "job_type": first.job_type,
                "day": normalize_day(first.day),
                "experiment": normalize_experiment(first.experiment),
                "pairing": "explicit_user_confirmed_manifest",
            }
            if first.job_type == "experiment":
                manifest["categories"] = list(CATEGORIES)
            manifest["pairs"] = sorted(
                records,
                key=lambda r: (str(r.get("category") or ""), str(r.get("clip_id"))),
            )
            save_yaml(manifest_path, manifest)
            errors = validate_task(task_dir, first.job_type)
            if errors:
                raise ImportConflict("导入后校验失败：\n" + "\n".join(errors))
            ready_path.write_text(
                f"validated_at: {datetime.now().isoformat(timespec='seconds')}\n",
                encoding="utf-8",
            )
            messages.append(f"导入并验证完成：{task_dir}")
        except Exception:
            for destination in committed:
                destination.unlink(missing_ok=True)
            if original_manifest is None:
                manifest_path.unlink(missing_ok=True)
            else:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_bytes(original_manifest)
            if original_ready is None:
                ready_path.unlink(missing_ok=True)
            else:
                ready_path.parent.mkdir(parents=True, exist_ok=True)
                ready_path.write_bytes(original_ready)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return messages
