from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .discovery import normalize_clip, normalize_day, normalize_experiment
from .models import CATEGORIES, PairSpec, TaskType
from .video import VIDEO_EXTENSIONS, VideoError, probe_video, sha256_file


class ManifestError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"无法读取任务清单：{path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"任务清单格式错误：{path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _find_numeric_pairs(
    task_dir: Path, job_type: TaskType, category: str | None = None
) -> list[PairSpec]:
    base = task_dir / category if category else task_dir
    ir_dir, rgb_dir = base / "ir", base / "rgb"
    if not ir_dir.is_dir() or not rgb_dir.is_dir():
        return []
    ir_files = [
        p
        for p in ir_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    rgb_files = [
        p
        for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    non_numeric = [p.name for p in ir_files + rgb_files if not p.stem.isdigit()]
    if non_numeric:
        raise ManifestError(f"input 中存在非数字视频文件名：{non_numeric}")
    ir_groups: dict[int, list[Path]] = {}
    rgb_groups: dict[int, list[Path]] = {}
    for path in ir_files:
        ir_groups.setdefault(int(path.stem), []).append(path)
    for path in rgb_files:
        rgb_groups.setdefault(int(path.stem), []).append(path)
    duplicate_ids = [
        number
        for number in ir_groups.keys() | rgb_groups.keys()
        if len(ir_groups.get(number, [])) != 1 or len(rgb_groups.get(number, [])) != 1
    ]
    if duplicate_ids:
        formatted = [f"{number:02d}" for number in sorted(duplicate_ids)]
        raise ManifestError(f"IR/RGB 编号缺失或重复：{formatted}")
    ir = {number: paths[0] for number, paths in ir_groups.items()}
    rgb = {number: paths[0] for number, paths in rgb_groups.items()}
    if ir.keys() != rgb.keys():
        missing_ir = sorted(rgb.keys() - ir.keys())
        missing_rgb = sorted(ir.keys() - rgb.keys())
        raise ManifestError(
            f"IR/RGB 编号集合不一致；缺 IR={missing_ir}，缺 RGB={missing_rgb}"
        )
    pairs: list[PairSpec] = []
    for number in sorted(ir.keys() & rgb.keys()):
        pairs.append(
            PairSpec(
                job_type=job_type,
                day=task_dir.parent.name,
                experiment=task_dir.name,
                category=category,
                clip_id=f"{number:02d}",
                ir_path=ir[number],
                rgb_path=rgb[number],
                confirmed=True,
                status="已就绪",
            )
        )
    return pairs


def read_task_pairs(task_dir: Path, expected_type: TaskType) -> list[PairSpec]:
    manifest_path = task_dir / "job.yaml"
    if not manifest_path.exists():
        raise ManifestError(f"缺少 job.yaml：{task_dir}")
    data = load_yaml(manifest_path)
    if data.get("job_type") != expected_type:
        raise ManifestError(f"任务类型与程序不一致：{manifest_path}")
    day = normalize_day(str(data.get("day", "")))
    experiment = normalize_experiment(str(data.get("experiment", "")))
    if task_dir.parent.name != day or task_dir.name != experiment:
        raise ManifestError(f"job.yaml 的日期/实验号与目录不一致：{task_dir}")
    schema = int(data.get("schema_version", 1))
    if schema == 1:
        if expected_type == "calibration":
            pairs = _find_numeric_pairs(task_dir, expected_type)
        else:
            pairs = []
            categories = data.get("categories", list(CATEGORIES))
            for category in categories:
                if category in CATEGORIES:
                    pairs.extend(_find_numeric_pairs(task_dir, expected_type, category))
        if not pairs:
            raise ManifestError(f"没有找到同编号 RGB/IR 视频对：{task_dir}")
        return pairs
    if schema != 2:
        raise ManifestError(f"不支持的 schema_version={schema}：{manifest_path}")
    raw_pairs = data.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ManifestError(f"schema v2 清单缺少 pairs：{manifest_path}")
    result: list[PairSpec] = []
    seen: set[tuple[str | None, str]] = set()
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise ManifestError(f"pairs 中存在无效记录：{manifest_path}")
        clip_id = normalize_clip(str(item.get("clip_id", "")))
        category = item.get("category")
        if expected_type == "experiment" and category not in CATEGORIES:
            raise ManifestError(f"无效实验类别 {category!r}：{manifest_path}")
        if expected_type == "calibration":
            category = None
        key = (category, clip_id)
        if key in seen:
            raise ManifestError(f"重复 clip：{key}")
        seen.add(key)
        if item.get("user_confirmed") is not True:
            raise ManifestError(f"clip{clip_id} 尚未人工确认")
        paths = item.get("paths") or {}
        ir_path = (task_dir / str(paths.get("ir", ""))).resolve()
        rgb_path = (task_dir / str(paths.get("rgb", ""))).resolve()
        task_resolved = task_dir.resolve()
        if (
            task_resolved not in ir_path.parents
            or task_resolved not in rgb_path.parents
        ):
            raise ManifestError("清单视频路径越出任务目录")
        pair = PairSpec(
            job_type=expected_type,
            day=day,
            experiment=experiment,
            category=category,
            clip_id=clip_id,
            ir_path=ir_path,
            rgb_path=rgb_path,
            confirmed=True,
            status="已就绪",
        )
        result.append(pair)
    return result


def validate_task(task_dir: Path, expected_type: TaskType) -> list[str]:
    errors: list[str] = []
    try:
        pairs = read_task_pairs(task_dir, expected_type)
    except (ManifestError, ValueError) as exc:
        return [str(exc)]
    for pair in pairs:
        for modality, path in (("IR", pair.ir_path), ("RGB", pair.rgb_path)):
            if not path.is_file():
                errors.append(f"{pair.unit_name} 缺少 {modality}：{path}")
                continue
            try:
                metadata = probe_video(path)
                if modality == "IR":
                    pair.ir_metadata = metadata
                else:
                    pair.rgb_metadata = metadata
            except VideoError as exc:
                errors.append(str(exc))
    manifest = load_yaml(task_dir / "job.yaml")
    if int(manifest.get("schema_version", 1)) == 2:
        records = {
            (item.get("category"), normalize_clip(str(item.get("clip_id", "")))): item
            for item in manifest.get("pairs", [])
            if isinstance(item, dict)
        }
        for pair in pairs:
            record = records.get((pair.category, normalize_clip(pair.clip_id)), {})
            expected = record.get("fingerprints") or {}
            for modality, path in (("ir", pair.ir_path), ("rgb", pair.rgb_path)):
                expected_hash = (expected.get(modality) or {}).get("sha256")
                if (
                    expected_hash
                    and path.is_file()
                    and sha256_file(path) != expected_hash
                ):
                    errors.append(
                        f"{pair.unit_name} 的 {modality.upper()} 文件内容与导入清单不一致"
                    )
    return errors


def find_ready_tasks(input_root: Path, job_type: TaskType) -> list[Path]:
    base = input_root / job_type
    if not base.exists():
        return []
    return sorted(
        (
            marker.parent
            for marker in base.glob("day*/experiment*/READY")
            if (marker.parent / "job.yaml").exists()
        ),
        key=lambda p: (p.parent.name, p.name),
    )
