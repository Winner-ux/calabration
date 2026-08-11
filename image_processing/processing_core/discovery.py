from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .models import CATEGORIES, PairSpec, TaskType
from .video import VIDEO_EXTENSIONS, VideoError, probe_video, sha256_file

DAY_RE = re.compile(r"^day(\d+)$", re.IGNORECASE)
EXPERIMENT_RE = re.compile(r"^(?:experiment)?(\d+)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\d+$")


def normalize_day(value: str) -> str:
    match = DAY_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"日期目录必须是 dayN：{value}")
    return f"day{int(match.group(1))}"


def normalize_experiment(value: str) -> str:
    match = EXPERIMENT_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"实验号必须是数字或 experimentN：{value}")
    return f"experiment{int(match.group(1))}"


def normalize_clip(value: str) -> str:
    text = value.strip()
    text = text[4:] if text.lower().startswith("clip") else text
    if not NUMERIC_RE.fullmatch(text):
        raise ValueError(f"clip 编号必须是数字：{value}")
    return f"{int(text):02d}"


def _identity(
    relative_parent: Path, job_type: TaskType
) -> tuple[str, str, str | None] | None:
    parts = relative_parent.parts
    day_index = next(
        (i for i, part in enumerate(parts) if DAY_RE.fullmatch(part)), None
    )
    if day_index is None:
        return None
    day = normalize_day(parts[day_index])
    following = list(parts[day_index + 1 :])
    experiment = "experiment1"
    if following and EXPERIMENT_RE.fullmatch(following[0]):
        experiment = normalize_experiment(following.pop(0))
    category = next((part for part in following if part in CATEGORIES), None)
    if job_type == "experiment" and category is None:
        return None
    return day, experiment, category


def _video_files(directory: Path) -> list[Path]:
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda p: p.name.lower(),
    )


def discover_numeric_pairs(
    source_root: Path, job_type: TaskType
) -> tuple[list[PairSpec], list[str]]:
    pairs: list[PairSpec] = []
    issues: list[str] = []
    accounted_files: set[Path] = set()
    if not source_root.exists():
        return pairs, [f"源目录不存在：{source_root}"]
    groups: dict[Path, dict[str, Path]] = defaultdict(dict)
    for modality_dir in source_root.rglob("*"):
        if modality_dir.is_dir() and modality_dir.name.lower() in {"ir", "rgb"}:
            groups[modality_dir.parent][modality_dir.name.lower()] = modality_dir
    for parent, modalities in sorted(
        groups.items(), key=lambda item: str(item[0]).lower()
    ):
        identity = _identity(parent.relative_to(source_root), job_type)
        if identity is None:
            issues.append(f"无法识别任务身份：{parent}")
            continue
        day, experiment, category = identity
        if set(modalities) != {"ir", "rgb"}:
            issues.append(f"缺少 IR 或 RGB 目录：{parent}")
            continue
        ir_files = _video_files(modalities["ir"])
        rgb_files = _video_files(modalities["rgb"])
        accounted_files.update(ir_files)
        accounted_files.update(rgb_files)
        ir_numeric: dict[int, list[Path]] = defaultdict(list)
        rgb_numeric: dict[int, list[Path]] = defaultdict(list)
        for path in ir_files:
            if NUMERIC_RE.fullmatch(path.stem):
                ir_numeric[int(path.stem)].append(path)
        for path in rgb_files:
            if NUMERIC_RE.fullmatch(path.stem):
                rgb_numeric[int(path.stem)].append(path)
        paired_paths: set[Path] = set()
        for number in sorted(ir_numeric.keys() & rgb_numeric.keys()):
            if len(ir_numeric[number]) != 1 or len(rgb_numeric[number]) != 1:
                issues.append(
                    f"编号 {number:02d} 存在多个候选，禁止自动配对："
                    f"IR={[p.name for p in ir_numeric[number]]}，"
                    f"RGB={[p.name for p in rgb_numeric[number]]}"
                )
                continue
            ir_path, rgb_path = ir_numeric[number][0], rgb_numeric[number][0]
            paired_paths.update((ir_path, rgb_path))
            pair = PairSpec(
                job_type=job_type,
                day=day,
                experiment=experiment,
                category=category,
                clip_id=f"{number:02d}",
                ir_path=ir_path,
                rgb_path=rgb_path,
                status="待人工确认",
            )
            try:
                pair.ir_metadata = probe_video(pair.ir_path)
                pair.rgb_metadata = probe_video(pair.rgb_path)
            except VideoError as exc:
                pair.status = "视频无效"
                pair.warnings.append(str(exc))
            pairs.append(pair)
        unmatched_ir = [p.name for p in ir_files if p not in paired_paths]
        unmatched_rgb = [p.name for p in rgb_files if p not in paired_paths]
        if unmatched_ir or unmatched_rgb:
            issues.append(
                f"{parent} 存在不能自动配对的文件；IR={unmatched_ir or '无'}，RGB={unmatched_rgb or '无'}"
            )
    unstructured = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path not in accounted_files
        ),
        key=lambda path: str(path).lower(),
    )
    if unstructured:
        grouped_unstructured: dict[Path, list[str]] = defaultdict(list)
        for path in unstructured:
            grouped_unstructured[path.parent].append(path.name)
        for parent, names in grouped_unstructured.items():
            issues.append(f"视频不在可识别的 IR/RGB 分类结构中：{parent}，文件={names}")
    if job_type == "calibration":
        by_size: dict[int, list[tuple[PairSpec, str, Path]]] = defaultdict(list)
        for pair in pairs:
            for modality, path in (("IR", pair.ir_path), ("RGB", pair.rgb_path)):
                by_size[path.stat().st_size].append((pair, modality, path))
        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            by_hash: dict[str, list[tuple[PairSpec, str, Path]]] = defaultdict(list)
            for item in same_size:
                by_hash[sha256_file(item[2])].append(item)
            for duplicates in by_hash.values():
                identities = {(item[0].day, item[0].experiment) for item in duplicates}
                if len(duplicates) > 1 and len(identities) > 1:
                    paths_text = "、".join(str(item[2]) for item in duplicates)
                    warning = f"跨任务重复视频（允许导入但会记录）：{paths_text}"
                    issues.append(warning)
                    for pair, _, _ in duplicates:
                        if warning not in pair.warnings:
                            pair.warnings.append(warning)
    return pairs, issues


def validate_experiment_pair(
    pair: PairSpec,
    target_size: tuple[int, int] = (1920, 1080),
    fps_tolerance: float = 0.001,
) -> list[str]:
    errors: list[str] = []
    if pair.category not in CATEGORIES:
        errors.append("实验类别必须是 health、health+sick 或 sick")
    try:
        ir = pair.ir_metadata or probe_video(pair.ir_path)
        rgb = pair.rgb_metadata or probe_video(pair.rgb_path)
    except VideoError as exc:
        return [str(exc)]
    if (ir.width, ir.height) != target_size:
        errors.append(
            f"IR 尺寸必须为 {target_size[0]}×{target_size[1]}，当前为 {ir.width}×{ir.height}"
        )
    if (rgb.width, rgb.height) != target_size:
        errors.append(
            f"RGB 尺寸必须为 {target_size[0]}×{target_size[1]}，当前为 {rgb.width}×{rgb.height}"
        )
    if abs(ir.fps - rgb.fps) > fps_tolerance:
        errors.append(f"FPS 不一致：IR={ir.fps:.6f}，RGB={rgb.fps:.6f}")
    return errors
