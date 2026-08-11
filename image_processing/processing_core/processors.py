from __future__ import annotations

import csv
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .discovery import validate_experiment_pair
from .locking import GlobalProcessingLock
from .manifest import find_ready_tasks, read_task_pairs, validate_task
from .models import PairSpec, ProcessMessage
from .outputs import (
    commit_staged_output,
    create_stage,
    existing_output_complete,
    write_json,
    write_run_report,
)
from .paths import ProjectPaths
from .video import (
    VideoError,
    fingerprint,
    joint_normalized_scores,
    normalize_frame,
    normalize_video_h264,
    probe_video,
    read_frame,
    select_bin_maxima,
    sharpness,
    write_png,
)

ProgressCallback = Callable[[str], None]


class ProcessingCancelled(RuntimeError):
    pass


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event and cancel_event.is_set():
        raise ProcessingCancelled("用户已取消处理")


def _write_csv(
    path: Path, fieldnames: list[str], records: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _preview_pair(ir: np.ndarray, rgb: np.ndarray, width: int = 480) -> np.ndarray:
    def fit(frame: np.ndarray) -> np.ndarray:
        ratio = width / frame.shape[1]
        return cv2.resize(
            frame,
            (width, max(1, int(frame.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )

    ir_small, rgb_small = fit(ir), fit(rgb)
    height = min(ir_small.shape[0], rgb_small.shape[0])
    ir_small, rgb_small = ir_small[:height], rgb_small[:height]
    cv2.putText(ir_small, "IR", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(
        rgb_small, "RGB", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    return cv2.hconcat([ir_small, rgb_small])


def _write_preview(path: Path, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if not pairs:
        raise VideoError("没有可用于预览的图像")
    rows = [_preview_pair(ir.copy(), rgb.copy()) for ir, rgb in pairs[:4]]
    target_width = max(row.shape[1] for row in rows)
    padded: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] < target_width:
            row = cv2.copyMakeBorder(
                row, 0, 0, 0, target_width - row.shape[1], cv2.BORDER_CONSTANT
            )
        padded.append(row)
    write_png(path, cv2.vconcat(padded))


def _validate_png_sets(ir_dir: Path, rgb_dir: Path, expected: int) -> dict[str, object]:
    ir_files, rgb_files = sorted(ir_dir.glob("*.png")), sorted(rgb_dir.glob("*.png"))
    ir_names, rgb_names = {p.name for p in ir_files}, {p.name for p in rgb_files}
    readable = len(ir_files) == len(rgb_files) == expected
    if readable:
        readable = all(
            cv2.imread(str(path)) is not None for path in ir_files + rgb_files
        )
    return {
        "exact_count_each": len(ir_files) == len(rgb_files) == expected,
        "matching_filename_sets": ir_names == rgb_names,
        "all_images_readable": readable,
    }


def _calibration_scores(
    pair: PairSpec, cancel_event: threading.Event | None
) -> tuple[list[dict[str, object]], np.ndarray]:
    ir_meta = pair.ir_metadata or probe_video(pair.ir_path)
    rgb_meta = pair.rgb_metadata or probe_video(pair.rgb_path)
    common_duration = min(ir_meta.duration_seconds, rgb_meta.duration_seconds)
    base_fps = min(ir_meta.fps, rgb_meta.fps)
    candidate_count = max(1, int(np.floor(common_duration * base_fps)))
    ir_cap, rgb_cap = (
        cv2.VideoCapture(str(pair.ir_path)),
        cv2.VideoCapture(str(pair.rgb_path)),
    )
    records: list[dict[str, object]] = []
    ir_scores: list[float] = []
    rgb_scores: list[float] = []
    try:
        for index in range(candidate_count):
            if index % 25 == 0:
                _check_cancel(cancel_event)
            timestamp = index / base_fps
            ir_index = min(ir_meta.frame_count - 1, int(round(timestamp * ir_meta.fps)))
            rgb_index = min(
                rgb_meta.frame_count - 1, int(round(timestamp * rgb_meta.fps))
            )
            ir_cap.set(cv2.CAP_PROP_POS_FRAMES, ir_index)
            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_index)
            ir_ok, ir_frame = ir_cap.read()
            rgb_ok, rgb_frame = rgb_cap.read()
            if not ir_ok or not rgb_ok:
                continue
            ir_value, rgb_value = sharpness(ir_frame), sharpness(rgb_frame)
            ir_scores.append(ir_value)
            rgb_scores.append(rgb_value)
            records.append(
                {
                    "ir_source_frame_0_based": ir_index,
                    "ir_timestamp_seconds": ir_index / ir_meta.fps,
                    "ir_sharpness": ir_value,
                    "rgb_source_frame_0_based": rgb_index,
                    "rgb_timestamp_seconds": rgb_index / rgb_meta.fps,
                    "rgb_sharpness": rgb_value,
                }
            )
    finally:
        ir_cap.release()
        rgb_cap.release()
    if not records:
        raise VideoError(f"{pair.unit_name} 没有可同步读取的帧")
    return records, joint_normalized_scores(ir_scores, rgb_scores)


def _process_calibration_task(
    paths: ProjectPaths,
    task_dir: Path,
    count: int,
    force: bool,
    progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
    target_size: tuple[int, int],
) -> ProcessMessage:
    task_name = f"calibration/{task_dir.parent.name}/{task_dir.name}"
    target = paths.output_root / "calibration" / task_dir.parent.name / task_dir.name
    if existing_output_complete(target, "calibration") and not force:
        return ProcessMessage(
            task_name, "skipped", "已有完整结果，默认不覆盖", str(target)
        )
    if target.exists() and not force:
        return ProcessMessage(
            task_name, "failed", "已有不完整结果；请核对后勾选重新处理", str(target)
        )
    errors = validate_task(task_dir, "calibration")
    if errors:
        return ProcessMessage(task_name, "failed", "；".join(errors))
    pairs = read_task_pairs(task_dir, "calibration")
    stage_root = create_stage(paths, "calibration")
    stage = stage_root / task_dir.parent.name / task_dir.name
    stage.mkdir(parents=True)
    all_records: list[dict[str, object]] = []
    clip_summaries: dict[str, object] = {}
    try:
        for pair in pairs:
            _check_cancel(cancel_event)
            _notify(progress, f"{task_name}：分析 {pair.unit_name} 清晰度")
            pair.ir_metadata = probe_video(pair.ir_path)
            pair.rgb_metadata = probe_video(pair.rgb_path)
            candidates, scores = _calibration_scores(pair, cancel_event)
            selected = select_bin_maxima(scores, count)
            preview_frames: list[tuple[np.ndarray, np.ndarray]] = []
            clip_records: list[dict[str, object]] = []
            for sequence, candidate_index in enumerate(selected, start=1):
                candidate = candidates[candidate_index]
                ir_frame = normalize_frame(
                    read_frame(pair.ir_path, int(candidate["ir_source_frame_0_based"])),
                    target_size,
                    rotate_portrait=False,
                )
                rgb_frame = normalize_frame(
                    read_frame(
                        pair.rgb_path, int(candidate["rgb_source_frame_0_based"])
                    ),
                    target_size,
                    rotate_portrait=True,
                )
                filename = f"clip{int(pair.clip_id):02d}_{sequence:03d}.png"
                write_png(stage / "images" / "ir" / filename, ir_frame)
                write_png(stage / "images" / "rgb" / filename, rgb_frame)
                preview_frames.append((ir_frame, rgb_frame))
                record = {
                    "clip": f"clip{int(pair.clip_id):02d}",
                    "sequence": sequence,
                    "filename": filename,
                    **candidate,
                    "timestamp_delta_seconds": abs(
                        float(candidate["ir_timestamp_seconds"])
                        - float(candidate["rgb_timestamp_seconds"])
                    ),
                }
                clip_records.append(record)
                all_records.append(record)
            _write_preview(
                stage / "preview" / f"clip{int(pair.clip_id):02d}.png", preview_frames
            )
            _notify(progress, f"{task_name}：生成 {pair.unit_name} 规范化视频")
            normalize_video_h264(
                pair.ir_path,
                stage / "videos" / "ir" / f"ir_{int(pair.clip_id):02d}.mp4",
                pair.ir_metadata,
                target_size,
                rotate_portrait=False,
            )
            normalize_video_h264(
                pair.rgb_path,
                stage / "videos" / "rgb" / f"rgb_{int(pair.clip_id):02d}.mp4",
                pair.rgb_metadata,
                target_size,
                rotate_portrait=True,
            )
            clip_summaries[pair.clip_id] = {
                "clip_id": pair.clip_id,
                "count": count,
                "ir_source": str(pair.ir_path.resolve()),
                "rgb_source": str(pair.rgb_path.resolve()),
                "ir_fingerprint": fingerprint(pair.ir_path),
                "rgb_fingerprint": fingerprint(pair.rgb_path),
                "ir_metadata": pair.ir_metadata.to_dict(),
                "rgb_metadata": pair.rgb_metadata.to_dict(),
                "records": clip_records,
            }
        fields = [
            "clip",
            "sequence",
            "filename",
            "ir_source_frame_0_based",
            "ir_timestamp_seconds",
            "ir_sharpness",
            "rgb_source_frame_0_based",
            "rgb_timestamp_seconds",
            "rgb_sharpness",
            "timestamp_delta_seconds",
        ]
        _write_csv(stage / "frames.csv", fields, all_records)
        validation = _validate_png_sets(
            stage / "images" / "ir", stage / "images" / "rgb", count * len(pairs)
        )
        if not all(validation.values()):
            raise VideoError(f"标定输出完整性校验失败：{validation}")
        write_json(
            stage / "summary.json",
            {
                "schema_version": 2,
                "status": "complete",
                "task": task_name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "parameters": {
                    "target_size": list(target_size),
                    "count_per_clip": count,
                    "selection_method": "synchronized_common_time_joint_sharpness_v2",
                },
                "clips": clip_summaries,
                "validation": validation,
            },
        )
        _check_cancel(cancel_event)
        backup = commit_staged_output(paths, stage, target, force)
        message = "标定任务完成"
        if backup:
            message += f"；旧结果已备份到 {backup}"
        return ProcessMessage(task_name, "success", message, str(target))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _experiment_scores(
    pair: PairSpec, common_frames: int, cancel_event: threading.Event | None
) -> tuple[np.ndarray, list[float], list[float]]:
    ir_cap, rgb_cap = (
        cv2.VideoCapture(str(pair.ir_path)),
        cv2.VideoCapture(str(pair.rgb_path)),
    )
    ir_scores: list[float] = []
    rgb_scores: list[float] = []
    try:
        for index in range(common_frames):
            if index % 50 == 0:
                _check_cancel(cancel_event)
            ir_ok, ir_frame = ir_cap.read()
            rgb_ok, rgb_frame = rgb_cap.read()
            if not ir_ok or not rgb_ok:
                raise VideoError(f"{pair.unit_name} 在第 {index} 帧提前结束")
            ir_scores.append(sharpness(ir_frame))
            rgb_scores.append(sharpness(rgb_frame))
    finally:
        ir_cap.release()
        rgb_cap.release()
    return joint_normalized_scores(ir_scores, rgb_scores), ir_scores, rgb_scores


def _process_experiment_pair(
    paths: ProjectPaths,
    pair: PairSpec,
    count: int,
    force: bool,
    progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
    target_size: tuple[int, int],
) -> ProcessMessage:
    clip_name = f"clip{int(pair.clip_id):02d}"
    task_name = f"experiment/{pair.day}/{pair.experiment}/{pair.category}/{clip_name}"
    target = (
        paths.output_root
        / "experiment"
        / pair.day
        / pair.experiment
        / str(pair.category)
        / clip_name
    )
    if existing_output_complete(target, "experiment") and not force:
        return ProcessMessage(
            task_name, "skipped", "已有完整结果，默认不覆盖", str(target)
        )
    if target.exists() and not force:
        return ProcessMessage(
            task_name, "failed", "已有不完整结果；请核对后勾选重新处理", str(target)
        )
    pair.ir_metadata, pair.rgb_metadata = (
        probe_video(pair.ir_path),
        probe_video(pair.rgb_path),
    )
    validation_errors = validate_experiment_pair(pair, target_size=target_size)
    if validation_errors:
        return ProcessMessage(task_name, "failed", "；".join(validation_errors))
    common_frames = min(pair.ir_metadata.frame_count, pair.rgb_metadata.frame_count)
    if common_frames < count:
        return ProcessMessage(
            task_name, "failed", f"共同帧数 {common_frames} 少于要求的 {count}"
        )
    stage_root = create_stage(paths, "experiment")
    stage = stage_root / pair.day / pair.experiment / str(pair.category) / clip_name
    stage.mkdir(parents=True)
    try:
        _notify(progress, f"{task_name}：分析 {common_frames} 个共同帧")
        combined, ir_scores, rgb_scores = _experiment_scores(
            pair, common_frames, cancel_event
        )
        selected = select_bin_maxima(combined, count)
        records: list[dict[str, object]] = []
        preview_frames: list[tuple[np.ndarray, np.ndarray]] = []
        ir_cap, rgb_cap = (
            cv2.VideoCapture(str(pair.ir_path)),
            cv2.VideoCapture(str(pair.rgb_path)),
        )
        try:
            if not ir_cap.isOpened() or not rgb_cap.isOpened():
                raise VideoError(f"{pair.unit_name} 无法重新打开以导出所选帧")
            for sequence, frame_index in enumerate(selected, start=1):
                if sequence % 50 == 0:
                    _check_cancel(cancel_event)
                    _notify(progress, f"{task_name}：导出 {sequence}/{count}")
                ir_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ir_ok, ir_frame = ir_cap.read()
                rgb_ok, rgb_frame = rgb_cap.read()
                if not ir_ok or not rgb_ok:
                    raise VideoError(
                        f"{pair.unit_name} 无法导出第 {frame_index} 个共同帧"
                    )
                filename = f"{sequence:06d}.png"
                write_png(stage / "ir" / filename, ir_frame)
                write_png(stage / "rgb" / filename, rgb_frame)
                if len(preview_frames) < 4 and sequence in {
                    1,
                    max(1, count // 3),
                    max(1, 2 * count // 3),
                    count,
                }:
                    preview_frames.append((ir_frame, rgb_frame))
                records.append(
                    {
                        "sequence": sequence,
                        "filename": filename,
                        "source_frame_0_based": frame_index,
                        "timestamp_seconds": frame_index / pair.ir_metadata.fps,
                        "ir_sharpness": ir_scores[frame_index],
                        "rgb_sharpness": rgb_scores[frame_index],
                        "combined_normalized_sharpness": float(combined[frame_index]),
                    }
                )
        finally:
            ir_cap.release()
            rgb_cap.release()
        _write_csv(
            stage / "frames.csv",
            [
                "sequence",
                "filename",
                "source_frame_0_based",
                "timestamp_seconds",
                "ir_sharpness",
                "rgb_sharpness",
                "combined_normalized_sharpness",
            ],
            records,
        )
        _write_preview(stage / "sync_preview.png", preview_frames)
        validation = _validate_png_sets(stage / "ir", stage / "rgb", count)
        if not all(validation.values()):
            raise VideoError(f"实验输出完整性校验失败：{validation}")
        write_json(
            stage / "summary.json",
            {
                "schema_version": 2,
                "status": "complete",
                "task": f"experiment/{pair.day}/{pair.experiment}",
                "category": pair.category,
                "clip": clip_name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "parameters": {
                    "count": count,
                    "fps_tolerance": 0.001,
                    "png_compression": 3,
                },
                "sources": {
                    "ir": {
                        "path": str(pair.ir_path.resolve()),
                        "fingerprint": fingerprint(pair.ir_path),
                        "metadata": pair.ir_metadata.to_dict(),
                    },
                    "rgb": {
                        "path": str(pair.rgb_path.resolve()),
                        "fingerprint": fingerprint(pair.rgb_path),
                        "metadata": pair.rgb_metadata.to_dict(),
                    },
                },
                "common_frames": common_frames,
                "common_duration_seconds": common_frames / pair.ir_metadata.fps,
                "first_selected_frame": selected[0],
                "last_selected_frame": selected[-1],
                "selection": "disjoint_temporal_bins_joint_normalized_sharpness",
                "validation": validation,
            },
        )
        _check_cancel(cancel_event)
        backup = commit_staged_output(paths, stage, target, force)
        message = f"已导出 {count} 对同步图片"
        if backup:
            message += f"；旧结果已备份到 {backup}"
        return ProcessMessage(task_name, "success", message, str(target))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def run_calibration_batch(
    paths: ProjectPaths,
    count: int = 4,
    force: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    target_size: tuple[int, int] = (1920, 1080),
) -> tuple[list[ProcessMessage], Path]:
    if count <= 0:
        raise ValueError("标定抽帧数量必须大于 0")
    results: list[ProcessMessage] = []
    with GlobalProcessingLock(paths, "calibration"):
        tasks = find_ready_tasks(paths.input_root, "calibration")
        if not tasks:
            results.append(
                ProcessMessage("calibration", "warning", "没有找到 READY 标定任务")
            )
        for task in tasks:
            try:
                results.append(
                    _process_calibration_task(
                        paths, task, count, force, progress, cancel_event, target_size
                    )
                )
            except ProcessingCancelled:
                results.append(
                    ProcessMessage(str(task), "warning", "用户已取消，未提交当前任务")
                )
                break
            except Exception as exc:
                results.append(ProcessMessage(str(task), "failed", str(exc)))
    report = write_run_report(
        paths,
        "calibration",
        {"count": count, "force": force},
        [r.to_dict() for r in results],
    )
    return results, report


def run_experiment_batch(
    paths: ProjectPaths,
    count: int = 750,
    force: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    target_size: tuple[int, int] = (1920, 1080),
) -> tuple[list[ProcessMessage], Path]:
    if count <= 0:
        raise ValueError("实验抽帧数量必须大于 0")
    results: list[ProcessMessage] = []
    with GlobalProcessingLock(paths, "experiment"):
        tasks = find_ready_tasks(paths.input_root, "experiment")
        if not tasks:
            results.append(
                ProcessMessage("experiment", "warning", "没有找到 READY 实验任务")
            )
        for task in tasks:
            errors = validate_task(task, "experiment")
            if errors:
                results.append(ProcessMessage(str(task), "failed", "；".join(errors)))
                continue
            for pair in read_task_pairs(task, "experiment"):
                try:
                    results.append(
                        _process_experiment_pair(
                            paths,
                            pair,
                            count,
                            force,
                            progress,
                            cancel_event,
                            target_size,
                        )
                    )
                except ProcessingCancelled:
                    results.append(
                        ProcessMessage(
                            pair.unit_name, "warning", "用户已取消，未提交当前任务"
                        )
                    )
                    break
                except Exception as exc:
                    results.append(ProcessMessage(pair.unit_name, "failed", str(exc)))
            if cancel_event and cancel_event.is_set():
                break
    report = write_run_report(
        paths,
        "experiment",
        {"count": count, "force": force},
        [r.to_dict() for r in results],
    )
    return results, report
