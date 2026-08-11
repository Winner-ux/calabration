from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .models import VideoMetadata

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


class VideoError(RuntimeError):
    pass


def probe_video(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{path}")
    metadata = VideoMetadata(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    if (
        metadata.width <= 0
        or metadata.height <= 0
        or metadata.fps <= 0
        or metadata.frame_count <= 0
    ):
        raise VideoError(f"视频元数据无效：{path}")
    return metadata


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "filename": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"无法打开视频：{path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise VideoError(f"无法读取 {path.name} 的第 {index} 帧")
    return frame


def normalize_frame(
    frame: np.ndarray, target_size: tuple[int, int], rotate_portrait: bool = True
) -> np.ndarray:
    if rotate_portrait and frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if (frame.shape[1], frame.shape[0]) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame


def sharpness(frame: np.ndarray, score_width: int = 480) -> float:
    height, width = frame.shape[:2]
    if width > score_width:
        ratio = score_width / width
        frame = cv2.resize(
            frame,
            (score_width, max(1, int(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def joint_normalized_scores(
    ir_scores: Iterable[float], rgb_scores: Iterable[float]
) -> np.ndarray:
    ir = np.asarray(list(ir_scores), dtype=np.float64)
    rgb = np.asarray(list(rgb_scores), dtype=np.float64)
    if len(ir) == 0 or len(ir) != len(rgb):
        raise ValueError("清晰度序列为空或长度不一致")
    ir_scale = max(float(np.median(ir)), 1e-9)
    rgb_scale = max(float(np.median(rgb)), 1e-9)
    ir_norm = np.maximum(ir / ir_scale, 1e-9)
    rgb_norm = np.maximum(rgb / rgb_scale, 1e-9)
    return 2.0 / (1.0 / ir_norm + 1.0 / rgb_norm)


def select_bin_maxima(scores: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("抽帧数量必须大于 0")
    if len(scores) < count:
        raise ValueError(f"共同帧数 {len(scores)} 少于要求的 {count}")
    result: list[int] = []
    for indices in np.array_split(np.arange(len(scores)), count):
        local = int(np.argmax(scores[indices]))
        result.append(int(indices[local]))
    return result


def write_png(path: Path, frame: np.ndarray, compression: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
        raise VideoError(f"PNG 写入失败：{path}")


def normalize_video_h264(
    source: Path,
    destination: Path,
    metadata: VideoMetadata,
    target_size: tuple[int, int] = (1920, 1080),
    rotate_portrait: bool = True,
) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VideoError("缺少 imageio-ffmpeg，无法生成规范化视频") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    if rotate_portrait and metadata.height > metadata.width:
        filters.append("transpose=2")
    filters.append(f"scale={target_size[0]}:{target_size[1]}")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command, capture_output=True, text=True, creationflags=creation_flags
    )
    if completed.returncode != 0 or not destination.exists():
        tail = completed.stderr[-1000:] if completed.stderr else "未知错误"
        raise VideoError(f"视频规范化失败：{source.name}\n{tail}")
