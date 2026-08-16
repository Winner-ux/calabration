from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskType = Literal["calibration", "experiment"]
CATEGORIES = ("health", "health+sick", "sick")


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duration_seconds"] = round(self.duration_seconds, 6)
        return result


@dataclass
class PairSpec:
    job_type: TaskType
    day: str
    experiment: str
    clip_id: str
    ir_path: Path
    rgb_path: Path
    category: str | None = None
    confirmed: bool = False
    status: str = "待确认"
    warnings: list[str] = field(default_factory=list)
    ir_metadata: VideoMetadata | None = None
    rgb_metadata: VideoMetadata | None = None

    @property
    def unit_name(self) -> str:
        clip = f"clip{int(self.clip_id):02d}"
        return f"{self.category}/{clip}" if self.category else clip


@dataclass(frozen=True)
class ProcessMessage:
    task: str
    status: Literal["success", "skipped", "failed", "warning"]
    message: str
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
