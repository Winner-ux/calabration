from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def workspace(self) -> Path:
        return self.root / "processing_workspace"

    @property
    def input_root(self) -> Path:
        return self.workspace / "input"

    @property
    def output_root(self) -> Path:
        return self.workspace / "output"

    @property
    def system_root(self) -> Path:
        return self.workspace / "_system"

    @property
    def calibration_source(self) -> Path:
        return self.root / "cabration_video"

    @property
    def experiment_source(self) -> Path:
        return self.root / "vedio"

    def ensure_system_dirs(self) -> None:
        for name in ("staging", "locks", "backups", "runs"):
            (self.system_root / name).mkdir(parents=True, exist_ok=True)
