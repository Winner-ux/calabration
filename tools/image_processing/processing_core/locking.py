from __future__ import annotations

import json
import os
from datetime import datetime

from .paths import ProjectPaths


class ProcessingLocked(RuntimeError):
    pass


class GlobalProcessingLock:
    def __init__(self, paths: ProjectPaths, owner: str):
        self.paths = paths
        self.owner = owner
        self.path = paths.system_root / "locks" / "processing.lock"
        self.acquired = False

    def __enter__(self) -> "GlobalProcessingLock":
        self.paths.ensure_system_dirs()
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "owner": self.owner,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            detail = (
                self.path.read_text(encoding="utf-8", errors="replace")
                if self.path.exists()
                else ""
            )
            raise ProcessingLocked(
                f"已有处理任务正在运行。\n锁文件：{self.path}\n{detail}"
            ) from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self.acquired = False
