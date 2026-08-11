from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ProjectPaths


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def create_stage(paths: ProjectPaths, job_type: str) -> Path:
    paths.ensure_system_dirs()
    stage = paths.system_root / "staging" / f"{job_type}_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    return stage


def backup_existing(paths: ProjectPaths, target: Path) -> Path | None:
    if not target.exists():
        return None
    relative = target.relative_to(paths.output_root)
    backup = paths.system_root / "backups" / timestamp() / relative
    counter = 1
    candidate = backup
    while candidate.exists():
        candidate = backup.with_name(f"{backup.name}_{counter}")
        counter += 1
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(candidate))
    return candidate


def promote_directory(stage: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"目标输出已经存在：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(target)


def commit_staged_output(
    paths: ProjectPaths, stage: Path, target: Path, force: bool
) -> Path | None:
    """Promote a validated stage and restore the old output if promotion fails."""
    backup = backup_existing(paths, target) if force else None
    try:
        promote_directory(stage, target)
    except Exception:
        if backup and backup.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(target))
        raise
    return backup


def existing_output_complete(target: Path, job_type: str) -> bool:
    summary_path = target / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "complete":
        return False
    validation = summary.get("validation")
    if not isinstance(validation, dict) or not all(
        bool(value) for value in validation.values()
    ):
        return False
    if not (target / "frames.csv").is_file():
        return False
    if job_type == "calibration":
        ir_dir, rgb_dir = target / "images" / "ir", target / "images" / "rgb"
        if not (target / "preview").is_dir():
            return False
        if (
            not (target / "videos" / "ir").is_dir()
            or not (target / "videos" / "rgb").is_dir()
        ):
            return False
    else:
        ir_dir, rgb_dir = target / "ir", target / "rgb"
        if not (target / "sync_preview.png").is_file():
            return False
    if not ir_dir.is_dir() or not rgb_dir.is_dir():
        return False
    ir_names = {p.name for p in ir_dir.glob("*.png")}
    rgb_names = {p.name for p in rgb_dir.glob("*.png")}
    return bool(ir_names) and ir_names == rgb_names


def write_run_report(
    paths: ProjectPaths,
    job_type: str,
    parameters: dict[str, Any],
    results: list[dict[str, Any]],
) -> Path:
    report = paths.system_root / "runs" / f"run_{timestamp()}_{job_type}.json"
    write_json(
        report,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "job_type": job_type,
            "parameters": parameters,
            "results": results,
        },
    )
    return report
