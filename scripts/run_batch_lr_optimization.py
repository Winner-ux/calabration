"""Run the resumable batch-size / learning-rate optimization experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from data_pipeline.schema import DataContractError
from main.inference_utils import sha256_file
from scripts.batch_lr_experiment import (
    BASELINE_GROUP,
    GROUPS,
    W2_LOSS_WEIGHTS,
    candidate_matrix,
    latest_completed_run,
    matching_runs,
    memory_preflight_status,
    metric_count,
    rank_curves,
    read_json,
    summarize_curve,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "runs" / "batch_lr_optimization"
DEFAULT_INIT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "loss_weight_ablation"
    / "W2"
    / "seed42"
    / "train-20260814-142003-172869"
    / "last.pt"
)


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _gpu_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if completed.returncode:
            return None
        return max(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _run(
    command: list[str], log_path: Path, *, tolerate_oom: bool = False
) -> dict[str, Any]:
    _append_log(log_path, "COMMAND: " + subprocess.list2cmdline(command))
    peak_nvidia_mib = _gpu_memory_used_mib()
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_environment,
        )
        while process.poll() is None:
            used = _gpu_memory_used_mib()
            if used is not None:
                peak_nvidia_mib = max(peak_nvidia_mib or 0, used)
            time.sleep(0.1)
        returncode = int(process.returncode or 0)
    if returncode:
        tail = ""
        try:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            )
        except OSError:
            pass
        oom = "out of memory" in tail.lower()
        if not (tolerate_oom and oom):
            raise RuntimeError(
                f"命令失败 returncode={returncode}: {subprocess.list2cmdline(command)}"
            )
        return {"returncode": returncode, "oom": True, "peak_nvidia_smi_mib": peak_nvidia_mib}
    return {"returncode": 0, "oom": False, "peak_nvidia_smi_mib": peak_nvidia_mib}


def _base_train_command(
    args: argparse.Namespace, group: str, seed: int, output_root: Path
) -> list[str]:
    spec = GROUPS[group]
    command = [
        sys.executable,
        "-m",
        "main.train",
        "--data-root",
        str(args.data_root.resolve()),
        "--config",
        str(args.config.resolve()),
        "--split-version",
        args.split_version,
        "--output-root",
        str(output_root.resolve()),
        "--batch-size",
        str(spec.batch_size),
        "--learning-rate",
        str(spec.learning_rate),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--ir-mode",
        "gray",
    ]
    for name, value in W2_LOSS_WEIGHTS.items():
        command.extend((f"--lambda-{name}", str(value)))
    if args.device:
        command.extend(("--device", args.device))
    return command


def _latest_smoke_result(output_root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(
        output_root.glob("smoke-*/smoke_result.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        raise DataContractError(f"未找到 smoke_result.json: {output_root}")
    return paths[-1].parent, read_json(paths[-1])


def _preflight_group(
    args: argparse.Namespace,
    group: str,
    init_sha256: str,
    total_gpu_bytes: int,
    log: Path,
) -> dict[str, Any]:
    status_path = args.experiment_root.resolve() / "preflight" / group / "status.json"
    if status_path.is_file():
        existing = read_json(status_path)
        if (
            existing.get("initialization_sha256") == init_sha256
            and existing.get("spec") == candidate_matrix()[group]
        ):
            _append_log(log, f"SKIP preflight group={group}: {status_path}")
            return existing
    output_root = status_path.parent / "runs"
    command = _base_train_command(args, group, 42, output_root)
    command.extend(("--epochs", "1", "--smoke-only", "--max-steps", "1"))
    command.extend(("--init-checkpoint", str(args.init_checkpoint.resolve())))
    execution = _run(command, log, tolerate_oom=True)
    result: dict[str, Any] = {
        "group": group,
        "spec": candidate_matrix()[group],
        "initialization_sha256": init_sha256,
        "gpu_total_memory_bytes": total_gpu_bytes,
        "peak_nvidia_smi_mib": execution["peak_nvidia_smi_mib"],
    }
    if execution["oom"]:
        result.update(
            {
                "status": "unavailable",
                "reason": "CUDA out of memory during one-step real-data preflight",
                "peak_gpu_memory_bytes": None,
                "peak_gpu_memory_reserved_bytes": None,
            }
        )
    else:
        run_dir, smoke = _latest_smoke_result(output_root)
        allocated = int(smoke.get("peak_gpu_memory_bytes", 0))
        reserved = int(smoke.get("peak_gpu_memory_reserved_bytes", 0))
        status = memory_preflight_status(
            peak_allocated_bytes=allocated, total_bytes=total_gpu_bytes
        )
        result.update(
            {
                "status": status,
                "reason": (
                    "peak allocated memory <= 95% of total GPU memory"
                    if status == "safe"
                    else "peak allocated memory exceeded 95% of total GPU memory"
                ),
                "run_dir": str(run_dir.resolve()),
                "peak_gpu_memory_bytes": allocated,
                "peak_gpu_memory_reserved_bytes": reserved,
                "allocated_fraction": allocated / total_gpu_bytes,
                "reserved_fraction": reserved / total_gpu_bytes,
                "train_loss": smoke.get("train_loss"),
                "train_grad_norm_max": smoke.get("train_grad_norm_max"),
            }
        )
    write_json(status_path, result)
    return result


def _run_preflight(args: argparse.Namespace, log: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise DataContractError("正式 batch/LR 实验需要 CUDA GPU")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    init_sha256 = sha256_file(args.init_checkpoint.resolve())
    groups = {
        group: _preflight_group(
            args, group, init_sha256, int(properties.total_memory), log
        )
        for group in GROUPS
    }
    payload = {
        "gpu": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "unsafe_threshold_fraction": 0.95,
        "initialization_sha256": init_sha256,
        "groups": groups,
    }
    write_json(args.experiment_root.resolve() / "preflight.json", payload)
    return payload


def _train_group(
    args: argparse.Namespace,
    group: str,
    seed: int,
    epochs: int,
    init_sha256: str,
    log: Path,
) -> Path:
    output_root = args.experiment_root.resolve() / "training" / group / f"seed{seed}"
    while True:
        runs = matching_runs(
            output_root,
            group,
            seed,
            initialization_sha256=init_sha256,
        )
        complete = [
            run
            for run in runs
            if metric_count(run) >= epochs and (run / "last.pt").is_file()
        ]
        if complete:
            _append_log(
                log,
                f"SKIP complete group={group} seed={seed} epochs={epochs}: {complete[-1]}",
            )
            return complete[-1]
        command = _base_train_command(args, group, seed, output_root)
        command.extend(("--epochs", str(epochs)))
        resumable = [
            run
            for run in runs
            if 0 < metric_count(run) < epochs
            and (run / "last.pt").is_file()
            and (run / "best.pt").is_file()
        ]
        if resumable:
            previous = resumable[-1]
            command.extend(
                (
                    "--resume",
                    str(previous / "last.pt"),
                    "--best-checkpoint",
                    str(previous / "best.pt"),
                )
            )
        else:
            command.extend(("--init-checkpoint", str(args.init_checkpoint.resolve())))
        _run(command, log)


def _summaries(
    root: Path, groups: list[str], seed: int, epochs: int, init_sha256: str
) -> list[dict[str, Any]]:
    return [
        summarize_curve(
            group,
            seed,
            latest_completed_run(
                root,
                group,
                seed,
                epochs,
                initialization_sha256=init_sha256,
            ),
        )
        for group in groups
    ]


def _run_screening(
    args: argparse.Namespace,
    safe_groups: list[str],
    init_sha256: str,
    log: Path,
) -> dict[str, Any]:
    for group in safe_groups:
        _train_group(args, group, 42, 5, init_sha256, log)
    decision = rank_curves(
        _summaries(args.experiment_root.resolve(), safe_groups, 42, 5, init_sha256)
    )
    if len(decision["shortlist"]) != 2:
        raise DataContractError(f"初筛未得到两个候选: {decision['shortlist']}")
    decision.update(
        {
            "phase": "screening",
            "epochs": 5,
            "seed": 42,
            "test_images_or_metrics_accessed": False,
        }
    )
    write_json(args.experiment_root.resolve() / "screening_decision.json", decision)
    return decision


def _run_refinement(
    args: argparse.Namespace,
    shortlist: list[str],
    init_sha256: str,
    log: Path,
) -> dict[str, Any]:
    groups = [BASELINE_GROUP, *shortlist]
    for group in groups:
        _train_group(args, group, 42, 15, init_sha256, log)
    decision = rank_curves(
        _summaries(args.experiment_root.resolve(), groups, 42, 15, init_sha256)
    )
    candidates = [
        row
        for row in decision["ranking"]
        if row["group"] != BASELINE_GROUP and row["eligible_curve"]
    ]
    provisional = str(candidates[0]["group"]) if candidates else BASELINE_GROUP
    decision.update(
        {
            "phase": "refinement",
            "epochs": 15,
            "seed": 42,
            "provisional_winner": provisional,
            "test_images_or_metrics_accessed": False,
        }
    )
    write_json(args.experiment_root.resolve() / "refinement_decision.json", decision)
    return decision


def _run_confirmation(
    args: argparse.Namespace,
    group: str,
    init_sha256: str,
    log: Path,
) -> None:
    if group == BASELINE_GROUP:
        return
    for seed in (43, 44):
        _train_group(args, group, seed, 15, init_sha256, log)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_ROOT / "dataset.yaml")
    parser.add_argument("--split-version", default="calibrated_v1")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phase",
        choices=("preflight", "screen", "refine", "confirm", "all"),
        default="all",
    )
    parser.add_argument("--group", choices=tuple(GROUPS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.experiment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.init_checkpoint.resolve().is_file():
        raise FileNotFoundError(args.init_checkpoint)
    init_sha256 = sha256_file(args.init_checkpoint.resolve())
    write_json(root / "candidate_matrix.json", candidate_matrix())
    log = root / "runner.log"

    preflight_path = root / "preflight.json"
    preflight = (
        _run_preflight(args, log)
        if args.phase in ("preflight", "all") or not preflight_path.is_file()
        else read_json(preflight_path)
    )
    if args.phase == "preflight":
        return 0
    safe_groups = [
        group
        for group in GROUPS
        if preflight["groups"][group]["status"] == "safe"
    ]
    if BASELINE_GROUP not in safe_groups:
        raise DataContractError("G0 未通过显存预检，无法执行实验")
    if args.group:
        if args.group not in safe_groups:
            raise DataContractError(f"指定组未通过显存预检: {args.group}")
        if args.phase == "screen":
            _train_group(args, args.group, 42, 5, init_sha256, log)
            return 0

    screening_path = root / "screening_decision.json"
    screening = (
        _run_screening(args, safe_groups, init_sha256, log)
        if args.phase in ("screen", "all") or not screening_path.is_file()
        else read_json(screening_path)
    )
    if args.phase == "screen":
        return 0
    shortlist = [str(group) for group in screening["shortlist"]]

    refinement_path = root / "refinement_decision.json"
    refinement = (
        _run_refinement(args, shortlist, init_sha256, log)
        if args.phase in ("refine", "all") or not refinement_path.is_file()
        else read_json(refinement_path)
    )
    if args.phase == "refine":
        return 0
    confirmation_group = str(args.group or refinement["provisional_winner"])
    _run_confirmation(args, confirmation_group, init_sha256, log)
    write_json(
        root / "training_complete.json",
        {
            "status": "training_complete_pending_quality_evaluation",
            "initialization_sha256": init_sha256,
            "safe_groups": safe_groups,
            "unsafe_groups": [group for group in GROUPS if group not in safe_groups],
            "shortlist": shortlist,
            "confirmed_group": confirmation_group,
            "confirmed_seeds": list((42, 43, 44)),
            "test_images_or_metrics_accessed": False,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
