"""可恢复地执行损失权重筛选、三种子确认与统一评估。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from data_pipeline.schema import DataContractError
from main.inference_utils import sha256_file
from main.train import DEFAULT_LOSS_WEIGHTS
from scripts.loss_weight_experiment import (
    BASELINE_RUNS,
    SEEDS,
    WEIGHT_GROUPS,
    matching_runs,
    metric_count,
    read_json,
    weights_dict,
)


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _run(command: list[str], log_path: Path) -> None:
    _append_log(log_path, "COMMAND: " + subprocess.list2cmdline(command))
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"命令失败 returncode={completed.returncode}: {command}")


def _verify_baselines(project_root: Path, init_checkpoint: Path) -> dict[str, object]:
    expected_init_sha = sha256_file(init_checkpoint)
    audits: list[dict[str, object]] = []
    expected_split_hash: str | None = None
    for seed, relative in BASELINE_RUNS.items():
        run_dir = project_root / relative
        config = read_json(run_dir / "run_config.json")
        init = config.get("initialization") or {}
        raw_weights = config.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
        actual_weights = {name: float(raw_weights[name]) for name in DEFAULT_LOSS_WEIGHTS}
        checks = {
            "seed": int(config.get("seed", -1)) == seed,
            "ir_mode": config.get("ir_mode") == "gray",
            "learning_rate": float(config.get("learning_rate", -1)) == 1e-5,
            "batch_size": int(config.get("batch_size", -1)) == 2,
            "epochs": int(config.get("epochs", -1)) == 10,
            "weights": actual_weights == DEFAULT_LOSS_WEIGHTS,
            "initialization_sha256": init.get("sha256") == expected_init_sha,
            "last_checkpoint": (run_dir / "last.pt").is_file(),
            "metrics": metric_count(run_dir) == 10,
        }
        split_hash = str(config.get("split_manifest_sha256", ""))
        if expected_split_hash is None:
            expected_split_hash = split_hash
        checks["split_hash"] = bool(split_hash) and split_hash == expected_split_hash
        audits.append({"seed": seed, "run_dir": str(run_dir), "checks": checks})
        if not all(checks.values()):
            raise DataContractError(f"W0 基线审计失败 seed={seed}: {checks}")
    return {
        "status": "passed",
        "initialization_sha256": expected_init_sha,
        "split_manifest_sha256": expected_split_hash,
        "current_loss_re_evaluation_required": True,
        "runs": audits,
    }


def _train_group(args: argparse.Namespace, group: str, seed: int, epochs: int, log: Path) -> None:
    output_root = args.ablation_root.resolve() / group / f"seed{seed}"
    while True:
        runs = matching_runs(output_root, group, seed)
        complete = [run for run in runs if metric_count(run) >= epochs and (run / "last.pt").is_file()]
        if complete:
            _append_log(log, f"SKIP complete group={group} seed={seed} epochs={epochs}: {complete[-1]}")
            return
        command = [
            sys.executable,
            "-m", "main.train",
            "--data-root", str(args.data_root.resolve()),
            "--config", str(args.config.resolve()),
            "--split-version", args.split_version,
            "--output-root", str(output_root),
            "--epochs", str(epochs),
            "--batch-size", "2",
            "--learning-rate", "1e-5",
            "--num-workers", str(args.num_workers),
            "--seed", str(seed),
            "--ir-mode", "gray",
        ]
        for name, value in weights_dict(group).items():
            command.extend((f"--lambda-{name.replace('_', '-')}", str(value)))
        if args.device:
            command.extend(("--device", args.device))
        resumable = [
            run for run in runs
            if 0 < metric_count(run) < epochs
            and (run / "last.pt").is_file()
            and (run / "best.pt").is_file()
        ]
        if resumable:
            previous = resumable[-1]
            command.extend((
                "--resume", str(previous / "last.pt"),
                "--best-checkpoint", str(previous / "best.pt"),
            ))
        else:
            command.extend(("--init-checkpoint", str(args.init_checkpoint.resolve())))
        _run(command, log)


def _evaluate(args: argparse.Namespace, phase: str, log: Path, shortlist: list[str] | None = None) -> None:
    command = [
        sys.executable,
        "-m", "scripts.evaluate_loss_weight_ablation",
        "--data-root", str(args.data_root.resolve()),
        "--config", str(args.config.resolve()),
        "--split-version", args.split_version,
        "--ablation-root", str(args.ablation_root.resolve()),
        "--phase", phase,
    ]
    if shortlist:
        command.extend(("--groups", *shortlist))
    if phase == "final":
        command.append("--full-resolution")
    if args.device:
        command.extend(("--device", args.device))
    _run(command, log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()

    root = args.ablation_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = root / "ablation_runner.log"
    project_root = Path(__file__).resolve().parents[1]
    audit = _verify_baselines(project_root, args.init_checkpoint.resolve())
    (root / "baseline_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    matrix = {
        group: {"label": spec["label"], "weights": weights_dict(group)}
        for group, spec in WEIGHT_GROUPS.items()
    }
    (root / "candidate_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for group in ("W1", "W2", "W3", "W4", "W5"):
        _train_group(args, group, 42, 5, log)
    _evaluate(args, "screening", log)
    screening = read_json(root / "screening_decision.json")
    shortlist = [str(group) for group in screening["shortlist"]]
    if len(shortlist) != 2:
        raise DataContractError(f"筛选必须得到两个候选，实际为 {shortlist}")

    for group in shortlist:
        _train_group(args, group, 42, 10, log)
        for seed in SEEDS[1:]:
            _train_group(args, group, seed, 10, log)
    _evaluate(args, "final", log, shortlist)
    (root / "ABLATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete_pending_blind_review",
                "screening_groups": ["W1", "W2", "W3", "W4", "W5"],
                "shortlist": shortlist,
                "seeds": list(SEEDS),
                "test_images_or_metrics_accessed": False,
                "test_manifest_integrity_read_by_training_entrypoint": True,
                "decision": str(root / "decision.json"),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
