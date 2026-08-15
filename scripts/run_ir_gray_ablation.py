"""顺序执行可恢复的 gray/learned_gray 三种子 train/val 消融。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODES = ("gray", "learned_gray")
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_count(run_dir: Path) -> int:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _matching_runs(output_root: Path, mode: str, seed: int) -> list[Path]:
    runs: list[Path] = []
    if not output_root.is_dir():
        return runs
    for config_path in output_root.glob("train-*/run_config.json"):
        config = _read_json(config_path)
        if config.get("ir_mode") == mode and int(config.get("seed", -1)) == seed:
            runs.append(config_path.parent)
    return sorted(runs, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    root = args.ablation_root.resolve()
    log_path = root / "ablation_runner.log"
    for mode in MODES:
        for seed in SEEDS:
            output_root = root / mode / f"seed{seed}"
            while True:
                runs = _matching_runs(output_root, mode, seed)
                completed_runs = [run for run in runs if _metric_count(run) >= args.epochs]
                if completed_runs:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"SKIP complete mode={mode} seed={seed}: {completed_runs[-1]}\n")
                    break
                command = [
                    sys.executable,
                    "-m",
                    "main.train",
                    "--data-root", str(args.data_root.resolve()),
                    "--config", str(args.config.resolve()),
                    "--split-version", args.split_version,
                    "--output-root", str(output_root),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--learning-rate", str(args.learning_rate),
                    "--num-workers", str(args.num_workers),
                    "--seed", str(seed),
                    "--ir-mode", mode,
                ]
                if args.device:
                    command.extend(("--device", args.device))
                resumable = [
                    run for run in runs
                    if _metric_count(run) > 0
                    and (run / "last.pt").is_file()
                    and (run / "best.pt").is_file()
                ]
                if resumable:
                    previous = resumable[-1]
                    command.extend(
                        (
                            "--resume", str(previous / "last.pt"),
                            "--best-checkpoint", str(previous / "best.pt"),
                        )
                    )
                else:
                    command.extend(("--init-checkpoint", str(args.init_checkpoint.resolve())))
                _run(command, log_path)

    evaluation = [
        sys.executable,
        "-m",
        "scripts.evaluate_ir_gray_ablation",
        "--data-root", str(args.data_root.resolve()),
        "--config", str(args.config.resolve()),
        "--split-version", args.split_version,
        "--ablation-root", str(root),
    ]
    if args.device:
        evaluation.extend(("--device", args.device))
    _run(evaluation, log_path)
    (root / "ABLATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "modes": list(MODES),
                "seeds": list(SEEDS),
                "epochs": args.epochs,
                "test_images_or_metrics_accessed": False,
                "test_manifest_integrity_read_by_training_entrypoint": True,
                "decision": str(root / "decision.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
