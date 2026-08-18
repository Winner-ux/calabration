"""Finalize an accepted batch/LR experiment and update the dataset defaults."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from data_pipeline.schema import DataContractError, load_dataset_config
from scripts.batch_lr_experiment import BASELINE_GROUP, GROUPS, read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "batch_lr_optimization"
DEFAULT_CONFIG = PROJECT_ROOT / "datasets" / "calibrated_v1" / "dataset.yaml"


def _replace_training_scalar(text: str, key: str, value: str) -> str:
    lines = text.splitlines(keepends=True)
    in_training = False
    replaced = 0
    pattern = re.compile(rf"^(\s{{2}}{re.escape(key)}:\s*).*$")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "training:":
            in_training = True
            continue
        if in_training and stripped and not line.startswith((" ", "\t")):
            in_training = False
        if in_training:
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            body = line[: -len(newline)] if newline else line
            match = pattern.match(body)
            if match:
                lines[index] = f"{match.group(1)}{value}{newline}"
                replaced += 1
    if replaced != 1:
        raise DataContractError(
            f"dataset config training.{key} expected once, found {replaced}"
        )
    return "".join(lines)


def _require_visual_pass(root: Path, winner: str) -> dict[str, Any]:
    review_path = root / "visual_review.json"
    review = read_json(review_path)
    if review.get("status") != "pass":
        raise DataContractError(f"blind visual review has not passed: {review_path}")
    if review.get("selected_group") != winner:
        raise DataContractError(
            f"visual review group {review.get('selected_group')} != winner {winner}"
        )
    required_checks = {
        "checkerboard",
        "halo",
        "noise",
        "brightness_drift",
        "color_shift",
    }
    checks = review.get("artifact_checks")
    if not isinstance(checks, dict) or not required_checks.issubset(checks):
        raise DataContractError("visual_review.json is missing required artifact checks")
    if any(str(checks[name]).lower() != "pass" for name in required_checks):
        raise DataContractError("one or more blind-review artifact checks failed")
    return review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    config_path = args.config.resolve()
    decision_path = root / "quality_decision.json"
    decision = read_json(decision_path)
    winner = decision.get("automatic_winner")
    if not winner:
        completion = {
            "status": "complete_retain_G0",
            "selected_group": BASELINE_GROUP,
            "config_updated": False,
            "reason": "no non-baseline candidate passed all automatic gates",
        }
        write_json(root / "EXPERIMENT_COMPLETE.json", completion)
        print(json.dumps(completion, ensure_ascii=False, indent=2))
        return 0
    winner = str(winner)
    if winner == BASELINE_GROUP or winner not in GROUPS:
        raise DataContractError(f"unexpected automatic winner: {winner}")
    candidate = decision["candidates"][winner]
    if not candidate.get("automatic_checks_pass"):
        raise DataContractError(f"automatic checks failed for {winner}")
    review = _require_visual_pass(root, winner)

    spec = GROUPS[winner]
    before = load_dataset_config(config_path)
    original_text = config_path.read_text(encoding="utf-8")
    updated_text = _replace_training_scalar(
        original_text, "batch_size", str(spec.batch_size)
    )
    updated_text = _replace_training_scalar(
        updated_text, "learning_rate", f"{spec.learning_rate:.10g}"
    )
    if updated_text != original_text:
        config_path.write_text(updated_text, encoding="utf-8")
    after = load_dataset_config(config_path)
    if int(after["training"]["batch_size"]) != spec.batch_size or not abs(
        float(after["training"]["learning_rate"]) - spec.learning_rate
    ) < 1e-15:
        raise DataContractError("updated dataset training defaults failed verification")

    decision["visual_review"] = review
    decision["final_acceptance"] = {
        "status": "accepted",
        "selected_group": winner,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
    }
    write_json(decision_path, decision)
    completion = {
        "status": "complete",
        "selected_group": winner,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "config_updated": updated_text != original_text,
        "config_path": str(config_path),
        "previous_training_defaults": {
            "batch_size": int(before["training"]["batch_size"]),
            "learning_rate": float(before["training"]["learning_rate"]),
        },
        "test_images_or_metrics_accessed": False,
    }
    write_json(root / "EXPERIMENT_COMPLETE.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
