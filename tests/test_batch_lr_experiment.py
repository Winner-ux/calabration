from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.batch_lr_experiment import (
    GROUPS,
    W2_LOSS_WEIGHTS,
    candidate_matrix,
    curve_roughness,
    improvement_percent,
    matching_runs,
    memory_preflight_status,
    rank_curves,
    summarize_curve,
)


METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "epoch_seconds",
    "samples_per_second",
    "peak_gpu_memory_bytes",
    "peak_gpu_memory_reserved_bytes",
)


def _write_run(
    root: Path,
    group: str,
    *,
    train: list[float],
    validation: list[float],
    initialization_sha256: str = "abc",
) -> Path:
    spec = GROUPS[group]
    run = root / f"train-{group.lower()}"
    run.mkdir(parents=True)
    config = {
        "smoke_only": False,
        "seed": 42,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "ir_mode": "gray",
        "loss_weights": W2_LOSS_WEIGHTS,
        "initialization": {"sha256": initialization_sha256},
    }
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with (run / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for epoch, (train_loss, val_loss) in enumerate(zip(train, validation, strict=True)):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_seconds": 10 + epoch,
                    "samples_per_second": 100 + epoch,
                    "peak_gpu_memory_bytes": 1000,
                    "peak_gpu_memory_reserved_bytes": 1200,
                }
            )
    (run / "last.pt").touch()
    return run


class BatchLRExperimentTests(unittest.TestCase):
    def test_matrix_contains_requested_physical_batch32(self) -> None:
        matrix = candidate_matrix()
        self.assertEqual(tuple(matrix), tuple(f"G{i}" for i in range(9)))
        self.assertEqual(matrix["G0"]["batch_size"], 2)
        self.assertEqual(matrix["G0"]["learning_rate"], 1e-5)
        self.assertEqual(matrix["G8"]["batch_size"], 32)
        self.assertEqual(matrix["G8"]["learning_rate"], 2e-5)

    def test_curve_roughness_removes_constant_trend(self) -> None:
        self.assertAlmostEqual(curve_roughness([1.0, 0.9, 0.8, 0.7]), 0.0)
        self.assertGreater(curve_roughness([1.0, 0.8, 0.95, 0.7]), 0.05)

    def test_improvement_and_memory_thresholds(self) -> None:
        self.assertAlmostEqual(improvement_percent(0.2, 0.18), 10.0)
        self.assertEqual(
            memory_preflight_status(peak_allocated_bytes=95, total_bytes=100),
            "safe",
        )
        self.assertEqual(
            memory_preflight_status(peak_allocated_bytes=96, total_bytes=100),
            "unsafe",
        )
        self.assertEqual(
            memory_preflight_status(
                peak_allocated_bytes=None, total_bytes=100, failed=True
            ),
            "unavailable",
        )

    def test_matching_run_checks_group_seed_weights_and_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(
                root,
                "G4",
                train=[0.5, 0.4, 0.3],
                validation=[0.3, 0.25, 0.2],
            )
            self.assertEqual(
                matching_runs(root, "G4", 42, initialization_sha256="abc"),
                [run],
            )
            self.assertEqual(
                matching_runs(root, "G4", 42, initialization_sha256="wrong"),
                [],
            )

    def test_curve_summary_shortlists_finite_groups_but_keeps_final_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _write_run(
                root / "g0",
                "G0",
                train=[0.50, 0.45, 0.40, 0.35, 0.30],
                validation=[0.220, 0.215, 0.210, 0.205, 0.200],
            )
            stable = _write_run(
                root / "g4",
                "G4",
                train=[0.50, 0.44, 0.38, 0.32, 0.26],
                validation=[0.215, 0.210, 0.205, 0.200, 0.195],
            )
            flat_bad = _write_run(
                root / "g6",
                "G6",
                train=[0.50, 0.50, 0.50, 0.50, 0.50],
                validation=[0.30, 0.30, 0.30, 0.30, 0.30],
            )
            ranked = rank_curves(
                [
                    summarize_curve("G0", 42, baseline),
                    summarize_curve("G4", 42, stable),
                    summarize_curve("G6", 42, flat_bad),
                ]
            )
            self.assertEqual(ranked["shortlist"], ["G4", "G6"])
            bad = next(row for row in ranked["ranking"] if row["group"] == "G6")
            self.assertFalse(bad["quality_guardrail_pass"])
            self.assertFalse(bad["eligible_curve"])


if __name__ == "__main__":
    unittest.main()
