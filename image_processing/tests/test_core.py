from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing_core.discovery import discover_numeric_pairs, validate_experiment_pair
from processing_core.importer import ImportConflict, import_confirmed_pairs
from processing_core.locking import GlobalProcessingLock, ProcessingLocked
from processing_core.manifest import read_task_pairs
from processing_core.models import PairSpec
from processing_core.paths import ProjectPaths
from processing_core.processors import run_calibration_batch, run_experiment_batch


def make_video(
    path: Path, size: tuple[int, int], fps: float, frames: int, seed: int = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("测试环境无法创建 MP4")
    rng = np.random.default_rng(seed)
    for index in range(frames):
        image = rng.integers(0, 30, (size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(
            image,
            (index % max(1, size[0] - 20), 5),
            (min(size[0] - 1, index % max(1, size[0] - 20) + 18), 25),
            (220, 220, 220),
            -1,
        )
        writer.write(image)
    writer.release()


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = ProjectPaths(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_discovery_only_auto_pairs_numeric_stems_and_preserves_category(
        self,
    ) -> None:
        base = self.root / "vedio" / "day7" / "1" / "health"
        make_video(base / "ir" / "01.mp4", (160, 90), 10, 8)
        make_video(base / "rgb" / "01.mp4", (160, 90), 10, 8, seed=2)
        make_video(base / "ir" / "random.mp4", (160, 90), 10, 8, seed=3)
        pairs, issues = discover_numeric_pairs(self.root / "vedio", "experiment")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].day, "day7")
        self.assertEqual(pairs[0].experiment, "experiment1")
        self.assertEqual(pairs[0].category, "health")
        self.assertTrue(any("不能自动配对" in item for item in issues))

    def test_calibration_discovery_reports_cross_task_duplicate_hashes(self) -> None:
        first = self.root / "cabration_video" / "day3" / "1"
        second = self.root / "cabration_video" / "day4" / "1"
        make_video(first / "ir" / "01.mp4", (160, 90), 10, 8)
        make_video(first / "rgb" / "01.mp4", (90, 160), 15, 12, seed=2)
        (second / "ir").mkdir(parents=True)
        (second / "rgb").mkdir(parents=True)
        (second / "ir" / "01.mp4").write_bytes((first / "ir" / "01.mp4").read_bytes())
        (second / "rgb" / "01.mp4").write_bytes((first / "rgb" / "01.mp4").read_bytes())
        pairs, issues = discover_numeric_pairs(
            self.root / "cabration_video", "calibration"
        )
        self.assertEqual(len(pairs), 2)
        self.assertTrue(any("跨任务重复视频" in item for item in issues))
        self.assertTrue(all(pair.warnings for pair in pairs))

    def test_duplicate_numeric_identity_is_never_auto_paired(self) -> None:
        base = self.root / "cabration_video" / "day6" / "1"
        make_video(base / "ir" / "01.mp4", (160, 90), 10, 8)
        make_video(base / "ir" / "1.mov", (160, 90), 10, 8, seed=2)
        make_video(base / "rgb" / "01.mp4", (90, 160), 15, 12, seed=3)
        pairs, issues = discover_numeric_pairs(
            self.root / "cabration_video", "calibration"
        )
        self.assertEqual(pairs, [])
        self.assertTrue(any("多个候选" in item for item in issues))

    def test_experiment_validation_rejects_wrong_size_or_fps(self) -> None:
        ir, rgb = self.root / "ir.mp4", self.root / "rgb.mp4"
        make_video(ir, (160, 90), 10, 8)
        make_video(rgb, (90, 160), 12, 8)
        pair = PairSpec(
            "experiment", "day1", "experiment1", "01", ir, rgb, "health", True
        )
        errors = validate_experiment_pair(pair, target_size=(160, 90))
        self.assertTrue(any("RGB 尺寸" in item for item in errors))
        self.assertTrue(any("FPS 不一致" in item for item in errors))

    def test_import_copies_and_writes_schema_v2_without_touching_source(self) -> None:
        ir, rgb = self.root / "来源" / "红外.mp4", self.root / "来源" / "彩色.mp4"
        make_video(ir, (160, 90), 10, 8)
        make_video(rgb, (90, 160), 15, 12, seed=2)
        pair = PairSpec(
            "calibration", "day8", "experiment2", "1", ir, rgb, confirmed=True
        )
        messages = import_confirmed_pairs(self.paths, [pair])
        task = self.paths.input_root / "calibration" / "day8" / "experiment2"
        self.assertTrue(ir.exists() and rgb.exists())
        self.assertTrue((task / "ir" / "01.mp4").exists())
        self.assertTrue((task / "rgb" / "01.mp4").exists())
        self.assertTrue((task / "READY").exists())
        data = yaml.safe_load((task / "job.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 2)
        self.assertTrue(data["pairs"][0]["user_confirmed"])
        self.assertTrue(messages)

    def test_import_refuses_different_existing_target(self) -> None:
        source_ir, source_rgb = (
            self.root / "source_ir.mp4",
            self.root / "source_rgb.mp4",
        )
        make_video(source_ir, (160, 90), 10, 8)
        make_video(source_rgb, (160, 90), 10, 8, seed=2)
        target = (
            self.paths.input_root
            / "calibration"
            / "day1"
            / "experiment1"
            / "ir"
            / "01.mp4"
        )
        make_video(target, (160, 90), 10, 8, seed=9)
        pair = PairSpec(
            "calibration",
            "day1",
            "experiment1",
            "01",
            source_ir,
            source_rgb,
            confirmed=True,
        )
        with self.assertRaises(ImportConflict):
            import_confirmed_pairs(self.paths, [pair])

    def test_global_lock_prevents_both_processors(self) -> None:
        with GlobalProcessingLock(self.paths, "calibration"):
            with self.assertRaises(ProcessingLocked):
                with GlobalProcessingLock(self.paths, "experiment"):
                    pass

    def test_v1_manifest_is_still_readable(self) -> None:
        task = self.paths.input_root / "experiment" / "day3" / "experiment1"
        make_video(task / "health" / "ir" / "01.mp4", (160, 90), 10, 8)
        make_video(task / "health" / "rgb" / "01.mp4", (160, 90), 10, 8, seed=2)
        (task / "job.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "job_type": "experiment",
                    "day": "day3",
                    "experiment": "experiment1",
                    "categories": ["health"],
                }
            ),
            encoding="utf-8",
        )
        pairs = read_task_pairs(task, "experiment")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].category, "health")

    def test_partial_import_upgrades_v1_without_dropping_other_clips(self) -> None:
        task = self.paths.input_root / "calibration" / "day3" / "experiment1"
        for number in (1, 2):
            make_video(task / "ir" / f"{number:02d}.mp4", (160, 90), 10, 8, seed=number)
            make_video(
                task / "rgb" / f"{number:02d}.mp4", (90, 160), 15, 12, seed=number + 10
            )
        (task / "job.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "job_type": "calibration",
                    "day": "day3",
                    "experiment": "experiment1",
                }
            ),
            encoding="utf-8",
        )
        pair = PairSpec(
            "calibration",
            "day3",
            "experiment1",
            "01",
            task / "ir" / "01.mp4",
            task / "rgb" / "01.mp4",
            confirmed=True,
        )
        import_confirmed_pairs(self.paths, [pair])
        data = yaml.safe_load((task / "job.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual({item["clip_id"] for item in data["pairs"]}, {"01", "02"})


class ProcessorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = ProjectPaths(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_experiment_atomic_output_and_existing_result_skip(self) -> None:
        task = self.paths.input_root / "experiment" / "day9" / "experiment1"
        make_video(task / "health" / "ir" / "01.mp4", (160, 90), 10, 12)
        make_video(task / "health" / "rgb" / "01.mp4", (160, 90), 10, 12, seed=2)
        (task / "job.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "job_type": "experiment",
                    "day": "day9",
                    "experiment": "experiment1",
                    "categories": ["health"],
                }
            ),
            encoding="utf-8",
        )
        (task / "READY").write_text("test\n", encoding="utf-8")
        results, _ = run_experiment_batch(self.paths, count=4, target_size=(160, 90))
        self.assertEqual(results[0].status, "success")
        output = (
            self.paths.output_root
            / "experiment"
            / "day9"
            / "experiment1"
            / "health"
            / "clip01"
        )
        self.assertEqual(len(list((output / "ir").glob("*.png"))), 4)
        self.assertEqual(
            {p.name for p in (output / "ir").glob("*.png")},
            {p.name for p in (output / "rgb").glob("*.png")},
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        skipped, _ = run_experiment_batch(self.paths, count=5, target_size=(160, 90))
        self.assertEqual(skipped[0].status, "skipped")
        self.assertEqual(len(list((output / "ir").glob("*.png"))), 4)
        replaced, _ = run_experiment_batch(
            self.paths, count=5, force=True, target_size=(160, 90)
        )
        self.assertEqual(replaced[0].status, "success")
        self.assertEqual(len(list((output / "ir").glob("*.png"))), 5)
        backups = list((self.paths.system_root / "backups").rglob("clip01"))
        self.assertTrue(
            any(len(list((backup / "ir").glob("*.png"))) == 4 for backup in backups)
        )

    def test_calibration_different_fps_and_portrait_rgb(self) -> None:
        task = self.paths.input_root / "calibration" / "day9" / "experiment2"
        make_video(task / "ir" / "01.mp4", (160, 90), 10, 14)
        make_video(task / "rgb" / "01.mp4", (90, 160), 15, 21, seed=2)
        (task / "job.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "job_type": "calibration",
                    "day": "day9",
                    "experiment": "experiment2",
                }
            ),
            encoding="utf-8",
        )
        (task / "READY").write_text("test\n", encoding="utf-8")
        results, _ = run_calibration_batch(self.paths, count=2, target_size=(160, 90))
        self.assertEqual(results[0].status, "success", results[0].message)
        output = self.paths.output_root / "calibration" / "day9" / "experiment2"
        self.assertEqual(len(list((output / "images" / "ir").glob("*.png"))), 2)
        self.assertTrue((output / "videos" / "rgb" / "rgb_01.mp4").exists())


if __name__ == "__main__":
    unittest.main()
