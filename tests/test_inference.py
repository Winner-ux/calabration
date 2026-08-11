from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from data_pipeline.manifest import write_csv_atomic
from data_pipeline.schema import (
    SAMPLE_FIELDS,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    canonical_rows_hash,
    load_dataset_config,
)
from inference import build_parser, collect_split_input_pairs, run_inference
from inference_utils import (
    infer_pair_tensor,
    load_inference_pair,
    load_model_checkpoint,
    sliding_window_inference,
    window_positions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IdentityFusion(torch.nn.Module):
    def forward(self, vis: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        del ir
        return vis


def _make_formal_split_fixture(root: Path) -> tuple[Path, dict, dict[str, str]]:
    data_root = root / "data"
    data_root.mkdir()
    (data_root / "dataset.yaml").write_text(
        (PROJECT_ROOT / "data" / "dataset.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_dataset_config(data_root / "dataset.yaml")
    split_to_id: dict[str, str] = {}
    samples: list[dict[str, str]] = []
    for index, split_name in enumerate(("train", "val", "test"), start=1):
        sample_id = f"day003__experiment{index:03d}__health__clip001__000001"
        split_to_id[split_name] = sample_id
        relative_root = Path(
            "paired", "day003", f"experiment{index:03d}", "health", "clip001"
        )
        rgb_relative = relative_root / "rgb" / "000001.png"
        ir_relative = relative_root / "ir" / "000001.png"
        array = np.full((64, 64, 3), index * 40, dtype=np.uint8)
        for relative in (rgb_relative, ir_relative):
            target = data_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, mode="RGB").save(target)
        row = {field: "" for field in SAMPLE_FIELDS}
        row.update(
            {
                "sample_id": sample_id,
                "ir_path": ir_relative.as_posix(),
                "rgb_path": rgb_relative.as_posix(),
                "category": "health",
                "category_label": "health",
                "study_day": "day003",
                "experiment_id": f"experiment{index:03d}",
                "clip_id": "clip001",
                "frame_sequence": "1",
                "leakage_group_id": f"group-{index}",
                "width": "64",
                "height": "64",
                "ir_mode": "RGB",
                "rgb_mode": "RGB",
                "usable": "true",
            }
        )
        samples.append(row)
    manifest_dir = data_root / "manifests"
    write_csv_atomic(manifest_dir / "samples.csv", samples, SAMPLE_FIELDS)
    split_dir = data_root / "splits" / "v1"
    for split_name, sample_id in split_to_id.items():
        write_csv_atomic(
            split_dir / f"{split_name}.csv",
            [{"sample_id": sample_id}],
            ["sample_id"],
        )
    manifest_hash = canonical_rows_hash(samples, SAMPLE_FIELDS)
    (split_dir / "split_config.json").write_text(
        json.dumps(
            {
                "split_version": "v1",
                "split_schema_version": SPLIT_SCHEMA_VERSION,
                "formal": True,
                "algorithm": config["splitting"]["algorithm"],
                "dataset_manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "audit.json").write_text(
        json.dumps(
            {
                "audit_level": "full",
                "fatal_count": 0,
                "formal_ready": True,
                "dataset_manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    return data_root, config, split_to_id


class InferenceTests(unittest.TestCase):
    def test_formal_test_split_collection_and_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, config, split_to_id = _make_formal_split_fixture(root)
            pairs, source = collect_split_input_pairs(data_root, config, "v1", "test")
            self.assertEqual([pair.sample_id for pair in pairs], [split_to_id["test"]])
            self.assertEqual(pairs[0].category, "health")
            self.assertEqual(pairs[0].leakage_group_id, "group-3")
            self.assertEqual(
                pairs[0].relative_path.as_posix(),
                "day003/experiment003/health/clip001/000001.png",
            )
            self.assertEqual(source["mode"], "formal_split")
            checkpoint = root / "checkpoint.pt"
            torch.save({"model": IdentityFusion().state_dict()}, checkpoint)
            args = build_parser().parse_args(
                [
                    "--split-version",
                    "v1",
                    "--split",
                    "test",
                    "--data-root",
                    str(data_root),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--mode",
                    "full",
                    "--output-root",
                    str(root / "runs"),
                ]
            )
            with patch("inference.ResNetFusion", return_value=IdentityFusion()):
                run_dir = run_inference(args)
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["result_status"], "heldout_split_inference")
            self.assertEqual(metadata["pair_count"], 1)
            self.assertEqual(metadata["pairs"][0]["sample_id"], split_to_id["test"])
            output = (
                run_dir
                / "images"
                / "day003"
                / "experiment003"
                / "health"
                / "clip001"
                / "000001.png"
            )
            self.assertTrue(output.is_file())

    def test_split_collection_rejects_unknown_duplicate_and_cross_split_ids(self) -> None:
        cases = (
            (["missing"], "不存在的 ID"),
            (None, "重复 sample_id"),
            ("train_overlap", "跨 split 重复"),
        )
        for replacement, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root, config, split_to_id = _make_formal_split_fixture(root)
                if replacement is None:
                    values = [split_to_id["test"], split_to_id["test"]]
                elif replacement == "train_overlap":
                    values = [split_to_id["train"]]
                else:
                    values = replacement
                write_csv_atomic(
                    data_root / "splits" / "v1" / "test.csv",
                    ({"sample_id": value} for value in values),
                    ["sample_id"],
                    overwrite=True,
                )
                with self.assertRaisesRegex(DataContractError, expected):
                    collect_split_input_pairs(data_root, config, "v1", "test")

    def test_formal_test_split_rejects_random_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root, _, _ = _make_formal_split_fixture(root)
            args = build_parser().parse_args(
                [
                    "--split-version",
                    "v1",
                    "--data-root",
                    str(data_root),
                    "--allow-random-weights",
                    "--device",
                    "cpu",
                ]
            )
            with self.assertRaisesRegex(DataContractError, "禁止使用随机"):
                run_inference(args)

    def test_window_positions_cover_right_edge(self) -> None:
        positions = window_positions(length=91, tile_size=32, stride=24)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 59)
        covered = [False] * 91
        for start in positions:
            for index in range(start, start + 32):
                covered[index] = True
        self.assertTrue(all(covered))

    def test_sliding_window_identity_has_no_seam_and_restores_size(self) -> None:
        generator = torch.Generator().manual_seed(7)
        vis = torch.rand((1, 3, 75, 91), generator=generator)
        ir = torch.rand((1, 1, 75, 91), generator=generator)
        fused = sliding_window_inference(
            IdentityFusion(),
            vis,
            ir,
            device=torch.device("cpu"),
            tile_size=32,
            overlap=8,
            factor=32,
        )
        self.assertEqual(tuple(fused.shape), (1, 3, 75, 91))
        self.assertTrue(torch.allclose(fused, vis, atol=1e-6))

    def test_auto_mode_falls_back_and_full_guard_rejects(self) -> None:
        vis = torch.rand(1, 3, 64, 96)
        ir = torch.rand(1, 1, 64, 96)
        fused, details = infer_pair_tensor(
            IdentityFusion(),
            vis,
            ir,
            device=torch.device("cpu"),
            requested_mode="auto",
            tile_size=32,
            overlap=8,
            factor=32,
            max_full_tokens=100,
        )
        self.assertEqual(details["actual_mode"], "sliding")
        self.assertIsNotNone(details["fallback_reason"])
        self.assertEqual(tuple(fused.shape), tuple(vis.shape))
        with self.assertRaisesRegex(DataContractError, "安全阈值"):
            infer_pair_tensor(
                IdentityFusion(),
                vis,
                ir,
                device=torch.device("cpu"),
                requested_mode="full",
                factor=32,
                max_full_tokens=100,
            )

    def test_png_pair_contract_and_bt601(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vis_path = root / "000001.png"
            ir_dir = root / "ir"
            ir_dir.mkdir()
            ir_path = ir_dir / "000001.png"
            vis_array = np.zeros((17, 19, 3), dtype=np.uint8)
            vis_array[..., 0] = 255
            ir_array = np.zeros((17, 19, 3), dtype=np.uint8)
            ir_array[..., 1] = 255
            Image.fromarray(vis_array, mode="RGB").save(vis_path)
            Image.fromarray(ir_array, mode="RGB").save(ir_path)
            config = load_dataset_config(PROJECT_ROOT / "data" / "dataset.yaml")
            vis, ir = load_inference_pair(vis_path, ir_path, config)
            self.assertEqual(tuple(vis.shape), (1, 3, 17, 19))
            self.assertEqual(tuple(ir.shape), (1, 1, 17, 19))
            self.assertTrue(torch.allclose(ir, torch.full_like(ir, 0.587), atol=1e-6))

    def test_actual_model_untrained_smoke_is_explicitly_marked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vis_dir = root / "vis"
            ir_dir = root / "ir"
            vis_dir.mkdir()
            ir_dir.mkdir()
            generator = np.random.default_rng(42)
            array = generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(array, mode="RGB").save(vis_dir / "000001.png")
            Image.fromarray(array, mode="RGB").save(ir_dir / "000001.png")
            args = build_parser().parse_args(
                [
                    "--vis",
                    str(vis_dir / "000001.png"),
                    "--ir",
                    str(ir_dir / "000001.png"),
                    "--allow-random-weights",
                    "--device",
                    "cpu",
                    "--mode",
                    "full",
                    "--output-root",
                    str(root / "runs"),
                ]
            )
            run_dir = run_inference(args)
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["result_status"], "smoke_only_untrained")
            self.assertEqual(metadata["pair_count"], 1)
            output = run_dir / "images" / "SMOKE_UNTRAINED__000001.png"
            self.assertTrue(output.is_file())
            with Image.open(output) as fused:
                self.assertEqual(fused.size, (64, 64))

    def test_training_checkpoint_layout_loads_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            source = torch.nn.Linear(3, 2)
            torch.save({"model": source.state_dict(), "epoch": 4}, checkpoint)
            target = torch.nn.Linear(3, 2)
            metadata = load_model_checkpoint(target, checkpoint)
            self.assertEqual(metadata["layout"], "training_checkpoint")
            self.assertEqual(metadata["epoch"], 4)
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()
