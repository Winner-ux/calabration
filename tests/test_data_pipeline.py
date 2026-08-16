from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data_pipeline.audit import audit_dataset, write_audit_outputs
from data_pipeline.manifest import (
    build_manifest_rows,
    validate_experiment_contract,
    write_csv_atomic,
    write_manifest_files,
)
from data_pipeline.sampling import EpochShuffleSampler
from data_pipeline.schema import (
    GROUP_ASSIGNMENT_FIELDS,
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    read_csv_rows,
)
from data_pipeline.splitting import create_split_version, generate_nested_group_assignments
from data_pipeline.transforms import PairedGeometryTransform
from main.dataset import PairedFusionDataset
from main.inference_utils import load_initial_model_checkpoint
from main.loss import FusionLoss
from model import GrayEnhancer, CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion
from main.train import (
    DEFAULT_LOSS_WEIGHTS,
    _loss_weights_from_args,
    _validate_checkpoint_loss_weights,
    run_training,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
CATEGORIES = ("health", "health_sick", "sick")


def _write_rgb(path: Path, category_index: int, experiment_index: int) -> None:
    y, x = np.mgrid[:64, :64]
    base = (x * 3 + y * 5 + category_index * 17 + experiment_index * 11) % 256
    array = np.stack((base, base, base), axis=-1).astype(np.uint8)
    array[0, 0] = (
        category_index * 40 + experiment_index,
        experiment_index,
        category_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _make_synthetic_data(root: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    data_root = root / "data"
    data_root.mkdir(parents=True)
    config = yaml.safe_load((DEFAULT_DATA_ROOT / "dataset.yaml").read_text(encoding="utf-8"))
    config["preprocessing"]["crop_size"] = 64
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["amp"] = False
    (data_root / "dataset.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    assignments: list[dict[str, str]] = []
    for experiment_index in range(1, 7):
        experiment = f"experiment{experiment_index:03d}"
        for category_index, category in enumerate(CATEGORIES, start=1):
            clip = "clip001"
            clip_dir = data_root / "paired" / "day003" / experiment / category / clip
            filename = "000001.png"
            _write_rgb(clip_dir / "ir" / filename, category_index, experiment_index)
            _write_rgb(clip_dir / "rgb" / filename, category_index, experiment_index)
            with (clip_dir / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "sequence",
                        "filename",
                        "source_frame_index_0_based",
                        "timestamp_seconds",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sequence": 1,
                        "filename": filename,
                        "source_frame_index_0_based": experiment_index * 100,
                        "timestamp_seconds": f"{experiment_index:.3f}",
                    }
                )
            summary = {
                "schema_version": 1,
                "sources": {
                    "ir": {
                        "fingerprint": {
                            "sample_sha256": f"{experiment}-{category}-{clip}-ir"
                        }
                    },
                    "rgb": {
                        "fingerprint": {
                            "sample_sha256": f"{experiment}-{category}-{clip}-rgb"
                        }
                    },
                },
            }
            (clip_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            _write_rgb(
                clip_dir / "sync_preview.png", category_index, experiment_index
            )
            assignments.append(
                {
                    "study_day": "day003",
                    "experiment_id": experiment,
                    "category": category,
                    "clip_id": clip,
                    "subject_id": f"subject-{experiment_index}",
                    "batch_id": f"batch-{experiment_index}",
                    "scene_id": f"scene-{experiment_index}",
                    "session_id": f"session-{experiment_index}",
                    "lighting_condition": "controlled",
                    "scene_condition": "tank",
                    "leakage_group_id": f"group-{experiment_index}",
                }
            )
    manifest_dir = data_root / "manifests"
    write_csv_atomic(
        manifest_dir / "group_assignments.csv",
        assignments,
        GROUP_ASSIGNMENT_FIELDS,
    )
    loaded_config = load_dataset_config(data_root / "dataset.yaml")
    samples, groups = build_manifest_rows(
        data_root, loaded_config, manifest_dir / "group_assignments.csv"
    )
    write_manifest_files(
        manifest_dir / "samples.csv", manifest_dir / "groups.csv", samples, groups
    )
    return loaded_config, samples, groups


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_context = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_context.name)
        cls.config, cls.samples, cls.groups = _make_synthetic_data(cls.root)
        cls.data_root = cls.root / "data"
        cls.split_dir = create_split_version(
            cls.data_root, "v1", cls.samples, cls.groups, cls.config, seed=42
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_manifest_contract_and_deterministic_split(self) -> None:
        self.assertEqual(len(self.samples), 18)
        self.assertEqual(len(self.groups), 6)
        self.assertTrue(all(row["usable"] == "true" for row in self.samples))
        self.assertFalse(any("sync_preview" in row["ir_path"] for row in self.samples))
        first = generate_nested_group_assignments(
            self.samples, self.groups, self.config, seed=42
        )
        second = generate_nested_group_assignments(
            self.samples, self.groups, self.config, seed=42
        )
        third = generate_nested_group_assignments(
            self.samples, self.groups, self.config, seed=43
        )
        self.assertEqual(first, second)
        counts = Counter(assignment.inner_split for assignment in first.values())
        self.assertEqual(counts, {"train": 4, "val": 1, "test": 1})
        self.assertEqual(
            Counter(assignment.inner_split for assignment in third.values()),
            {"train": 4, "val": 1, "test": 1},
        )
        sets = {
            split: {
                row["sample_id"]
                for row in read_csv_rows(self.split_dir / f"{split}.csv", ["sample_id"])
            }
            for split in ("train_pool", "train", "val", "test")
        }
        self.assertEqual(sets["train_pool"], sets["train"] | sets["val"])
        self.assertFalse(sets["train_pool"] & sets["test"])
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        for experiment_index in range(1, 7):
            experiment = f"experiment{experiment_index:03d}"
            experiment_splits = {
                first[row["leakage_group_id"]].inner_split
                for row in self.samples
                if row["experiment_id"] == experiment
            }
            self.assertEqual(len(experiment_splits), 1, experiment)

    def test_experiment_contract_rejects_missing_category_duplicate_clip_and_split_group(self) -> None:
        missing = [
            row
            for row in self.samples
            if not (
                row["experiment_id"] == "experiment001"
                and row["category"] == "sick"
            )
        ]
        with self.assertRaisesRegex(DataContractError, "类别不完整"):
            validate_experiment_contract(missing, self.config)

        duplicate = [dict(row) for row in self.samples]
        source = next(
            row
            for row in duplicate
            if row["experiment_id"] == "experiment001" and row["category"] == "health"
        )
        extra = dict(source)
        extra["sample_id"] += "__duplicate"
        extra["clip_id"] = "clip002"
        duplicate.append(extra)
        with self.assertRaisesRegex(DataContractError, "每类必须恰好"):
            validate_experiment_contract(duplicate, self.config)

        mixed_groups = [dict(row) for row in self.samples]
        next(
            row
            for row in mixed_groups
            if row["experiment_id"] == "experiment001" and row["category"] == "sick"
        )["leakage_group_id"] = "another-group"
        with self.assertRaisesRegex(DataContractError, "一个 leakage_group_id"):
            validate_experiment_contract(mixed_groups, self.config)

    def test_formal_split_rejects_fewer_than_six_groups(self) -> None:
        samples = [
            row for row in self.samples if row["experiment_id"] != "experiment006"
        ]
        groups = [row for row in self.groups if row["leakage_group_id"] != "group-6"]
        with self.assertRaisesRegex(DataContractError, "独立组不足"):
            generate_nested_group_assignments(samples, groups, self.config, seed=42)

    def test_two_experiments_sharing_a_group_move_together(self) -> None:
        samples = [dict(row) for row in self.samples]
        for row in samples:
            if row["experiment_id"] == "experiment006":
                row["leakage_group_id"] = "group-5"
        groups = [row for row in self.groups if row["leakage_group_id"] != "group-6"]
        config = copy.deepcopy(self.config)
        config["splitting"]["minimum_complete_groups"] = 5
        assignments = generate_nested_group_assignments(samples, groups, config, seed=42)
        splits = {
            assignments[row["leakage_group_id"]].inner_split
            for row in samples
            if row["experiment_id"] in {"experiment005", "experiment006"}
        }
        self.assertEqual(len(splits), 1)

    def test_epoch_shuffle_sampler_is_deterministic_without_replacement(self) -> None:
        sampler = EpochShuffleSampler(range(18), base_seed=42)
        sampler.set_epoch(0)
        epoch_zero = list(sampler)
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        resumed = EpochShuffleSampler(range(18), base_seed=42)
        resumed.set_epoch(1)
        self.assertEqual(epoch_one, list(resumed))
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(sorted(epoch_zero), list(range(18)))
        self.assertEqual(sorted(epoch_one), list(range(18)))

    def test_dataset_shapes_range_and_synchronized_crop(self) -> None:
        transform = PairedGeometryTransform(
            crop_size=32,
            horizontal_flip_probability=1.0,
            random_crop=True,
        )
        dataset = PairedFusionDataset(
            self.data_root,
            self.data_root / "manifests" / "samples.csv",
            self.split_dir / "train.csv",
            paired_transform=transform,
            config_path=self.data_root / "dataset.yaml",
            base_seed=7,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["vis"].shape), (3, 32, 32))
        self.assertEqual(tuple(sample["ir"].shape), (1, 32, 32))
        self.assertTrue(torch.allclose(sample["vis"][0], sample["ir"][0], atol=1e-6))
        self.assertTrue(torch.isfinite(sample["vis"]).all())
        self.assertGreaterEqual(float(sample["vis"].min()), 0.0)
        self.assertLessEqual(float(sample["vis"].max()), 1.0)

    def test_validation_preprocessing_is_deterministic(self) -> None:
        transform = PairedGeometryTransform(
            crop_size=32,
            horizontal_flip_probability=0.0,
            random_crop=False,
        )
        dataset = PairedFusionDataset(
            self.data_root,
            self.data_root / "manifests" / "samples.csv",
            self.split_dir / "val.csv",
            paired_transform=transform,
            config_path=self.data_root / "dataset.yaml",
        )
        first = dataset[0]
        second = dataset[0]
        self.assertTrue(torch.equal(first["vis"], second["vis"]))
        self.assertTrue(torch.equal(first["ir"], second["ir"]))

    def test_missing_group_mapping_is_rejected(self) -> None:
        empty = self.root / "empty_assignments.csv"
        write_csv_atomic(empty, [], GROUP_ASSIGNMENT_FIELDS)
        with self.assertRaisesRegex(DataContractError, "缺少人工"):
            build_manifest_rows(self.data_root, self.config, empty)

    def test_invalid_config_and_absolute_manifest_path_are_rejected(self) -> None:
        invalid_config = self.root / "invalid.yaml"
        config = dict(self.config)
        config["schema_version"] = 2
        invalid_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaises(DataContractError):
            load_dataset_config(invalid_config)

        malicious_rows = [dict(row) for row in self.samples]
        train_ids = {
            row["sample_id"]
            for row in read_csv_rows(self.split_dir / "train.csv", ["sample_id"])
        }
        selected = next(row for row in malicious_rows if row["sample_id"] in train_ids)
        selected["ir_path"] = r"C:\outside\000001.png"
        malicious_manifest = self.root / "malicious_samples.csv"
        write_csv_atomic(malicious_manifest, malicious_rows, SAMPLE_FIELDS)
        with self.assertRaisesRegex(DataContractError, "manifest"):
            PairedFusionDataset(
                self.data_root,
                malicious_manifest,
                self.split_dir / "train.csv",
                config_path=self.data_root / "dataset.yaml",
            )

    def test_missing_file_and_size_manifest_mismatch_are_rejected(self) -> None:
        broken_rows = [dict(row) for row in self.samples]
        train_ids = {
            row["sample_id"]
            for row in read_csv_rows(self.split_dir / "train.csv", ["sample_id"])
        }
        selected = next(row for row in broken_rows if row["sample_id"] in train_ids)
        selected["ir_path"] = selected["ir_path"].replace("000001.png", "missing.png")
        broken_manifest = self.root / "broken_samples.csv"
        write_csv_atomic(broken_manifest, broken_rows, SAMPLE_FIELDS)
        dataset = PairedFusionDataset(
            self.data_root,
            broken_manifest,
            self.split_dir / "train.csv",
            paired_transform=PairedGeometryTransform(crop_size=64, random_crop=False),
            config_path=self.data_root / "dataset.yaml",
        )
        broken_index = next(
            index for index, row in enumerate(dataset.samples) if row["sample_id"] == selected["sample_id"]
        )
        with self.assertRaises(DataContractError):
            dataset[broken_index]

        wrong_size_rows = [dict(row) for row in self.samples]
        selected = next(row for row in wrong_size_rows if row["sample_id"] in train_ids)
        selected["width"] = "999"
        wrong_manifest = self.root / "wrong_size_samples.csv"
        write_csv_atomic(wrong_manifest, wrong_size_rows, SAMPLE_FIELDS)
        dataset = PairedFusionDataset(
            self.data_root,
            wrong_manifest,
            self.split_dir / "train.csv",
            paired_transform=PairedGeometryTransform(crop_size=64, random_crop=False),
            config_path=self.data_root / "dataset.yaml",
        )
        wrong_index = next(
            index for index, row in enumerate(dataset.samples) if row["sample_id"] == selected["sample_id"]
        )
        with self.assertRaisesRegex(DataContractError, "manifest"):
            dataset[wrong_index]

    def test_full_audit_and_minimal_cpu_training_smoke(self) -> None:
        audit, registration, duplicates = audit_dataset(
            self.data_root, self.samples, self.groups, self.config, self.split_dir
        )
        self.assertEqual(audit["fatal_count"], 0)
        self.assertTrue(audit["formal_ready"])
        self.assertEqual(len(registration), 18)
        self.assertGreater(len(duplicates), 0)
        write_audit_outputs(
            self.data_root,
            audit,
            registration,
            duplicates,
            self.config,
            split_dir=self.split_dir,
            overwrite=True,
        )
        args = argparse.Namespace(
            data_root=self.data_root,
            config=self.data_root / "dataset.yaml",
            split_version="v1",
            output_root=self.root / "runs",
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            num_workers=0,
            seed=42,
            device="cpu",
            resume=None,
            no_amp=True,
            smoke_only=True,
            max_steps=1,
        )
        output = run_training(args)
        self.assertTrue((output / "smoke_result.json").is_file())
        self.assertFalse((output / "best.pt").exists())
        smoke = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
        self.assertEqual(smoke["loss_weights"], DEFAULT_LOSS_WEIGHTS)

    def test_audit_detects_cross_split_exact_duplicate_and_source_fingerprint(self) -> None:
        modified_samples = [dict(row) for row in self.samples]
        modified_groups = [dict(row) for row in self.groups]
        train_id = read_csv_rows(self.split_dir / "train.csv", ["sample_id"])[0]["sample_id"]
        val_id = read_csv_rows(self.split_dir / "val.csv", ["sample_id"])[0]["sample_id"]
        train_row = next(row for row in modified_samples if row["sample_id"] == train_id)
        val_row = next(row for row in modified_samples if row["sample_id"] == val_id)
        val_row["ir_path"] = train_row["ir_path"]
        val_row["ir_sha256"] = train_row["ir_sha256"]
        val_row["ir_phash"] = train_row["ir_phash"]
        train_group = next(
            row for row in modified_groups if row["leakage_group_id"] == train_row["leakage_group_id"]
        )
        val_group = next(
            row for row in modified_groups if row["leakage_group_id"] == val_row["leakage_group_id"]
        )
        val_group["rgb_source_fingerprint"] = train_group["rgb_source_fingerprint"]
        audit, _, _ = audit_dataset(
            self.data_root, modified_samples, modified_groups, self.config, self.split_dir
        )
        codes = {issue["code"] for issue in audit["fatals"]}
        self.assertIn("cross_split_exact_duplicate", codes)
        self.assertIn("cross_split_source_fingerprint", codes)


class ModelSmokeTests(unittest.TestCase):
    def test_loss_weight_validation_and_checkpoint_compatibility(self) -> None:
        defaults = _loss_weights_from_args(argparse.Namespace())
        self.assertEqual(defaults, DEFAULT_LOSS_WEIGHTS)
        custom = _loss_weights_from_args(
            argparse.Namespace(
                lambda_intensity=2,
                lambda_gradient=11,
                lambda_ssim=4,
                lambda_edge=1,
            )
        )
        self.assertEqual(custom, {"intensity": 2.0, "gradient": 11.0, "ssim": 4.0, "edge": 1.0})
        _validate_checkpoint_loss_weights({}, defaults)
        with self.assertRaises(DataContractError):
            _validate_checkpoint_loss_weights({}, custom)
        for kwargs in (
            {"lambda_intensity": -1, "lambda_gradient": 12, "lambda_ssim": 5, "lambda_edge": 2},
            {"lambda_intensity": float("nan"), "lambda_gradient": 11, "lambda_ssim": 5, "lambda_edge": 2},
            {"lambda_intensity": 0, "lambda_gradient": 0, "lambda_ssim": 0, "lambda_edge": 0},
        ):
            with self.assertRaises((DataContractError, ValueError)):
                _loss_weights_from_args(argparse.Namespace(**kwargs))

    def test_fusion_loss_default_custom_and_invalid_weights(self) -> None:
        generator = torch.Generator().manual_seed(29)
        fused = torch.rand(1, 3, 32, 32, generator=generator, requires_grad=True)
        ir = torch.rand(1, 1, 32, 32, generator=generator)
        vis = torch.rand(1, 3, 32, 32, generator=generator)
        default_loss, components = FusionLoss()(fused, ir, vis)
        expected = (
            components["intensity_loss"]
            + 10 * components["gradient_loss"]
            + 5 * components["ssim_loss"]
            + 2 * components["edge_loss"]
        )
        self.assertAlmostEqual(float(default_loss), expected, places=5)
        custom_loss, _ = FusionLoss(2, 11, 4, 1)(fused, ir, vis)
        custom_loss.backward()
        self.assertTrue(torch.isfinite(custom_loss))
        self.assertTrue(torch.isfinite(fused.grad).all())
        for weights in ((-1, 12, 5, 2), (float("nan"), 11, 5, 2), (0, 0, 0, 0)):
            with self.assertRaises(ValueError):
                FusionLoss(*weights)

    def test_gray_enhancer_is_identity_bounded_and_trainable(self) -> None:
        enhancer = GrayEnhancer().train()
        gray = torch.rand(2, 1, 32, 40)
        initial = enhancer(gray)
        self.assertTrue(torch.equal(initial, gray))
        self.assertGreaterEqual(float(initial.min()), 0.0)
        self.assertLessEqual(float(initial.max()), 1.0)

        target = torch.zeros_like(initial)
        torch.nn.functional.l1_loss(initial, target).backward()
        gradient = enhancer.residual_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_learned_gray_warm_start_matches_gray_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "baseline.pt"
            baseline = ResNetFusion(
                Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode="gray"
            ).eval()
            torch.save({"model": baseline.state_dict(), "epoch": 9}, checkpoint)
            candidate = ResNetFusion(
                Residual,
                DecoderBlock,
                FusionBlock,
                CrossAttention,
                ir_mode="learned_gray",
            ).eval()
            metadata = load_initial_model_checkpoint(
                candidate, checkpoint, ir_mode="learned_gray"
            )
            self.assertTrue(metadata["missing_keys"])
            self.assertTrue(
                all(
                    key.startswith("ir_encoder.enhancer.")
                    for key in metadata["missing_keys"]
                )
            )
            generator = torch.Generator().manual_seed(17)
            vis = torch.rand(1, 3, 64, 64, generator=generator)
            ir = torch.rand(1, 1, 64, 64, generator=generator)
            with torch.inference_mode():
                baseline_output = baseline(vis, ir)
                candidate_output = candidate(vis, ir)
            self.assertTrue(torch.equal(baseline_output, candidate_output))

    def test_tiny_cpu_forward_and_loss_backward(self) -> None:
        model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention).train()
        criterion = FusionLoss()
        vis = torch.rand(1, 3, 64, 64)
        ir = torch.rand(1, 1, 64, 64)
        fused = model(vis, ir)
        self.assertEqual(tuple(fused.shape), (1, 3, 64, 64))
        loss, _ = criterion(fused, ir, vis)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
