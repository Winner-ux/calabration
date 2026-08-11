"""Manifest 驱动的 PyTorch IR/RGB 融合 Dataset。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from data_pipeline.schema import (
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    parse_bool,
    read_csv_rows,
    resolve_data_path,
)
from data_pipeline.transforms import bt601_to_gray


class PairedFusionDataset(Dataset[dict[str, Any]]):
    """按 samples manifest 和 split manifest 加载配准的 VIS/IR 图像对。"""

    def __init__(
        self,
        data_root: str | Path,
        samples_manifest: str | Path,
        split_manifest: str | Path,
        paired_transform: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
        rgb_transform: Callable[..., torch.Tensor] | None = None,
        ir_conversion: str = "bt601",
        *,
        config_path: str | Path | None = None,
        base_seed: int = 0,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.config_path = Path(config_path) if config_path else self.data_root / "dataset.yaml"
        self.config = load_dataset_config(self.config_path)
        configured_conversion = self.config["image_contract"]["ir_conversion"]
        if ir_conversion != configured_conversion or ir_conversion != "bt601":
            raise DataContractError(
                f"IR 转换策略不一致: requested={ir_conversion}, configured={configured_conversion}"
            )
        rows = read_csv_rows(samples_manifest, SAMPLE_FIELDS)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise DataContractError("samples manifest 包含重复 sample_id")
        id_to_row = {row["sample_id"]: row for row in rows}
        split_rows = read_csv_rows(split_manifest, ["sample_id"])
        split_ids = [row["sample_id"].strip() for row in split_rows]
        if not split_ids:
            raise DataContractError(f"split manifest 为空: {split_manifest}")
        if len(split_ids) != len(set(split_ids)):
            raise DataContractError(f"split manifest 包含重复 sample_id: {split_manifest}")
        unknown = sorted(set(split_ids) - set(id_to_row))
        if unknown:
            raise DataContractError(f"split 引用了 samples.csv 中不存在的 ID: {unknown}")
        selected = [id_to_row[sample_id] for sample_id in split_ids]
        excluded = [row["sample_id"] for row in selected if not parse_bool(row["usable"])]
        if excluded:
            raise DataContractError(f"split 包含 usable=false 的样本: {excluded}")
        for row in selected:
            if not row.get("leakage_group_id"):
                raise DataContractError(f"样本缺少 leakage_group_id: {row['sample_id']}")
            resolve_data_path(self.data_root, row["ir_path"])
            resolve_data_path(self.data_root, row["rgb_path"])
        self.samples = selected
        self.paired_transform = paired_transform
        self.rgb_transform = rgb_transform
        self.ir_conversion = ir_conversion
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """使每个 epoch 的增强不同，同时保持同 seed 可复现。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _generator(self, sample_id: str, salt: str) -> torch.Generator:
        payload = f"{self.base_seed}:{self.epoch}:{sample_id}:{salt}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return torch.Generator().manual_seed(seed)

    def _load_pair(self, row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        ir_path = resolve_data_path(self.data_root, row["ir_path"])
        rgb_path = resolve_data_path(self.data_root, row["rgb_path"])
        if ir_path.stem != rgb_path.stem:
            raise DataContractError(
                f"IR/RGB stem 不一致: {ir_path.name} vs {rgb_path.name}"
            )
        try:
            with Image.open(rgb_path) as rgb_image:
                rgb_image.load()
                rgb_mode = rgb_image.mode
                rgb_size = rgb_image.size
                if rgb_mode != self.config["image_contract"]["rgb_source_mode"]:
                    raise DataContractError(
                        f"RGB 模式不符合 schema v1: {rgb_path} mode={rgb_mode}"
                    )
                vis = pil_to_tensor(rgb_image).to(torch.float32).div_(255.0)
            with Image.open(ir_path) as ir_image:
                ir_image.load()
                ir_mode = ir_image.mode
                ir_size = ir_image.size
                if ir_mode not in self.config["image_contract"]["ir_source_modes"]:
                    raise DataContractError(
                        f"IR 模式不符合 schema v1: {ir_path} mode={ir_mode}; 原生灰度/16-bit 需要升级 schema"
                    )
                ir_rgb = pil_to_tensor(ir_image).to(torch.float32).div_(255.0)
        except (OSError, UnidentifiedImageError) as exc:
            raise DataContractError(
                f"图像读取失败: RGB={rgb_path}, IR={ir_path}, error={exc}"
            ) from exc
        if rgb_size != ir_size:
            raise DataContractError(
                f"IR/RGB 空间尺寸不一致: RGB={rgb_path} {rgb_size}, IR={ir_path} {ir_size}"
            )
        if row.get("width") and row.get("height"):
            expected = (int(row["width"]), int(row["height"]))
            if rgb_size != expected:
                raise DataContractError(
                    f"图像尺寸与 manifest 不一致: {row['sample_id']} expected={expected}, actual={rgb_size}"
                )
        return vis, bt601_to_gray(ir_rgb)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        vis, ir = self._load_pair(row)
        if self.paired_transform is not None:
            vis, ir = self.paired_transform(
                vis,
                ir,
                generator=self._generator(row["sample_id"], "paired"),
            )
        if self.rgb_transform is not None:
            vis = self.rgb_transform(
                vis,
                generator=self._generator(row["sample_id"], "rgb"),
            )
        if vis.ndim != 3 or vis.shape[0] != 3:
            raise DataContractError(f"VIS 输出必须是 [3,H,W]: {row['sample_id']} -> {tuple(vis.shape)}")
        if ir.ndim != 3 or ir.shape[0] != 1:
            raise DataContractError(f"IR 输出必须是 [1,H,W]: {row['sample_id']} -> {tuple(ir.shape)}")
        if vis.shape[-2:] != ir.shape[-2:]:
            raise DataContractError(f"增强后 VIS/IR 尺寸不一致: {row['sample_id']}")
        factor = int(self.config["preprocessing"]["model_downsample_factor"])
        height, width = vis.shape[-2:]
        if height % factor or width % factor:
            raise DataContractError(
                f"输出尺寸必须能被 {factor} 整除: {row['sample_id']} -> {(height, width)}"
            )
        if not torch.isfinite(vis).all() or not torch.isfinite(ir).all():
            raise DataContractError(f"Dataset 输出包含 NaN/Inf: {row['sample_id']}")
        if vis.min() < 0 or vis.max() > 1 or ir.min() < 0 or ir.max() > 1:
            raise DataContractError(f"Dataset 输出超出 [0,1]: {row['sample_id']}")
        return {
            "vis": vis,
            "ir": ir,
            "sample_id": row["sample_id"],
            "leakage_group_id": row["leakage_group_id"],
            "category": row["category"],
        }


__all__ = ["PairedFusionDataset"]
