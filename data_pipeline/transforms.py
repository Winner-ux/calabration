"""IR 转换和成对、可复现的几何增强。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .schema import DataContractError


def bt601_to_gray(ir_rgb: torch.Tensor) -> torch.Tensor:
    """将 `[3,H,W]` RGB IR 按 BT.601 转换为 `[1,H,W]`。"""
    if ir_rgb.ndim != 3 or ir_rgb.shape[0] != 3:
        raise DataContractError(f"BT.601 输入必须是 [3,H,W]，实际为 {tuple(ir_rgb.shape)}")
    weights = ir_rgb.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (ir_rgb * weights).sum(dim=0, keepdim=True)


def _pad_pair_to_minimum(
    vis: torch.Tensor,
    ir: torch.Tensor,
    min_height: int,
    min_width: int,
    mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = vis.shape[-2:]
    pad_h = max(0, min_height - height)
    pad_w = max(0, min_width - width)
    if not pad_h and not pad_w:
        return vis, ir
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padding = (left, right, top, bottom)
    selected_mode = mode
    if mode == "reflect" and (
        left >= width or right >= width or top >= height or bottom >= height
    ):
        selected_mode = "replicate"
    return (
        torch_functional.pad(vis, padding, mode=selected_mode),
        torch_functional.pad(ir, padding, mode=selected_mode),
    )


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item())


def _uniform(low: float, high: float, generator: torch.Generator) -> float:
    return low + (high - low) * _rand(generator)


@dataclass(frozen=True)
class PairedGeometryTransform:
    """共享随机参数的裁剪、翻转和轻微旋转。"""

    crop_size: int = 448
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    rotation_probability: float = 0.0
    max_rotation_degrees: float = 5.0
    pad_mode: str = "reflect"
    random_crop: bool = True

    def __call__(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vis.shape[-2:] != ir.shape[-2:]:
            raise DataContractError(
                f"同步增强要求 VIS/IR 尺寸一致: {tuple(vis.shape)} vs {tuple(ir.shape)}"
            )
        vis, ir = _pad_pair_to_minimum(
            vis, ir, self.crop_size, self.crop_size, self.pad_mode
        )
        height, width = vis.shape[-2:]
        if self.random_crop:
            top = int(torch.randint(0, height - self.crop_size + 1, (), generator=generator).item())
            left = int(torch.randint(0, width - self.crop_size + 1, (), generator=generator).item())
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        vis = vis[:, top : top + self.crop_size, left : left + self.crop_size]
        ir = ir[:, top : top + self.crop_size, left : left + self.crop_size]

        if self.rotation_probability > 0 and _rand(generator) < self.rotation_probability:
            angle = _uniform(-self.max_rotation_degrees, self.max_rotation_degrees, generator)
            # 小角度旋转前共享 padding，减少人工黑边。
            margin = max(1, math.ceil(self.crop_size * math.sin(math.radians(abs(angle)))))
            padding = (margin, margin, margin, margin)
            mode = self.pad_mode
            if mode == "reflect" and margin >= self.crop_size:
                mode = "replicate"
            vis = torch_functional.pad(vis, padding, mode=mode)
            ir = torch_functional.pad(ir, padding, mode=mode)
            vis = vision_functional.rotate(vis, angle, InterpolationMode.BILINEAR)
            ir = vision_functional.rotate(ir, angle, InterpolationMode.BILINEAR)
            vis = vision_functional.center_crop(vis, [self.crop_size, self.crop_size])
            ir = vision_functional.center_crop(ir, [self.crop_size, self.crop_size])

        if _rand(generator) < self.horizontal_flip_probability:
            vis = vision_functional.hflip(vis)
            ir = vision_functional.hflip(ir)
        if _rand(generator) < self.vertical_flip_probability:
            vis = vision_functional.vflip(vis)
            ir = vision_functional.vflip(ir)
        return vis.contiguous(), ir.contiguous()


@dataclass(frozen=True)
class RGBColorTransform:
    """只作用于 VIS 的轻量颜色增强。"""

    probability: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.02

    def __call__(self, vis: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        if _rand(generator) >= self.probability:
            return vis
        brightness = _uniform(1 - self.brightness, 1 + self.brightness, generator)
        contrast = _uniform(1 - self.contrast, 1 + self.contrast, generator)
        saturation = _uniform(1 - self.saturation, 1 + self.saturation, generator)
        hue = _uniform(-self.hue, self.hue, generator)
        vis = vision_functional.adjust_brightness(vis, brightness)
        vis = vision_functional.adjust_contrast(vis, contrast)
        vis = vision_functional.adjust_saturation(vis, saturation)
        vis = vision_functional.adjust_hue(vis, hue)
        return vis.clamp_(0.0, 1.0)


def build_paired_transform(config: Mapping[str, Any], split: str) -> PairedGeometryTransform:
    preprocessing = config["preprocessing"]
    crop_size = int(preprocessing["crop_size"])
    if split == "train":
        settings = preprocessing["train"]
        return PairedGeometryTransform(
            crop_size=crop_size,
            horizontal_flip_probability=float(settings.get("horizontal_flip_probability", 0.0)),
            vertical_flip_probability=float(settings.get("vertical_flip_probability", 0.0)),
            rotation_probability=float(settings.get("rotation_probability", 0.0)),
            max_rotation_degrees=float(settings.get("max_rotation_degrees", 0.0)),
            pad_mode=str(preprocessing.get("pad_mode", "reflect")),
            random_crop=bool(settings.get("random_crop", True)),
        )
    return PairedGeometryTransform(
        crop_size=crop_size,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotation_probability=0.0,
        max_rotation_degrees=0.0,
        pad_mode=str(preprocessing.get("pad_mode", "reflect")),
        random_crop=False,
    )


def build_rgb_transform(config: Mapping[str, Any], split: str) -> RGBColorTransform | None:
    if split != "train":
        return None
    settings = config["preprocessing"].get("rgb_augmentation", {})
    if not settings.get("enabled", False):
        return None
    return RGBColorTransform(
        probability=float(settings.get("probability", 0.5)),
        brightness=float(settings.get("brightness", 0.1)),
        contrast=float(settings.get("contrast", 0.1)),
        saturation=float(settings.get("saturation", 0.1)),
        hue=float(settings.get("hue", 0.02)),
    )

