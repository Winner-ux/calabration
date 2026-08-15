"""不依赖 ground truth 的融合质量代理指标。"""

from __future__ import annotations

import math

import torch

from .loss import SSIMLoss


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"RGB 输入必须是 [B,3,H,W]，实际为 {tuple(image.shape)}")
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _entropy_8bit(gray: torch.Tensor) -> float:
    values = torch.clamp((gray.detach().float() * 255.0).round(), 0, 255)
    histogram = torch.histc(values.cpu(), bins=256, min=0, max=255)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * torch.log2(probabilities)).sum())


def fusion_quality_metrics(
    fused: torch.Tensor,
    ir: torch.Tensor,
    vis: torch.Tensor,
    *,
    ssim: SSIMLoss,
) -> dict[str, float]:
    """计算单样本质量代理；输入 batch size 必须为 1。"""
    if fused.shape[0] != 1 or ir.shape[0] != 1 or vis.shape[0] != 1:
        raise ValueError("fusion_quality_metrics 只接受 batch size 1")
    fused_float = fused.detach().float()
    ir_float = ir.detach().float()
    vis_float = vis.detach().float()
    fused_gray = rgb_to_gray(fused_float)
    fused_gray_3 = fused_gray.expand(-1, 3, -1, -1)
    ir_3 = ir_float.expand(-1, 3, -1, -1)
    row_frequency = torch.mean((fused_gray[..., 1:, :] - fused_gray[..., :-1, :]) ** 2)
    column_frequency = torch.mean((fused_gray[..., :, 1:] - fused_gray[..., :, :-1]) ** 2)
    dx = fused_gray[..., :, 1:] - fused_gray[..., :, :-1]
    dy = fused_gray[..., 1:, :] - fused_gray[..., :-1, :]
    common_height = min(dx.shape[-2], dy.shape[-2])
    common_width = min(dx.shape[-1], dy.shape[-1])
    average_gradient = torch.sqrt(
        (
            dx[..., :common_height, :common_width] ** 2
            + dy[..., :common_height, :common_width] ** 2
        )
        / 2.0
        + 1e-12
    ).mean()
    return {
        "ir_ssim": float(ssim(fused_gray_3, ir_3)),
        "vis_ssim": float(ssim(fused_float, vis_float)),
        "entropy": _entropy_8bit(fused_gray),
        "standard_deviation": float(fused_gray.std(unbiased=False)),
        "average_gradient": float(average_gradient),
        "spatial_frequency": math.sqrt(float(row_frequency + column_frequency)),
    }


__all__ = ["fusion_quality_metrics", "rgb_to_gray"]
