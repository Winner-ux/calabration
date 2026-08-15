"""融合推理的图像契约、整图 padding 与加权滑窗工具。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from data_pipeline.schema import DataContractError
from data_pipeline.transforms import bt601_to_gray
from model import IR_MODES


def _validate_pair_tensors(vis: torch.Tensor, ir: torch.Tensor) -> None:
    if vis.ndim != 4 or vis.shape[0] != 1 or vis.shape[1] != 3:
        raise DataContractError(f"VIS 推理输入必须是 [1,3,H,W]，实际为 {tuple(vis.shape)}")
    if ir.ndim != 4 or ir.shape[0] != 1 or ir.shape[1] != 1:
        raise DataContractError(f"IR 推理输入必须是 [1,1,H,W]，实际为 {tuple(ir.shape)}")
    if vis.shape[-2:] != ir.shape[-2:]:
        raise DataContractError(
            f"VIS/IR 推理尺寸必须一致: {tuple(vis.shape)} vs {tuple(ir.shape)}"
        )
    if not torch.isfinite(vis).all() or not torch.isfinite(ir).all():
        raise DataContractError("VIS/IR 推理输入包含非有限值")
    if float(vis.min()) < 0.0 or float(vis.max()) > 1.0:
        raise DataContractError("VIS 推理输入必须位于 [0,1]")
    if float(ir.min()) < 0.0 or float(ir.max()) > 1.0:
        raise DataContractError("IR 推理输入必须位于 [0,1]")


def _load_rgb_png(path: Path, expected_modes: set[str], modality: str) -> torch.Tensor:
    if path.suffix.lower() != ".png":
        raise DataContractError(f"schema v1 的 {modality} 推理输入必须是 PNG: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise DataContractError(f"文件扩展名为 PNG 但实际格式不是 PNG: {path}")
            if image.mode not in expected_modes:
                raise DataContractError(
                    f"{modality} 图像模式必须属于 {sorted(expected_modes)}，实际为 {image.mode}: {path}"
                )
            array = np.array(image, dtype=np.uint8, copy=True)
    except (OSError, ValueError) as exc:
        raise DataContractError(f"无法读取 {modality} 图像 {path}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise DataContractError(f"{modality} 图像必须为三通道 RGB: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def load_inference_pair(
    vis_path: str | Path,
    ir_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 schema v1 读取同尺寸 PNG，返回 `[1,3,H,W]` 和 `[1,1,H,W]`。"""
    vis_file = Path(vis_path)
    ir_file = Path(ir_path)
    if not vis_file.is_file() or not ir_file.is_file():
        raise DataContractError(f"推理图像不存在: VIS={vis_file}, IR={ir_file}")
    if vis_file.stem != ir_file.stem:
        raise DataContractError(
            f"VIS/IR 文件 stem 必须一致: {vis_file.name} vs {ir_file.name}"
        )
    contract = config["image_contract"]
    vis = _load_rgb_png(
        vis_file, {str(contract.get("rgb_source_mode", "RGB"))}, "VIS"
    )
    ir_rgb = _load_rgb_png(
        ir_file, {str(mode) for mode in contract.get("ir_source_modes", ["RGB"])}, "IR"
    )
    if vis.shape[-2:] != ir_rgb.shape[-2:]:
        raise DataContractError(
            f"VIS/IR 图像尺寸必须一致: {tuple(vis.shape[-2:])} vs {tuple(ir_rgb.shape[-2:])}"
        )
    ir = bt601_to_gray(ir_rgb)
    vis = vis.unsqueeze(0).contiguous()
    ir = ir.unsqueeze(0).contiguous()
    _validate_pair_tensors(vis, ir)
    return vis, ir


def save_fused_png(tensor: torch.Tensor, path: str | Path) -> None:
    """将 `[1,3,H,W]` 或 `[3,H,W]` 的 `[0,1]` 张量保存为 8-bit RGB PNG。"""
    image = tensor.detach().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise DataContractError(f"融合输出必须是 [1,3,H,W] 或 [3,H,W]: {tuple(tensor.shape)}")
    if not torch.isfinite(image).all():
        raise DataContractError("融合输出包含非有限值")
    array = (
        image.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output, format="PNG")


def _right_bottom_pad(
    tensor: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_bottom = target_height - height
    pad_right = target_width - width
    if pad_bottom < 0 or pad_right < 0:
        raise DataContractError("目标 padding 尺寸不能小于输入尺寸")
    if not pad_bottom and not pad_right:
        return tensor
    selected_mode = mode
    if mode == "reflect" and (pad_bottom >= height or pad_right >= width):
        selected_mode = "replicate"
    if selected_mode not in {"reflect", "replicate"}:
        raise DataContractError(f"不支持的 padding 模式: {mode}")
    return torch_functional.pad(
        tensor, (0, pad_right, 0, pad_bottom), mode=selected_mode
    )


def pad_pair_to_multiple(
    vis: torch.Tensor,
    ir: torch.Tensor,
    factor: int,
    mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    _validate_pair_tensors(vis, ir)
    if factor <= 0:
        raise DataContractError("model_downsample_factor 必须为正数")
    height, width = vis.shape[-2:]
    target_height = math.ceil(height / factor) * factor
    target_width = math.ceil(width / factor) * factor
    return (
        _right_bottom_pad(vis, target_height, target_width, mode),
        _right_bottom_pad(ir, target_height, target_width, mode),
        (height, width),
    )


def window_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """返回覆盖完整一维范围、且最后一个窗口贴合边缘的起点。"""
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise DataContractError("length、tile_size 和 stride 必须为正数")
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def hann_weight(tile_size: int, minimum: float = 1e-3) -> torch.Tensor:
    if tile_size <= 0 or not 0.0 < minimum <= 1.0:
        raise DataContractError("tile_size 必须为正，Hann 最小权重必须位于 (0,1]")
    one_dimensional = torch.hann_window(tile_size, periodic=False, dtype=torch.float32)
    return torch.outer(one_dimensional, one_dimensional).clamp_min_(minimum)[None, None]


def _validate_prediction(prediction: torch.Tensor, expected_hw: tuple[int, int]) -> None:
    if prediction.ndim != 4 or prediction.shape[:2] != (1, 3):
        raise DataContractError(
            f"模型输出必须是 [1,3,H,W]，实际为 {tuple(prediction.shape)}"
        )
    if prediction.shape[-2:] != expected_hw:
        raise DataContractError(
            f"模型输出尺寸必须等于输入尺寸: {tuple(prediction.shape[-2:])} vs {expected_hw}"
        )
    if not torch.isfinite(prediction).all():
        raise DataContractError("模型输出包含非有限值")


def full_image_inference(
    model: torch.nn.Module,
    vis: torch.Tensor,
    ir: torch.Tensor,
    *,
    device: torch.device,
    factor: int = 32,
    pad_mode: str = "reflect",
    amp_enabled: bool = False,
) -> torch.Tensor:
    padded_vis, padded_ir, (height, width) = pad_pair_to_multiple(
        vis, ir, factor, pad_mode
    )
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            prediction = model(padded_vis.to(device), padded_ir.to(device))
    _validate_prediction(prediction, tuple(padded_vis.shape[-2:]))
    return prediction[..., :height, :width].float().cpu()


def sliding_window_inference(
    model: torch.nn.Module,
    vis: torch.Tensor,
    ir: torch.Tensor,
    *,
    device: torch.device,
    tile_size: int = 448,
    overlap: int = 64,
    factor: int = 32,
    pad_mode: str = "reflect",
    amp_enabled: bool = False,
) -> torch.Tensor:
    """逐 tile 推理，在 CPU 上用非零 Hann 权重进行 FP32 累计。"""
    _validate_pair_tensors(vis, ir)
    if tile_size <= 0 or tile_size % factor:
        raise DataContractError("tile_size 必须为正数且能被 model_downsample_factor 整除")
    if overlap < 0 or overlap >= tile_size:
        raise DataContractError("overlap 必须满足 0 <= overlap < tile_size")
    height, width = vis.shape[-2:]
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded_vis = _right_bottom_pad(vis, padded_height, padded_width, pad_mode)
    padded_ir = _right_bottom_pad(ir, padded_height, padded_width, pad_mode)
    stride = tile_size - overlap
    y_positions = window_positions(padded_height, tile_size, stride)
    x_positions = window_positions(padded_width, tile_size, stride)
    weight = hann_weight(tile_size)
    accumulated = torch.zeros((1, 3, padded_height, padded_width), dtype=torch.float32)
    normalization = torch.zeros((1, 1, padded_height, padded_width), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for top in y_positions:
            for left in x_positions:
                vis_tile = padded_vis[..., top : top + tile_size, left : left + tile_size]
                ir_tile = padded_ir[..., top : top + tile_size, left : left + tile_size]
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    prediction = model(vis_tile.to(device), ir_tile.to(device))
                _validate_prediction(prediction, (tile_size, tile_size))
                prediction_cpu = prediction.float().cpu()
                accumulated[..., top : top + tile_size, left : left + tile_size] += (
                    prediction_cpu * weight
                )
                normalization[..., top : top + tile_size, left : left + tile_size] += weight
    if float(normalization.min()) <= 0.0:
        raise DataContractError("滑窗累计存在未覆盖像素")
    fused = accumulated / normalization
    return fused[..., :height, :width]


def estimate_shallow_tokens(height: int, width: int, factor: int = 32) -> int:
    """估算 pad 后首个 1/4 分辨率融合层的 token 数。"""
    if height <= 0 or width <= 0 or factor <= 0:
        raise DataContractError("height、width 和 factor 必须为正数")
    padded_height = math.ceil(height / factor) * factor
    padded_width = math.ceil(width / factor) * factor
    return math.ceil(padded_height / 4) * math.ceil(padded_width / 4)


def infer_pair_tensor(
    model: torch.nn.Module,
    vis: torch.Tensor,
    ir: torch.Tensor,
    *,
    device: torch.device,
    requested_mode: str = "auto",
    tile_size: int = 448,
    overlap: int = 64,
    factor: int = 32,
    pad_mode: str = "reflect",
    max_full_tokens: int = 12_544,
    force_full: bool = False,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _validate_pair_tensors(vis, ir)
    if requested_mode not in {"auto", "full", "sliding"}:
        raise DataContractError(f"不支持的推理模式: {requested_mode}")
    if max_full_tokens <= 0:
        raise DataContractError("max_full_tokens 必须为正数")
    height, width = vis.shape[-2:]
    tokens = estimate_shallow_tokens(height, width, factor)
    actual_mode = requested_mode
    fallback_reason: str | None = None
    if requested_mode == "auto":
        if tokens > max_full_tokens:
            actual_mode = "sliding"
            fallback_reason = (
                f"首层注意力 token 数 {tokens} 超过安全阈值 {max_full_tokens}"
            )
        else:
            actual_mode = "full"
    if actual_mode == "full" and tokens > max_full_tokens and not force_full:
        raise DataContractError(
            f"full 模式首层注意力 token 数 {tokens} 超过安全阈值 {max_full_tokens}；"
            "请使用 sliding/auto，或显式传入 --force-full 承担 OOM 风险"
        )
    if actual_mode == "full":
        fused = full_image_inference(
            model,
            vis,
            ir,
            device=device,
            factor=factor,
            pad_mode=pad_mode,
            amp_enabled=amp_enabled,
        )
    else:
        fused = sliding_window_inference(
            model,
            vis,
            ir,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            factor=factor,
            pad_mode=pad_mode,
            amp_enabled=amp_enabled,
        )
    details = {
        "requested_mode": requested_mode,
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "input_height": height,
        "input_width": width,
        "output_height": int(fused.shape[-2]),
        "output_width": int(fused.shape[-1]),
        "shallow_attention_tokens": tokens,
    }
    return fused, details


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DataContractError(f"checkpoint 不存在: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DataContractError(f"无法安全读取 checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DataContractError(f"checkpoint 根节点必须是映射: {path}")
    return payload


def _checkpoint_state(payload: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(payload.get("model"), Mapping):
        state_dict = payload["model"]
        layout = "training_checkpoint"
    elif isinstance(payload.get("state_dict"), Mapping):
        state_dict = payload["state_dict"]
        layout = "state_dict_wrapper"
    elif payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state_dict = payload
        layout = "raw_state_dict"
    else:
        raise DataContractError("checkpoint 不包含可识别的 model/state_dict 张量映射")
    state_dict = dict(state_dict)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict, layout


def resolve_checkpoint_ir_mode(
    checkpoint_path: str | Path,
    requested_mode: str = "auto",
) -> str:
    """从 checkpoint 推断 IR 模式，并拒绝显式模式冲突。"""
    if requested_mode not in ("auto", *IR_MODES):
        raise DataContractError(f"非法 IR 模式: {requested_mode!r}")
    path = Path(checkpoint_path).resolve()
    payload = _checkpoint_payload(path)
    run_config = payload.get("run_config")
    stored_mode = "gray"
    if isinstance(run_config, Mapping):
        stored_mode = str(run_config.get("ir_mode", "gray"))
    if stored_mode not in IR_MODES:
        raise DataContractError(f"checkpoint 包含未知 ir_mode: {stored_mode!r}")
    if requested_mode != "auto" and requested_mode != stored_mode:
        raise DataContractError(
            f"显式 ir_mode 与 checkpoint 不一致: requested={requested_mode}, "
            f"checkpoint={stored_mode}"
        )
    return stored_mode


def load_initial_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    ir_mode: str,
) -> dict[str, Any]:
    """只加载模型权重；learned_gray 仅允许缺少新增增强器参数。"""
    if ir_mode not in IR_MODES:
        raise DataContractError(f"非法 IR 模式: {ir_mode!r}")
    path = Path(checkpoint_path).resolve()
    payload = _checkpoint_payload(path)
    state_dict, layout = _checkpoint_state(payload)
    try:
        if ir_mode == "gray":
            model.load_state_dict(state_dict, strict=True)
            missing_keys: list[str] = []
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys = list(incompatible.missing_keys)
            invalid_missing = [
                key for key in missing_keys if not key.startswith("ir_encoder.enhancer.")
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise DataContractError(
                    "learned_gray 初始化仅允许缺少 ir_encoder.enhancer.*；"
                    f"invalid_missing={invalid_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
    except RuntimeError as exc:
        raise DataContractError(f"初始化 checkpoint 与当前模型结构不兼容: {exc}") from exc
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "layout": layout,
        "source_epoch": payload.get("epoch"),
        "missing_keys": missing_keys,
    }


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """严格加载训练 checkpoint 或纯 state_dict，并返回可审计的来源信息。"""
    path = Path(checkpoint_path).resolve()
    payload = _checkpoint_payload(path)
    state_dict, layout = _checkpoint_state(payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise DataContractError(f"checkpoint 与当前模型结构不兼容: {exc}") from exc
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "layout": layout,
        "epoch": payload.get("epoch"),
        "run_config": payload.get("run_config"),
    }
