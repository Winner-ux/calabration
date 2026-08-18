"""批量加载已有标定模型，将 IR 图像配准到对应 RGB 图像坐标系。"""

from datetime import datetime
import os

import cv2
import numpy as np
import yaml

from config import (
    APPLY_IR_OUTPUT_PATH,
    APPLY_IR_PATH,
    APPLY_OUTPUT_EXTENSION,
    APPLY_OUTPUT_PATH,
    APPLY_PREVIEW_OUTPUT_PATH,
    APPLY_PREVIEW_RGB_WEIGHT,
    APPLY_REPORT_PATH,
    APPLY_RGB_OUTPUT_PATH,
    APPLY_RGB_PATH,
    CALIBRATION_MODEL_PATH,
    IMAGE_EXTENSIONS,
    MIN_VALID_WARP_RATIO,
    SYNC_DIAGNOSTIC_MAX_DIM,
    SYNC_MAX_LAG_FRAMES,
)
from registration import (
    calibration_reference_sizes,
    ensure_3ch,
    load_calibration,
    runtime_homography,
    warp_ir_to_rgb,
)


def _discover_images(folder):
    """以不区分大小写的文件名主干索引目录中的图像。"""
    os.makedirs(folder, exist_ok=True)
    images = {}
    duplicates = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        key = os.path.splitext(name)[0].casefold()
        if key in images:
            duplicates.setdefault(key, [images[key]]).append(path)
        else:
            images[key] = path
    if duplicates:
        details = "; ".join(
            f"{key}: {', '.join(paths)}" for key, paths in duplicates.items()
        )
        raise ValueError(f"同一目录中存在主文件名重复的图像：{details}")
    return images


def _validate_homography(H):
    if H.shape != (3, 3):
        raise ValueError(f"Homography 维度应为 3x3，实际为 {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("Homography 包含 NaN 或无穷值")
    if abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError("Homography 不可逆或接近奇异矩阵")


def _to_preview_8bit(image):
    """仅为预览转换动态范围，不改变分类保存的原始结果。"""
    image = ensure_3ch(image)
    if image.dtype == np.uint8:
        return image
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, np.uint8)
    values = image[finite]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(image.shape, np.uint8)
    normalized = (image.astype(np.float64) - low) * (255.0 / (high - low))
    normalized[~finite] = 0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _write_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise OSError(f"图像保存失败：{path}")


def _save_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)


def _edge_map(image):
    gray = cv2.cvtColor(_to_preview_8bit(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray))
    low = max(20, int(median * 0.33))
    high = max(low + 20, int(median * 0.90))
    return cv2.Canny(gray, low, high)


def _write_pair_diagnostics(folder, key, rgb, aligned_ir, valid_mask, fusion):
    """保存融合、双边缘和彩色双边缘诊断图。"""
    rgb_edges = _edge_map(rgb)
    ir_edges = _edge_map(aligned_ir)
    side_edges = np.hstack([rgb_edges, ir_edges])
    color_edges = (_to_preview_8bit(rgb).astype(np.float32) * 0.30).astype(np.uint8)
    color_edges[rgb_edges > 0] = (0, 255, 0)
    color_edges[ir_edges > 0] = (0, 0, 255)
    overlap = (rgb_edges > 0) & (ir_edges > 0) & (valid_mask > 0)
    color_edges[overlap] = (0, 255, 255)
    color_edges[valid_mask == 0] = _to_preview_8bit(rgb)[valid_mask == 0]
    paths = {
        "fusion": os.path.join(folder, f"{key}_fusion.png"),
        "edge_pair": os.path.join(folder, f"{key}_edges.png"),
        "colored_dual_edge": os.path.join(folder, f"{key}_dual_edges.png"),
    }
    _write_image(paths["fusion"], fusion)
    _write_image(paths["edge_pair"], side_edges)
    _write_image(paths["colored_dual_edge"], color_edges)
    return {name: os.path.abspath(path) for name, path in paths.items()}


def _motion_frame(image):
    gray = cv2.cvtColor(_to_preview_8bit(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1.0, SYNC_DIAGNOSTIC_MAX_DIM / max(width, height))
    if scale < 1.0:
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray.astype(np.float32)


def _lag_correlations(rgb_signal, ir_signal):
    rgb_signal = np.asarray(rgb_signal, np.float64)
    ir_signal = np.asarray(ir_signal, np.float64)
    values = {}
    for lag in range(-SYNC_MAX_LAG_FRAMES, SYNC_MAX_LAG_FRAMES + 1):
        if lag >= 0:
            rgb_part = rgb_signal[:len(rgb_signal) - lag or None]
            ir_part = ir_signal[lag:]
        else:
            rgb_part = rgb_signal[-lag:]
            ir_part = ir_signal[:len(ir_signal) + lag]
        if len(rgb_part) < 3 or np.std(rgb_part) == 0 or np.std(ir_part) == 0:
            correlation = None
        else:
            correlation = float(np.corrcoef(rgb_part, ir_part)[0, 1])
        values[str(lag)] = correlation
    finite = [(int(lag), value) for lag, value in values.items() if value is not None]
    best_lag, peak = max(finite, key=lambda pair: pair[1]) if finite else (None, None)
    return {
        "best_lag_frames": best_lag,
        "peak_correlation": peak,
        "correlations_by_lag": values,
    }


def _sync_diagnostic(rgb_signal, ir_signal):
    if len(rgb_signal) < 9 or len(rgb_signal) != len(ir_signal):
        return {
            "available": False,
            "reason": "at least 10 consecutive successful frames are required",
            "pairing_changed": False,
        }
    count = len(rgb_signal)
    boundaries = [0, count // 3, 2 * count // 3, count]
    segments = {}
    for index in range(3):
        start, end = boundaries[index], boundaries[index + 1]
        segments[f"segment_{index + 1}"] = _lag_correlations(
            rgb_signal[start:end], ir_signal[start:end]
        )
    full = _lag_correlations(rgb_signal, ir_signal)
    nonzero = [
        item["best_lag_frames"] for item in [full, *segments.values()]
        if item["best_lag_frames"] not in (None, 0)
    ]
    return {
        "available": True,
        "lag_definition": "compare RGB motion[t] with IR motion[t+lag]",
        "full_sequence": full,
        "segments": segments,
        "nonzero_lag_warning": bool(nonzero),
        "warning": (
            "non-zero integer lag detected; filenames were preserved and no frames were shifted"
            if nonzero else None
        ),
        "pairing_changed": False,
    }


def process_all_pairs(
    rgb_dir=APPLY_RGB_PATH,
    ir_dir=APPLY_IR_PATH,
    output_dir=APPLY_OUTPUT_PATH,
    selected_keys=None,
    model_path=CALIBRATION_MODEL_PATH,
):
    """处理所有同名 RGB/IR 图像对，并返回可序列化的运行报告。"""
    rgb_output = (
        APPLY_RGB_OUTPUT_PATH
        if output_dir == APPLY_OUTPUT_PATH
        else os.path.join(output_dir, "rgb")
    )
    ir_output = (
        APPLY_IR_OUTPUT_PATH
        if output_dir == APPLY_OUTPUT_PATH
        else os.path.join(output_dir, "ir")
    )
    preview_output = (
        APPLY_PREVIEW_OUTPUT_PATH
        if output_dir == APPLY_OUTPUT_PATH
        else os.path.join(output_dir, "preview")
    )
    report_path = (
        APPLY_REPORT_PATH
        if output_dir == APPLY_OUTPUT_PATH
        else os.path.join(output_dir, "application_report.yaml")
    )
    diagnostic_output = os.path.join(output_dir, "diagnostics")
    for folder in (rgb_output, ir_output, preview_output, diagnostic_output):
        os.makedirs(folder, exist_ok=True)

    H, calibration_data = load_calibration(model_path)
    if calibration_data.get("source") == "identity_fallback":
        raise FileNotFoundError(
            f"未找到标定模型：{model_path}\n"
            "请先运行 python calibrate.py 生成模型。"
        )
    _validate_homography(H)
    rgb_reference_size, ir_reference_size = calibration_reference_sizes(calibration_data)

    rgb_images = _discover_images(rgb_dir)
    ir_images = _discover_images(ir_dir)
    paired_keys = sorted(rgb_images.keys() & ir_images.keys())
    rgb_only = sorted(rgb_images.keys() - ir_images.keys())
    ir_only = sorted(ir_images.keys() - rgb_images.keys())
    if selected_keys is not None:
        requested = {str(key).casefold() for key in selected_keys}
        missing = sorted(requested - set(paired_keys))
        if missing:
            raise KeyError(f"找不到代表帧：{missing}")
        paired_keys = [key for key in paired_keys if key in requested]
    representative_keys = (
        paired_keys
        if len(paired_keys) <= 3
        else [paired_keys[0], paired_keys[len(paired_keys) // 2], paired_keys[-1]]
    )

    report = {
        "date": datetime.now().isoformat(),
        "model_path": os.path.abspath(model_path),
        "model_quality_status": calibration_data.get("calibration", {}).get(
            "quality_status"
        ),
        "model_geometry_passed": calibration_data.get("calibration", {}).get(
            "geometry_passed"
        ),
        "model_sample_count_passed": calibration_data.get("calibration", {}).get(
            "sample_count_passed"
        ),
        "model_diversity_warning": calibration_data.get("calibration", {}).get(
            "diversity_warning"
        ),
        "transform_direction": "IR_to_RGB",
        "Homography_calibration_IR_to_RGB": H.tolist(),
        "calibration_reference_sizes": {
            "rgb": list(rgb_reference_size),
            "ir": list(ir_reference_size),
        },
        "calibration_selected_pair": calibration_data.get("metadata", {}).get(
            "selected_pair"
        ),
        "input": {
            "rgb_dir": os.path.abspath(rgb_dir),
            "ir_dir": os.path.abspath(ir_dir),
            "rgb_only": rgb_only,
            "ir_only": ir_only,
        },
        "output_dir": os.path.abspath(output_dir),
        "representative_frames": representative_keys,
        "pairs": [],
    }

    previous_rgb_motion = None
    previous_ir_motion = None
    rgb_motion_signal = []
    ir_motion_signal = []

    for key in paired_keys:
        rgb_path = rgb_images[key]
        ir_path = ir_images[key]
        item = {
            "name": key,
            "rgb_input": os.path.abspath(rgb_path),
            "ir_input": os.path.abspath(ir_path),
        }
        try:
            rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
            if rgb is None:
                raise ValueError(f"无法读取 RGB 图像：{rgb_path}")
            if ir is None:
                raise ValueError(f"无法读取 IR 图像：{ir_path}")

            H_runtime, scale_details = runtime_homography(
                H, calibration_data, rgb, ir
            )
            _validate_homography(H_runtime)
            rgb_aligned, ir_aligned, valid_mask = warp_ir_to_rgb(rgb, ir, H_runtime)
            valid_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)
            output_name = key + APPLY_OUTPUT_EXTENSION
            rgb_result = os.path.join(rgb_output, output_name)
            ir_result = os.path.join(ir_output, output_name)
            preview_result = os.path.join(preview_output, output_name)

            _write_image(rgb_result, rgb_aligned)
            _write_image(ir_result, ir_aligned)

            rgb_preview = _to_preview_8bit(rgb_aligned)
            ir_preview = _to_preview_8bit(ir_aligned)
            fusion = cv2.addWeighted(
                rgb_preview,
                APPLY_PREVIEW_RGB_WEIGHT,
                ir_preview,
                1.0 - APPLY_PREVIEW_RGB_WEIGHT,
                0,
            )
            fusion[valid_mask == 0] = rgb_preview[valid_mask == 0]
            _write_image(preview_result, fusion)

            diagnostic_paths = None
            if key in representative_keys:
                diagnostic_paths = _write_pair_diagnostics(
                    diagnostic_output, key, rgb_preview, ir_preview, valid_mask, fusion
                )

            current_rgb_motion = _motion_frame(rgb)
            current_ir_motion = _motion_frame(ir)
            if previous_rgb_motion is not None:
                rgb_motion_signal.append(float(np.mean(cv2.absdiff(
                    current_rgb_motion, previous_rgb_motion
                ))))
                ir_motion_signal.append(float(np.mean(cv2.absdiff(
                    current_ir_motion, previous_ir_motion
                ))))
            previous_rgb_motion = current_rgb_motion
            previous_ir_motion = current_ir_motion

            item.update(
                {
                    "status": "success",
                    "rgb_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                    "ir_size": [int(ir.shape[1]), int(ir.shape[0])],
                    "output_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                    "valid_overlap_ratio": valid_ratio,
                    "low_overlap_warning": valid_ratio < MIN_VALID_WARP_RATIO,
                    "Homography_runtime_IR_to_RGB": H_runtime.tolist(),
                    "scale_adaptation": scale_details,
                    "rgb_output": os.path.abspath(rgb_result),
                    "ir_output": os.path.abspath(ir_result),
                    "preview_output": os.path.abspath(preview_result),
                    "diagnostics": diagnostic_paths,
                }
            )
        except Exception as error:
            item.update({"status": "failed", "error": str(error)})
        report["pairs"].append(item)

    success_count = sum(item["status"] == "success" for item in report["pairs"])
    successful = [item for item in report["pairs"] if item["status"] == "success"]
    coverage = [item["valid_overlap_ratio"] for item in successful]
    runtime_matrices = {
        yaml.safe_dump(item["Homography_runtime_IR_to_RGB"])
        for item in successful
    }
    if len(runtime_matrices) == 1:
        report["Homography_runtime_IR_to_RGB"] = successful[0][
            "Homography_runtime_IR_to_RGB"
        ]
        # 兼容旧报告消费者：顶层 Homography 指运行时矩阵。
        report["Homography_IR_to_RGB"] = report["Homography_runtime_IR_to_RGB"]
    report["synchronization_diagnostic"] = _sync_diagnostic(
        rgb_motion_signal, ir_motion_signal
    )
    report["summary"] = {
        "rgb_images": len(rgb_images),
        "ir_images": len(ir_images),
        "matched_pairs": len(paired_keys),
        "successful_pairs": success_count,
        "failed_pairs": len(paired_keys) - success_count,
        "unmatched_rgb": len(rgb_only),
        "unmatched_ir": len(ir_only),
        "minimum_valid_overlap_ratio": min(coverage) if coverage else None,
        "mean_valid_overlap_ratio": float(np.mean(coverage)) if coverage else None,
        "maximum_valid_overlap_ratio": max(coverage) if coverage else None,
        "low_overlap_warning_count": sum(
            bool(item["low_overlap_warning"]) for item in successful
        ),
        "output_sizes": sorted({tuple(item["output_size"]) for item in successful}),
    }
    report["summary"]["output_sizes"] = [
        list(size) for size in report["summary"]["output_sizes"]
    ]
    report["physical_limits"] = [
        "sub-frame exposure timing differences",
        "parallax from fish leaving the calibration plane",
        "camera-baseline disparity cannot be removed by one Homography",
    ]
    _save_report(report_path, report)
    return report


def main():
    print("=" * 68)
    print("Apply calibrated IR -> RGB registration")
    print("=" * 68)
    report = process_all_pairs()
    summary = report["summary"]
    print(f"Matched pairs: {summary['matched_pairs']}")
    print(f"Successful: {summary['successful_pairs']}")
    print(f"Failed: {summary['failed_pairs']}")
    print(f"Unmatched RGB/IR: {summary['unmatched_rgb']}/{summary['unmatched_ir']}")
    print(f"Saved to: {APPLY_OUTPUT_PATH}")
    if summary["matched_pairs"] == 0:
        raise RuntimeError(
            f"没有找到同名 RGB/IR 图像对。请将图像分别放入：\n"
            f"RGB: {APPLY_RGB_PATH}\nIR: {APPLY_IR_PATH}"
        )
    if summary["failed_pairs"]:
        raise RuntimeError(
            f"有 {summary['failed_pairs']} 对图像处理失败，详情见 {APPLY_REPORT_PATH}"
        )


if __name__ == "__main__":
    main()
