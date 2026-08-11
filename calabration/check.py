"""生成配准可视化，并用边缘 Chamfer 距离给出几何质量指标。"""

import os

import cv2
import numpy as np
import yaml

from config import APPLY_OUTPUT_PATH, APPLY_REPORT_PATH, CALIBRATION_MODEL_PATH, SAVE_PATH


def _edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    low = max(20, int(np.median(gray) * 0.33))
    high = max(low + 20, int(np.median(gray) * 0.90))
    return cv2.Canny(gray, low, high)


def checkerboard_overlay(rgb, ir, valid, grid_size=8):
    height, width = rgb.shape[:2]
    result = rgb.copy()
    for row in range(grid_size):
        for col in range(grid_size):
            if (row + col) % 2 == 0:
                continue
            y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            local_valid = valid[y0:y1, x0:x1] > 0
            result[y0:y1, x0:x1][local_valid] = ir[y0:y1, x0:x1][local_valid]
    return result


def edge_overlay(rgb, ir):
    rgb_edges, ir_edges = _edges(rgb), _edges(ir)
    result = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    result[rgb_edges > 0] = (0, 255, 0)
    result[ir_edges > 0] = (0, 0, 255)
    overlap = (rgb_edges > 0) & (ir_edges > 0)
    result[overlap] = (0, 255, 255)
    return result, rgb_edges, ir_edges


def chamfer_metrics(rgb_edges, ir_edges, valid):
    region = valid > 0
    rgb_mask = (rgb_edges > 0) & region
    ir_mask = (ir_edges > 0) & region
    if not rgb_mask.any() or not ir_mask.any():
        return {"median_px": None, "p90_px": None, "within_1px_ratio": 0.0,
                "rating": "Unavailable"}
    to_rgb = cv2.distanceTransform((~rgb_mask).astype(np.uint8), cv2.DIST_L2, 3)
    to_ir = cv2.distanceTransform((~ir_mask).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate([to_rgb[ir_mask], to_ir[rgb_mask]])
    median = float(np.median(distances))
    p75 = float(np.percentile(distances, 75))
    p90 = float(np.percentile(distances, 90))
    within = float(np.mean(distances <= 1.5))
    # P90 对非同步拍摄中的移动窗口/行人等内容变化极敏感；几何等级使用
    # 中位数与 P75，仍保留 P90 供诊断遮挡和场景变化。
    if median <= 1.0 and p75 <= 4.0:
        rating = "Excellent"
    elif median <= 2.0 and p75 <= 8.0:
        rating = "Good"
    elif median <= 4.0:
        rating = "Fair"
    else:
        rating = "Poor"
    return {"median_px": median, "p75_px": p75, "p90_px": p90,
            "within_1px_ratio": within,
            "rating": rating}


def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _registered_pairs():
    rgb_dir = os.path.join(APPLY_OUTPUT_PATH, "rgb")
    ir_dir = os.path.join(APPLY_OUTPUT_PATH, "ir")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(ir_dir):
        return []
    rgb_files = {
        os.path.splitext(name)[0]: os.path.join(rgb_dir, name)
        for name in os.listdir(rgb_dir) if name.lower().endswith(".png")
    }
    ir_files = {
        os.path.splitext(name)[0]: os.path.join(ir_dir, name)
        for name in os.listdir(ir_dir) if name.lower().endswith(".png")
    }
    names = sorted(set(rgb_files) & set(ir_files))
    if not names:
        return []
    selected = [names[0], names[len(names) // 2], names[-1]]
    return [(name, rgb_files[name], ir_files[name]) for name in selected]


def static_structure_metrics():
    """用首/中/末帧中重复出现的边缘近似提取静态鱼缸结构。"""
    pairs = _registered_pairs()
    if len(pairs) < 3:
        return {
            "available": False,
            "reason": "three representative registered frames are required",
        }
    rgb_edge_stack, ir_edge_stack, rgb_images = [], [], []
    names = []
    for name, rgb_path, ir_path in pairs:
        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path)
        if rgb is None or ir is None or rgb.shape != ir.shape:
            return {"available": False, "reason": f"cannot read representative frame {name}"}
        names.append(name)
        rgb_images.append(rgb)
        rgb_edge_stack.append(_edges(rgb) > 0)
        ir_edge_stack.append(_edges(ir) > 0)

    # 至少在两张代表帧中出现的边缘视为静态结构；移动鱼体通常会被排除。
    rgb_static = np.sum(rgb_edge_stack, axis=0) >= 2
    ir_static = np.sum(ir_edge_stack, axis=0) >= 2
    valid = np.full(rgb_static.shape, 255, np.uint8)
    metrics = chamfer_metrics(
        rgb_static.astype(np.uint8) * 255,
        ir_static.astype(np.uint8) * 255,
        valid,
    )
    to_rgb = cv2.distanceTransform((~rgb_static).astype(np.uint8), cv2.DIST_L2, 3)
    to_ir = cv2.distanceTransform((~ir_static).astype(np.uint8), cv2.DIST_L2, 3)
    all_distances = np.concatenate([to_rgb[ir_static], to_ir[rgb_static]])
    close = all_distances <= 5.0
    matched_metrics = {
        "maximum_match_distance_px": 5.0,
        "edge_sample_ratio": float(np.mean(close)),
        "median_px": float(np.median(all_distances[close])) if close.any() else None,
        "p90_px": float(np.percentile(all_distances[close], 90)) if close.any() else None,
    }
    background = np.median(np.stack(rgb_images).astype(np.float32), axis=0).astype(np.uint8)
    overlay = (background.astype(np.float32) * 0.30).astype(np.uint8)
    overlay[rgb_static] = (0, 255, 0)
    overlay[ir_static] = (0, 0, 255)
    overlay[rgb_static & ir_static] = (0, 255, 255)
    overlay_path = os.path.join(SAVE_PATH, "static_structure_edge_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    return {
        "available": True,
        "method": "edges present in at least 2 of first/middle/last frames",
        "representative_frames": names,
        "moving_fish_excluded_by_temporal_consensus": True,
        "metric_scope": (
            "unfiltered symmetric Chamfer distance; modality-specific edges and "
            "different transparent-container depth surfaces remain included"
        ),
        "matched_static_edges_within_5px": matched_metrics,
        "overlay_path": os.path.abspath(overlay_path),
        **metrics,
    }


def main():
    rgb = cv2.imread(os.path.join(SAVE_PATH, "aligned_rgb.png"))
    ir = cv2.imread(os.path.join(SAVE_PATH, "aligned_ir.png"))
    valid = cv2.imread(os.path.join(SAVE_PATH, "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        raise FileNotFoundError("请先运行 python registration.py")
    if valid is None:
        valid = np.full(rgb.shape[:2], 255, np.uint8)

    blend = cv2.addWeighted(rgb, 0.5, ir, 0.5, 0)
    blend[valid == 0] = rgb[valid == 0]
    overlay, rgb_edges, ir_edges = edge_overlay(rgb, ir)
    metrics = chamfer_metrics(rgb_edges, ir_edges, valid)

    cv2.imwrite(os.path.join(SAVE_PATH, "fusion_check.png"), blend)
    cv2.imwrite(os.path.join(SAVE_PATH, "checkerboard_overlay.png"),
                checkerboard_overlay(rgb, ir, valid))
    cv2.imwrite(os.path.join(SAVE_PATH, "edge_overlay.png"), overlay)
    side = np.hstack([rgb, ir])
    cv2.putText(side, "RGB", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(side, "Aligned IR", (rgb.shape[1] + 20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(SAVE_PATH, "side_by_side.png"), side)
    registration_path = os.path.join(SAVE_PATH, "registration_report.yaml")
    registration = _load_yaml(registration_path)
    calibration = _load_yaml(CALIBRATION_MODEL_PATH)
    application = _load_yaml(APPLY_REPORT_PATH)
    calibration_section = calibration.get("calibration", {})
    refinement = registration.get("pair_refinement", {})
    checkerboard = registration.get("checkerboard_alignment", {})
    if checkerboard.get("accepted"):
        corner_rmse = checkerboard.get("rmse_px", float("inf"))
        if corner_rmse <= 1.0:
            geometric_rating = "Excellent"
        elif corner_rmse <= 2.0:
            geometric_rating = "Good"
        else:
            geometric_rating = "Review required"
    elif (
        refinement.get("accepted")
        and refinement.get("median_error_px", float("inf")) <= 0.5
        and refinement.get("coverage", 0.0) >= 0.5
    ):
        geometric_rating = "Excellent"
    elif refinement.get("accepted"):
        geometric_rating = "Good"
    elif calibration_section.get("quality") == "Good":
        geometric_rating = "Calibration model passed"
    else:
        geometric_rating = "Calibration fallback - review required"

    calibration_corner_error = {
        "model_point_count": calibration_section.get("model_point_count"),
        "accepted_pair_count": calibration.get("metadata", {}).get("num_valid_pairs"),
        "rmse_px": calibration_section.get("global_rmse_px"),
        "p95_px": calibration_section.get("global_p95_error_px"),
        "max_px": calibration_section.get("global_max_error_px"),
        "leave_one_out_worst_rmse_px": calibration_section.get(
            "leave_one_out", {}
        ).get("worst_rmse_px"),
        "quality": calibration_section.get("quality"),
    }
    static_metrics = static_structure_metrics()
    application_summary = application.get("summary", {})
    valid_coverage = {
        "single_image_ratio": registration.get("valid_overlap_ratio"),
        "batch_minimum_ratio": application_summary.get("minimum_valid_overlap_ratio"),
        "batch_mean_ratio": application_summary.get("mean_valid_overlap_ratio"),
        "batch_low_overlap_warning_count": application_summary.get(
            "low_overlap_warning_count"
        ),
    }

    report = {
        "calibration_corner_error": calibration_corner_error,
        "static_fish_tank_structure_edge_error": static_metrics,
        "valid_coverage": valid_coverage,
        "synchronization_diagnostic": application.get("synchronization_diagnostic"),
        "geometric_registration": {
            "rating": geometric_rating,
            "transform_source": registration.get("transform_source"),
            "checkerboard_pattern": checkerboard.get("pattern"),
            "checkerboard_used_corners": checkerboard.get("used_corner_count"),
            "checkerboard_rmse_px": checkerboard.get("rmse_px"),
            "checkerboard_max_error_px": checkerboard.get("max_error_px"),
            "feature_inliers": refinement.get("inliers"),
            "feature_coverage": refinement.get("coverage"),
            "median_reprojection_error_px": refinement.get("median_error_px"),
            "rmse_reprojection_error_px": refinement.get("rmse_px"),
            "global_calibration_rmse_px": calibration_section.get("global_rmse_px"),
            "global_calibration_p95_px": calibration_section.get("global_p95_error_px"),
            "global_calibration_max_px": calibration_section.get("global_max_error_px"),
            "note": (
                "This rating covers the calibration plane. Static edges at other depths "
                "are reported separately."
            ),
        },
        # 保留旧字段，但明确它不能评价几何精度：运动鱼体和跨模态外观都会影响它。
        "scene_edge_consistency": {
            **metrics,
            "diagnostic_only": True,
            "not_used_for_geometric_acceptance": True,
        },
        "physical_limits": calibration.get("physical_limits", [
            "sub-frame exposure timing differences",
            "parallax from fish leaving the calibration plane",
            "camera-baseline disparity cannot be removed by one Homography",
        ]),
    }
    with open(os.path.join(SAVE_PATH, "quality_report.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)

    print("Geometric registration:", report["geometric_registration"])
    print("Calibration corner error:", calibration_corner_error)
    print("Static fish-tank structure edge error:", static_metrics)
    print("Valid coverage:", valid_coverage)
    print("Dynamic scene edge diagnostic:", metrics)
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
