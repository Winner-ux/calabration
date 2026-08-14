"""RGB/IR 图像配准：标定变换 + 当前图像对全局精配准。"""

from datetime import datetime
import os

import cv2
import numpy as np
import yaml

from config import *
from calibrate import detect_corners_robust, _corner_orders


def _gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_3ch(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def load_calibration(model_path=CALIBRATION_MODEL_PATH):
    path = model_path
    if not os.path.exists(path):
        return np.eye(3, dtype=np.float64), {"source": "identity_fallback"}
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    section = data.get("calibration", {})
    if "Homography" in section:
        H = np.asarray(section["Homography"], np.float64)
    elif "Affine" in section:
        affine = np.asarray(section["Affine"], np.float64)
        H = np.vstack([affine, [0.0, 0.0, 1.0]])
    else:
        raise KeyError(f"{path} 中没有 Homography 或 Affine")
    if section.get("consistent") is False:
        print(
            "[WARN] calibration.yaml 标记为 inconsistent："
            f"median spread={section.get('consistency_median_spread_px')} px, "
            f"max spread={section.get('consistency_max_spread_px')} px。\n"
            "       当前矩阵可继续应用，但普通场景可能因标定对不一致或视差而错位。"
        )
    return H / H[2, 2], data


def calibration_reference_sizes(calibration_data):
    """从模型元数据读取 (width, height) 参考尺寸。"""
    metadata = calibration_data.get("metadata", {})
    rgb_size = metadata.get("rgb_reference_size")
    ir_size = metadata.get("ir_reference_size")
    if rgb_size is None or ir_size is None:
        raise KeyError(
            "calibration.yaml 缺少 rgb_reference_size/ir_reference_size；"
            "请先重新运行 python calibrate.py，旧矩阵不能安全地跨分辨率应用"
        )
    rgb_size = tuple(int(value) for value in rgb_size)
    ir_size = tuple(int(value) for value in ir_size)
    if len(rgb_size) != 2 or len(ir_size) != 2 or min(*rgb_size, *ir_size) <= 0:
        raise ValueError("calibration.yaml 中的参考尺寸无效")
    return rgb_size, ir_size


def _validate_aspect_ratio(reference_size, runtime_size, label, tolerance):
    reference_width, reference_height = reference_size
    runtime_width, runtime_height = runtime_size
    if min(reference_width, reference_height, runtime_width, runtime_height) <= 0:
        raise ValueError(f"{label} 尺寸必须为正数")
    reference_ratio = reference_width / reference_height
    runtime_ratio = runtime_width / runtime_height
    relative_error = abs(runtime_ratio / reference_ratio - 1.0)
    if relative_error > tolerance:
        raise ValueError(
            f"{label} 宽高比与标定参考不一致：reference={reference_size}, "
            f"runtime={runtime_size}, relative_error={relative_error:.4%}, "
            f"tolerance={tolerance:.4%}。可能存在裁剪或 FOV 变化。"
        )
    return relative_error


def adapt_homography_to_runtime(
    H_calibration,
    calibration_rgb_size,
    calibration_ir_size,
    runtime_rgb_size,
    runtime_ir_size,
    aspect_tolerance=ASPECT_RATIO_TOLERANCE,
):
    """把标定像素坐标中的 IR->RGB 矩阵换算到运行图尺寸。

    H_runtime = S_rgb(calibration->runtime) @ H_calibration
                @ S_ir(runtime->calibration)
    """
    rgb_aspect_error = _validate_aspect_ratio(
        calibration_rgb_size, runtime_rgb_size, "RGB", aspect_tolerance
    )
    ir_aspect_error = _validate_aspect_ratio(
        calibration_ir_size, runtime_ir_size, "IR", aspect_tolerance
    )
    cal_rgb_w, cal_rgb_h = calibration_rgb_size
    cal_ir_w, cal_ir_h = calibration_ir_size
    run_rgb_w, run_rgb_h = runtime_rgb_size
    run_ir_w, run_ir_h = runtime_ir_size
    S_rgb = np.diag([
        run_rgb_w / cal_rgb_w,
        run_rgb_h / cal_rgb_h,
        1.0,
    ]).astype(np.float64)
    S_ir = np.diag([
        cal_ir_w / run_ir_w,
        cal_ir_h / run_ir_h,
        1.0,
    ]).astype(np.float64)
    H_runtime = S_rgb @ np.asarray(H_calibration, np.float64) @ S_ir
    H_runtime /= H_runtime[2, 2]
    details = {
        "calibration_rgb_size": list(calibration_rgb_size),
        "calibration_ir_size": list(calibration_ir_size),
        "runtime_rgb_size": list(runtime_rgb_size),
        "runtime_ir_size": list(runtime_ir_size),
        "rgb_scale_calibration_to_runtime": [
            run_rgb_w / cal_rgb_w, run_rgb_h / cal_rgb_h,
        ],
        "ir_scale_runtime_to_calibration": [
            cal_ir_w / run_ir_w, cal_ir_h / run_ir_h,
        ],
        "rgb_aspect_ratio_relative_error": rgb_aspect_error,
        "ir_aspect_ratio_relative_error": ir_aspect_error,
        "aspect_ratio_tolerance": float(aspect_tolerance),
    }
    return H_runtime, details


def runtime_homography(H_calibration, calibration_data, rgb, ir):
    rgb_reference, ir_reference = calibration_reference_sizes(calibration_data)
    rgb_runtime = (int(rgb.shape[1]), int(rgb.shape[0]))
    ir_runtime = (int(ir.shape[1]), int(ir.shape[0]))
    return adapt_homography_to_runtime(
        H_calibration,
        rgb_reference,
        ir_reference,
        rgb_runtime,
        ir_runtime,
    )


def _feature_image(image):
    gray = _gray(image)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)


def _coverage(points, width, height):
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, np.float32))
    return float(cv2.contourArea(hull) / max(width * height, 1))


def estimate_pair_homography(ir, rgb):
    """以 IR 为源、RGB 为目标估计全局 Homography，并返回质量指标。"""
    if not hasattr(cv2, "SIFT_create"):
        return None, {"accepted": False, "reason": "OpenCV 未提供 SIFT"}
    detector = cv2.SIFT_create(
        nfeatures=FEATURE_MAX_COUNT, contrastThreshold=0.02, edgeThreshold=12
    )
    ir_keypoints, ir_desc = detector.detectAndCompute(_feature_image(ir), None)
    rgb_keypoints, rgb_desc = detector.detectAndCompute(_feature_image(rgb), None)
    if ir_desc is None or rgb_desc is None:
        return None, {"accepted": False, "reason": "描述子不足"}

    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ir_desc, rgb_desc, k=2)
    matches = [a for pair in raw if len(pair) == 2 for a, b in [pair]
               if a.distance < FEATURE_RATIO_TEST * b.distance]
    if len(matches) < 8:
        return None, {"accepted": False, "reason": f"匹配点过少: {len(matches)}"}

    source = np.float32([ir_keypoints[m.queryIdx].pt for m in matches])
    target = np.float32([rgb_keypoints[m.trainIdx].pt for m in matches])
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(
        source, target, method,
        ransacReprojThreshold=REFINE_RANSAC_THRESHOLD,
        maxIters=10000, confidence=0.999,
    )
    if H is None or mask is None:
        return None, {"accepted": False, "reason": "RANSAC 单应估计失败"}

    inliers = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    inlier_errors = errors[inliers]
    h, w = ir.shape[:2]
    inlier_count = int(inliers.sum())
    ratio = float(inliers.mean())
    coverage = _coverage(source[inliers], w, h)
    median_error = float(np.median(inlier_errors))
    rmse = float(np.sqrt(np.mean(inlier_errors ** 2)))

    corners = np.float32([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area_ratio = abs(float(cv2.contourArea(warped_corners.astype(np.float32)))) / max(w * h, 1)
    finite = bool(np.isfinite(H).all() and np.isfinite(warped_corners).all())
    accepted = (
        finite
        and inlier_count >= REFINE_MIN_INLIERS
        and ratio >= REFINE_MIN_INLIER_RATIO
        and coverage >= REFINE_MIN_COVERAGE
        and median_error <= REFINE_MAX_MEDIAN_ERROR
        and 0.25 <= area_ratio <= 4.0
    )
    metrics = {
        "accepted": bool(accepted),
        "matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": ratio,
        "coverage": coverage,
        "median_error_px": median_error,
        "rmse_px": rmse,
        "warped_area_ratio": area_ratio,
        "reason": "quality gates passed" if accepted else "quality gates failed",
    }
    return H / H[2, 2], metrics


def _homography_rotation_deg(H, point):
    """计算指定点处 Homography 的局部旋转角和方向行列式。"""
    x, y = map(float, point)
    h = H
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    u_num = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v_num = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    jacobian = np.array([
        [
            (h[0, 0] * denominator - u_num * h[2, 0]) / denominator ** 2,
            (h[0, 1] * denominator - u_num * h[2, 1]) / denominator ** 2,
        ],
        [
            (h[1, 0] * denominator - v_num * h[2, 0]) / denominator ** 2,
            (h[1, 1] * denominator - v_num * h[2, 1]) / denominator ** 2,
        ],
    ])
    angle = float(np.degrees(np.arctan2(jacobian[1, 0], jacobian[0, 0])))
    return angle, float(np.linalg.det(jacobian))


def estimate_checkerboard_homography(ir, rgb, pattern):
    """检测并使用全部棋盘角点估计 IR -> RGB Homography。"""
    if pattern is None:
        return None, {"accepted": False, "reason": "CHECKERBOARD 未设置"}, None
    expected = int(pattern[0] * pattern[1])
    rgb_points = detect_corners_robust(rgb, pattern, is_ir=False)
    ir_points = detect_corners_robust(ir, pattern, is_ir=True)
    metrics = {
        "pattern": list(pattern),
        "expected_corner_count": expected,
        "rgb_detected_corner_count": 0 if rgb_points is None else len(rgb_points),
        "ir_detected_corner_count": 0 if ir_points is None else len(ir_points),
    }
    if rgb_points is None or ir_points is None:
        metrics.update({"accepted": False, "reason": "至少一幅图未检测到完整棋盘"})
        return None, metrics, None
    if len(rgb_points) != expected or len(ir_points) != expected:
        metrics.update({"accepted": False, "reason": "检测角点数量与 CHECKERBOARD 不一致"})
        return None, metrics, None

    center = np.mean(ir_points, axis=0)
    candidates = []
    for order, ordered_ir in _corner_orders(ir_points, pattern):
        # method=0：所有角点共同参与最小二乘拟合，不允许 RANSAC 丢弃角点。
        H, _ = cv2.findHomography(ordered_ir, rgb_points, method=0)
        if H is None:
            continue
        H = H / H[2, 2]
        projected = cv2.perspectiveTransform(
            ordered_ir.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - rgb_points, axis=1)
        angle, determinant = _homography_rotation_deg(H, center)
        mirrored = determinant <= 0
        excessive_rotation = abs(angle) > CHECKERBOARD_MAX_ROTATION_DEG
        # 先排除镜像/倒置解，再以全部角点 RMSE 选择。
        score = (
            int(mirrored),
            int(excessive_rotation),
            float(np.sqrt(np.mean(errors ** 2))),
        )
        candidates.append((score, order, H, ordered_ir, projected, errors, angle, determinant))

    if not candidates:
        metrics.update({"accepted": False, "reason": "无法计算棋盘单应矩阵"})
        return None, metrics, None
    _, order, H, ordered_ir, projected, errors, angle, determinant = min(
        candidates, key=lambda item: item[0]
    )
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    metrics.update({
        "accepted": True,
        "reason": "all checkerboard corners used",
        "corner_order": order,
        "used_corner_count": expected,
        "rmse_px": rmse,
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
        "within_1px_count": int(np.count_nonzero(errors <= 1.0)),
        "within_1px_ratio": float(np.mean(errors <= 1.0)),
        "local_rotation_deg": angle,
        "local_jacobian_determinant": determinant,
        "per_corner_error_px": [float(value) for value in errors],
    })
    diagnostic = {
        "rgb_points": rgb_points,
        "projected_ir_points": projected,
        "ordered_ir_points": ordered_ir,
        "errors": errors,
    }
    return H, metrics, diagnostic


def save_corner_diagnostic(rgb, diagnostic, path):
    """绿色=RGB角点，红色=变换后的IR角点，白线=残差。"""
    canvas = ensure_3ch(rgb).copy()
    rgb_points = diagnostic["rgb_points"]
    projected = diagnostic["projected_ir_points"]
    errors = diagnostic["errors"]
    for index, (target, source, error) in enumerate(zip(rgb_points, projected, errors)):
        target_pt = tuple(np.rint(target).astype(int))
        source_pt = tuple(np.rint(source).astype(int))
        cv2.line(canvas, target_pt, source_pt, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, target_pt, 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, source_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index + 1), (target_pt[0] + 5, target_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    cv2.putText(
        canvas, f"corners={len(errors)}  RMSE={rmse:.3f}px  max={errors.max():.3f}px",
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.imwrite(path, canvas)


def warp_ir_to_rgb(rgb, ir, H):
    height, width = rgb.shape[:2]
    aligned = cv2.warpPerspective(
        ensure_3ch(ir), H, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.full(ir.shape[:2], 255, np.uint8)
    valid_mask = cv2.warpPerspective(
        source_mask, H, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    return ensure_3ch(rgb), aligned, valid_mask


def _save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def main():
    print("=" * 68)
    print("RGB-IR registration")
    print("=" * 68)
    rgb = cv2.imread(RGB_TEST, cv2.IMREAD_UNCHANGED)
    ir = cv2.imread(IR_TEST, cv2.IMREAD_UNCHANGED)
    if rgb is None or ir is None:
        raise FileNotFoundError(f"测试图像不存在\nRGB: {RGB_TEST}\nIR: {IR_TEST}")

    calibrated_H, calibration_data = load_calibration()
    selected_H, source = calibrated_H, "calibration"
    scale_adaptation = None
    refine_metrics = {"accepted": False, "reason": "pair refinement disabled"}
    checkerboard_metrics = {"accepted": False, "reason": "checkerboard mode disabled"}
    corner_diagnostic = None

    mode = str(REGISTRATION_MODE).strip().lower()
    if mode == "checkerboard":
        board_H, checkerboard_metrics, corner_diagnostic = \
            estimate_checkerboard_homography(ir, rgb, CHECKERBOARD)
        if board_H is None or not checkerboard_metrics["accepted"]:
            raise RuntimeError(
                "棋盘角点配准失败：" + checkerboard_metrics.get("reason", "unknown")
                + f"\nRGB检测数={checkerboard_metrics.get('rgb_detected_corner_count')}"
                + f"，IR检测数={checkerboard_metrics.get('ir_detected_corner_count')}"
                + f"，期望={checkerboard_metrics.get('expected_corner_count')}"
            )
        selected_H, source = board_H, "checkerboard_corners"
    elif mode == "features" and ENABLE_PAIR_REFINEMENT:
        refined_H, refine_metrics = estimate_pair_homography(ir, rgb)
        if refined_H is not None and refine_metrics["accepted"]:
            selected_H, source = refined_H, "pair_refinement"
        else:
            raise RuntimeError("全局特征配准失败：" + refine_metrics.get("reason", "unknown"))
    elif mode != "calibration":
        raise ValueError(
            f"未知 REGISTRATION_MODE={REGISTRATION_MODE!r}，"
            "可选 checkerboard/features/calibration"
        )
    else:
        selected_H, scale_adaptation = runtime_homography(
            calibrated_H, calibration_data, rgb, ir
        )

    rgb_aligned, ir_aligned, valid_mask = warp_ir_to_rgb(rgb, ir, selected_H)
    valid_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)
    if valid_ratio < MIN_VALID_WARP_RATIO:
        print(f"[WARN] 有效重叠区域仅 {valid_ratio:.1%}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    cv2.imwrite(os.path.join(SAVE_PATH, "aligned_rgb.png"), rgb_aligned)
    cv2.imwrite(os.path.join(SAVE_PATH, "aligned_ir.png"), ir_aligned)
    cv2.imwrite(os.path.join(SAVE_PATH, "valid_mask.png"), valid_mask)
    fusion = cv2.addWeighted(rgb_aligned, 0.5, ir_aligned, 0.5, 0)
    fusion[valid_mask == 0] = rgb_aligned[valid_mask == 0]
    cv2.imwrite(os.path.join(SAVE_PATH, "fusion_preview.png"), fusion)
    if corner_diagnostic is not None:
        save_corner_diagnostic(
            rgb_aligned, corner_diagnostic,
            os.path.join(SAVE_PATH, "checkerboard_corner_alignment.png"),
        )

    report = {
        "date": datetime.now().isoformat(),
        "transform_source": source,
        "transform_direction": "IR_to_RGB",
        "Homography_calibration_IR_to_RGB": calibrated_H.tolist(),
        "Homography_runtime_IR_to_RGB": selected_H.tolist(),
        "Homography_IR_to_RGB": selected_H.tolist(),
        "scale_adaptation": scale_adaptation,
        "valid_overlap_ratio": valid_ratio,
        "checkerboard_alignment": checkerboard_metrics,
        "pair_refinement": refine_metrics,
        "calibration_selected_pair": calibration_data.get("metadata", {}).get("selected_pair"),
    }
    _save_yaml(os.path.join(SAVE_PATH, "registration_report.yaml"), report)

    print(f"Transform source: {source}")
    print(f"H calibration (IR -> RGB):\n{calibrated_H}")
    print(f"H runtime (IR -> RGB):\n{selected_H}")
    print(f"Valid overlap: {valid_ratio:.1%}")
    if source == "checkerboard_corners":
        print("Checkerboard alignment:", {
            key: value for key, value in checkerboard_metrics.items()
            if key != "per_corner_error_px"
        })
    elif ENABLE_PAIR_REFINEMENT:
        print("Pair refinement:", refine_metrics)
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
