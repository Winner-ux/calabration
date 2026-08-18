"""使用全部合格棋盘图像对建立 IR -> RGB 全局 Homography。"""

from datetime import datetime
import glob
import os

import cv2
import numpy as np
import yaml

from config import *


def _as_points(corners):
    """统一 OpenCV 返回的 (N,1,2)/(N,2) 角点形状。"""
    if corners is None:
        return None
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return points if len(points) else None


def _gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _preprocess_variants(gray, is_ir):
    clip_values = (
        [CLAHE_CLIP_STRONG, CLAHE_CLIP_LIMIT, CLAHE_CLIP_LIGHT]
        if is_ir else [CLAHE_CLIP_LIMIT, CLAHE_CLIP_LIGHT]
    )
    variants = [gray]
    for clip in clip_values:
        variants.append(
            cv2.createCLAHE(
                clipLimit=float(clip), tileGridSize=CLAHE_TILE_SIZE
            ).apply(gray)
        )
    return variants


def detect_corners_robust(image, pattern, is_ir=False):
    gray = _gray(image)
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    if hasattr(cv2, "findChessboardCornersSB"):
        for prepared in _preprocess_variants(gray, is_ir):
            try:
                ok, corners = cv2.findChessboardCornersSB(
                    prepared, pattern, flags=sb_flags
                )
            except cv2.error:
                ok, corners = False, None
            if ok:
                return _as_points(corners)

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for prepared in _preprocess_variants(gray, is_ir):
        ok, corners = cv2.findChessboardCorners(
            prepared, pattern, flags=classic_flags
        )
        if not ok:
            continue
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(
            prepared, corners, SUBPIX_WINDOW, SUBPIX_ZONE, SUBPIX_CRITERIA
        )
        return _as_points(refined)
    return None


def _find_images(folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, "*" + ext)))
        files.extend(glob.glob(os.path.join(folder, "*" + ext.upper())))
    return sorted(set(files))


def load_image_pairs(rgb_dir=RGB_PATH, ir_dir=IR_PATH):
    rgb_files, ir_files = _find_images(rgb_dir), _find_images(ir_dir)
    rgb_by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in rgb_files}
    ir_by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in ir_files}
    common = sorted(set(rgb_by_name) & set(ir_by_name))
    if common:
        return [(rgb_by_name[name], ir_by_name[name]) for name in common]
    return list(zip(rgb_files, ir_files))


def rotate_calibration_ir(image, rotation=CALIBRATION_IR_ROTATION):
    rotation = str(rotation).strip().lower()
    operations = {
        "none": None,
        "clockwise_90": cv2.ROTATE_90_CLOCKWISE,
        "counterclockwise_90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "rotate_180": cv2.ROTATE_180,
    }
    if rotation not in operations:
        raise ValueError(f"不支持 CALIBRATION_IR_ROTATION={rotation!r}")
    operation = operations[rotation]
    return image.copy() if operation is None else cv2.rotate(image, operation)


def _corner_orders(points, pattern):
    cols, rows = pattern
    grid = points.reshape(rows, cols, 2)
    candidates = [
        ("normal", grid),
        ("reverse_180", grid[::-1, ::-1]),
        ("flip_rows", grid[::-1, :]),
        ("flip_cols", grid[:, ::-1]),
    ]
    unique = []
    for name, candidate in candidates:
        flat = np.ascontiguousarray(candidate.reshape(-1, 2))
        if not any(np.allclose(flat, old) for _, old in unique):
            unique.append((name, flat))
    return unique


def _project_errors(H, source, target):
    projected = cv2.perspectiveTransform(
        np.asarray(source, np.float32).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    return np.linalg.norm(projected - np.asarray(target), axis=1)


def _error_metrics(errors):
    errors = np.asarray(errors, np.float64)
    return {
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
    }


def _quality_reasons(metrics):
    reasons = []
    if metrics["rmse_px"] > PAIR_MAX_RMSE_PX:
        reasons.append(
            f"RMSE {metrics['rmse_px']:.3f} > {PAIR_MAX_RMSE_PX:.3f} px"
        )
    if metrics["p95_error_px"] > PAIR_MAX_P95_PX:
        reasons.append(
            f"P95 {metrics['p95_error_px']:.3f} > {PAIR_MAX_P95_PX:.3f} px"
        )
    if metrics["max_error_px"] > PAIR_MAX_ERROR_PX:
        reasons.append(
            f"max {metrics['max_error_px']:.3f} > {PAIR_MAX_ERROR_PX:.3f} px"
        )
    return reasons


def compute_pair_homography(rgb_points, ir_points, pattern):
    """用完整角点估计单对模型；该模型只用于诊断和角点顺序判定。"""
    expected = int(pattern[0] * pattern[1])
    if len(rgb_points) != expected or len(ir_points) != expected:
        return None
    best = None
    for order, reordered_ir in _corner_orders(ir_points, pattern):
        H, _ = cv2.findHomography(reordered_ir, rgb_points, method=0)
        if H is None:
            continue
        H = H / H[2, 2]
        errors = _project_errors(H, reordered_ir, rgb_points)
        metrics = _error_metrics(errors)
        center = np.mean(reordered_ir, axis=0)
        probe = np.float32(
            [center, center + [10.0, 0.0], center + [0.0, 10.0]]
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(probe, H).reshape(-1, 2)
        axis_x, axis_y = mapped[1] - mapped[0], mapped[2] - mapped[0]
        determinant = float(axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0])
        rotation = float(np.degrees(np.arctan2(axis_x[1], axis_x[0])))
        score = (
            int(determinant <= 0),
            int(abs(rotation) > CHECKERBOARD_MAX_ROTATION_DEG),
            metrics["rmse_px"],
            metrics["median_error_px"],
        )
        if best is None or score < best["score"]:
            best = {
                "H": H,
                "order": order,
                "ir_ordered": reordered_ir,
                "rotation_deg": rotation,
                "used_corner_count": expected,
                "ratio": 1.0,
                "score": score,
                **metrics,
            }
            # 兼容旧的内部字段名。
            best["rmse"] = metrics["rmse_px"]
            best["median_error"] = metrics["median_error_px"]
            best["p95_error"] = metrics["p95_error_px"]
            best["max_error"] = metrics["max_error_px"]
    return best


def _fit_homography(results, robust):
    source = np.concatenate([item["ir_ordered"] for item in results], axis=0)
    target = np.concatenate([item["rgb"] for item in results], axis=0)
    if robust:
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, _ = cv2.findHomography(
            source, target, method, RANSAC_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        )
    else:
        H, _ = cv2.findHomography(source, target, method=0)
    if H is None:
        raise RuntimeError("全局 Homography 估计失败")
    return H / H[2, 2]


def _filter_with_model(results, H, stage):
    accepted, rejected = [], []
    for item in results:
        metrics = _error_metrics(_project_errors(H, item["ir_ordered"], item["rgb"]))
        item[f"{stage}_metrics"] = metrics
        reasons = _quality_reasons(metrics)
        if reasons:
            item["rejection_reasons"].extend(
                f"{stage}: {reason}" for reason in reasons
            )
            rejected.append(item)
        else:
            accepted.append(item)
    return accepted, rejected


def _build_global_model(preliminary):
    robust_H = _fit_homography(preliminary, robust=True)
    accepted, rejected = _filter_with_model(preliminary, robust_H, "robust_global")
    if not accepted:
        raise RuntimeError("稳健全局模型拒绝了全部标定对")

    # 最终模型始终使用所有合格对的全部角点。若最终模型暴露整对异常，
    # 则拒绝整对后重拟合，绝不删除单个角点。
    while True:
        final_H = _fit_homography(accepted, robust=False)
        kept, newly_rejected = _filter_with_model(accepted, final_H, "final_global")
        if not newly_rejected:
            return final_H, accepted, rejected
        rejected.extend(newly_rejected)
        accepted = kept
        if not accepted:
            raise RuntimeError("最终全局模型没有剩余合格标定对")


def _reference_points(width, height):
    xs = np.linspace(0, width - 1, 3)
    ys = np.linspace(0, height - 1, 3)
    return np.float32([(x, y) for y in ys for x in xs]).reshape(-1, 1, 2)


def select_medoid(results, image_size, reference_H=None):
    """返回最接近全局投影行为的代表图像，仅用于诊断可视化。"""
    width, height = image_size
    refs = _reference_points(width, height)
    projections = [cv2.perspectiveTransform(refs, item["H"]) for item in results]
    if reference_H is not None:
        reference = cv2.perspectiveTransform(refs, reference_H)
        distances = [
            float(np.median(np.linalg.norm(p.reshape(-1, 2) - reference.reshape(-1, 2), axis=1)))
            for p in projections
        ]
        selected = int(np.argmin(distances))
        return selected, float(np.median(distances)), float(np.max(distances))

    pairwise = np.zeros((len(results), len(results)), np.float64)
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            distance = float(np.median(np.linalg.norm(
                projections[i].reshape(-1, 2) - projections[j].reshape(-1, 2), axis=1
            )))
            pairwise[i, j] = pairwise[j, i] = distance
    selected = int(np.argmin(np.median(pairwise, axis=1)))
    spread = pairwise[selected]
    return selected, float(np.median(spread)), float(np.max(spread))


def _global_metrics(H, results):
    errors = np.concatenate([
        _project_errors(H, item["ir_ordered"], item["rgb"]) for item in results
    ])
    return _error_metrics(errors)


def _leave_one_out(results):
    entries = []
    if len(results) < 2:
        return {"pairs": entries, "worst_rmse_px": None, "passed": False}
    for held_out in results:
        training = [item for item in results if item is not held_out]
        H = _fit_homography(training, robust=False)
        metrics = _error_metrics(
            _project_errors(H, held_out["ir_ordered"], held_out["rgb"])
        )
        entries.append({"name": held_out["name"], **metrics})
    worst = max(item["rmse_px"] for item in entries)
    return {
        "pairs": entries,
        "worst_rmse_px": float(worst),
        "threshold_px": float(LOOCV_MAX_RMSE_PX),
        "passed": bool(worst <= LOOCV_MAX_RMSE_PX),
    }


def _projection_consistency(results, global_H, ir_size):
    refs = _reference_points(*ir_size)
    global_projection = cv2.perspectiveTransform(refs, global_H).reshape(-1, 2)
    distances = []
    per_pair = []
    for item in results:
        projection = cv2.perspectiveTransform(refs, item["H"]).reshape(-1, 2)
        errors = np.linalg.norm(projection - global_projection, axis=1)
        metrics = _error_metrics(errors)
        per_pair.append({"name": item["name"], **metrics})
        distances.extend(errors.tolist())
    summary = _error_metrics(np.asarray(distances, np.float64))
    return {
        "reference_grid": "3x3 full IR reference frame; extrapolation diagnostic only",
        "summary": summary,
        "pairs": per_pair,
    }


def visualize_corners(image, corners, pattern, save_path, title):
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
    cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2), True)
    cv2.putText(vis, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)


def _validate_reference_size(actual, expected, label):
    if tuple(actual) != tuple(expected):
        raise ValueError(
            f"{label} 尺寸为 {tuple(actual)}，配置参考尺寸为 {tuple(expected)}"
        )


def _diversity_metrics(accepted, reference_size):
    width, height = reference_size
    centers = np.asarray([item["rgb"].mean(axis=0) for item in accepted])
    areas = [
        cv2.contourArea(cv2.convexHull(item["rgb"].astype(np.float32)))
        / float(width * height)
        for item in accepted
    ]
    center_span = centers.max(axis=0) - centers.min(axis=0)
    center_span_ratio = [
        float(center_span[0] / width), float(center_span[1] / height)
    ]
    area_ratio = float(max(areas) / min(areas)) if min(areas) > 0 else float("inf")
    warning = bool(
        max(center_span_ratio) < CALIBRATION_DIVERSITY_MIN_CENTER_SPAN_RATIO
        and area_ratio < CALIBRATION_DIVERSITY_MIN_AREA_RATIO
    )
    return {
        "warning": warning,
        "center_span_px": [float(center_span[0]), float(center_span[1])],
        "center_span_ratio": center_span_ratio,
        "checkerboard_area_ratio": area_ratio,
        "minimum_center_span_ratio": CALIBRATION_DIVERSITY_MIN_CENTER_SPAN_RATIO,
        "minimum_area_ratio": CALIBRATION_DIVERSITY_MIN_AREA_RATIO,
        "enforcement": "warning_only",
    }


def calibrate_group(rgb_dir, ir_dir, model_output_dir, settings=None):
    settings = dict(settings or {})
    pattern = tuple(settings.get("checkerboard", CHECKERBOARD))
    rgb_reference_size = tuple(settings.get(
        "rgb_reference_size", CALIBRATION_RGB_REFERENCE_SIZE
    ))
    ir_reference_size = tuple(settings.get(
        "ir_reference_size", CALIBRATION_IR_REFERENCE_SIZE
    ))
    ir_rotation = settings.get("ir_rotation", CALIBRATION_IR_ROTATION)
    formal_minimum = int(settings.get("minimum_valid_pairs", MIN_VALID_PAIRS))
    provisional_minimum = int(settings.get(
        "minimum_provisional_valid_pairs", MIN_PROVISIONAL_VALID_PAIRS
    ))
    if pattern is None:
        raise ValueError("高精度模式需要明确设置 CHECKERBOARD=(cols, rows)")
    expected_corners = int(pattern[0] * pattern[1])
    pairs = load_image_pairs(rgb_dir, ir_dir)
    if not pairs:
        raise RuntimeError("未找到标定图像")

    print("=" * 72)
    print(f"RGB-IR global calibration | checkerboard={pattern} | pairs={len(pairs)}")
    print("=" * 72)
    corners_dir = os.path.join(model_output_dir, "corners")
    os.makedirs(corners_dir, exist_ok=True)
    detected, records = [], []

    for index, (rgb_path, ir_path) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        record = {"name": name, "accepted": False, "rejection_reasons": []}
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        ir_raw = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or ir_raw is None:
            record["rejection_reasons"].append("image read failed")
            records.append(record)
            continue
        ir = rotate_calibration_ir(ir_raw, ir_rotation)
        try:
            _validate_reference_size(
                (rgb.shape[1], rgb.shape[0]), rgb_reference_size, "RGB"
            )
            _validate_reference_size(
                (ir.shape[1], ir.shape[0]), ir_reference_size, "rotated IR"
            )
        except ValueError as error:
            record["rejection_reasons"].append(str(error))
            records.append(record)
            continue

        rgb_points = detect_corners_robust(rgb, pattern, False)
        ir_points = detect_corners_robust(ir, pattern, True)
        record["rgb_detected_corner_count"] = 0 if rgb_points is None else len(rgb_points)
        record["ir_detected_corner_count"] = 0 if ir_points is None else len(ir_points)
        if (
            rgb_points is None or ir_points is None
            or len(rgb_points) != expected_corners or len(ir_points) != expected_corners
        ):
            record["rejection_reasons"].append(
                f"incomplete corners: RGB={record['rgb_detected_corner_count']}, "
                f"IR={record['ir_detected_corner_count']}, expected={expected_corners}"
            )
            records.append(record)
            print(f"[{index + 1:02d}] {name}: REJECT incomplete corners")
            continue

        result = compute_pair_homography(rgb_points, ir_points, pattern)
        if result is None:
            record["rejection_reasons"].append("pair Homography estimation failed")
            records.append(record)
            continue
        result.update({
            "name": name, "rgb": rgb_points, "ir": ir_points,
            "rejection_reasons": record["rejection_reasons"],
            "rgb_detected_corner_count": len(rgb_points),
            "ir_detected_corner_count": len(ir_points),
        })
        local_metrics = {
            key: result[key] for key in
            ("rmse_px", "median_error_px", "p95_error_px", "max_error_px")
        }
        result["pair_fit_metrics"] = local_metrics
        result["rejection_reasons"].extend(
            f"pair_fit: {reason}" for reason in _quality_reasons(local_metrics)
        )
        records.append(result)
        visualize_corners(
            rgb, rgb_points, pattern,
            os.path.join(corners_dir, f"rgb_{name}.jpg"), f"RGB {name}",
        )
        visualize_corners(
            ir, result["ir_ordered"], pattern,
            os.path.join(corners_dir, f"ir_{name}.jpg"), f"IR rotated {name}",
        )
        if not result["rejection_reasons"]:
            detected.append(result)
        state = "CANDIDATE" if not result["rejection_reasons"] else "REJECT"
        print(
            f"[{index + 1:02d}] {name}: {state} order={result['order']} "
            f"corners={expected_corners}/{expected_corners} "
            f"RMSE={result['rmse_px']:.3f}px P95={result['p95_error_px']:.3f}px "
            f"max={result['max_error_px']:.3f}px"
        )

    if not detected:
        raise RuntimeError("没有通过单对质量门槛的标定对")
    global_H, accepted, model_rejected = _build_global_model(detected)
    rejected_ids = {id(item) for item in model_rejected}
    for item in accepted:
        item["accepted"] = True
    for item in records:
        if id(item) in rejected_ids:
            item["accepted"] = False

    global_metrics = _global_metrics(global_H, accepted)
    loocv = _leave_one_out(accepted)
    selected_index, median_spread, max_spread = select_medoid(
        accepted, ir_reference_size, reference_H=global_H
    )
    chosen = accepted[selected_index]
    projection = _projection_consistency(
        accepted, global_H, ir_reference_size
    )
    geometry_passed = bool(
        global_metrics["rmse_px"] <= GLOBAL_MAX_RMSE_PX
        and global_metrics["p95_error_px"] <= GLOBAL_MAX_P95_PX
        and global_metrics["max_error_px"] <= GLOBAL_MAX_ERROR_PX
        and loocv["passed"]
    )
    sample_count_passed = len(accepted) >= formal_minimum
    eligible_for_application = len(accepted) >= provisional_minimum
    diversity = _diversity_metrics(accepted, rgb_reference_size)
    if not eligible_for_application:
        quality_status = "failed"
    elif diversity["warning"] or not geometry_passed:
        quality_status = "provisional_warning"
    elif sample_count_passed:
        quality_status = "formal"
    else:
        quality_status = "provisional"

    serialized_pairs = []
    for item in records:
        entry = {
            "name": item["name"],
            "accepted": bool(item.get("accepted", False)),
            "rejection_reasons": item.get("rejection_reasons", []),
            "rgb_detected_corner_count": int(item.get("rgb_detected_corner_count", 0)),
            "ir_detected_corner_count": int(item.get("ir_detected_corner_count", 0)),
            "expected_corner_count": expected_corners,
        }
        if "H" in item:
            entry.update({
                "order": item["order"],
                "used_corner_count": int(item["used_corner_count"]),
                "rmse_px": float(item["rmse_px"]),
                "median_error_px": float(item["median_error_px"]),
                "p95_error_px": float(item["p95_error_px"]),
                "max_error_px": float(item["max_error_px"]),
                "inlier_ratio": 1.0,
                "Homography": item["H"].tolist(),
                "pair_fit_metrics": item.get("pair_fit_metrics"),
                "robust_global_metrics": item.get("robust_global_metrics"),
                "final_global_metrics": item.get("final_global_metrics"),
            })
        serialized_pairs.append(entry)

    document = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "checkerboard": list(pattern),
            "square_size_mm": float(SQUARE_SIZE),
            "num_total_pairs": len(pairs),
            "num_valid_pairs": len(accepted),
            "num_rejected_pairs": len(pairs) - len(accepted),
            "selected_pair": chosen["name"],
            "accepted_pairs": [item["name"] for item in accepted],
            "rejected_pairs": [
                item["name"] for item in records if not item.get("accepted", False)
            ],
            "rgb_reference_size": list(rgb_reference_size),
            "ir_reference_size": list(ir_reference_size),
            "ir_rotation": ir_rotation,
            "model_point_count": len(accepted) * expected_corners,
            "coordinate_convention": {
                "size_order": "width_height",
                "pixel_origin": "top_left",
                "axes": "x_right_y_down",
                "transform_direction": "IR_to_RGB",
                "homography_space": "calibration_reference_pixels",
            },
        },
        "calibration": {
            # 兼容旧消费者：字段名和矩阵方向保持不变，但内容改为全局模型。
            "Homography": global_H.tolist(),
            "Affine": global_H[:2].tolist(),
            "pair_rmse_px": float(chosen["rmse_px"]),
            "pair_median_error_px": float(chosen["median_error_px"]),
            "inlier_ratio": 1.0,
            "consistency_median_spread_px": median_spread,
            "consistency_max_spread_px": max_spread,
            "consistent": geometry_passed,
            "quality": "Good" if geometry_passed else "Needs review",
            "geometry_passed": geometry_passed,
            "sample_count_passed": sample_count_passed,
            "diversity_warning": diversity["warning"],
            "eligible_for_application": eligible_for_application,
            "quality_status": quality_status,
            "diversity": diversity,
            "model_type": "global_homography_all_accepted_corners_least_squares",
            "model_point_count": len(accepted) * expected_corners,
            "global_rmse_px": global_metrics["rmse_px"],
            "global_median_error_px": global_metrics["median_error_px"],
            "global_p95_error_px": global_metrics["p95_error_px"],
            "global_max_error_px": global_metrics["max_error_px"],
            "quality_thresholds": {
                "pair_rmse_px": PAIR_MAX_RMSE_PX,
                "pair_p95_px": PAIR_MAX_P95_PX,
                "pair_max_px": PAIR_MAX_ERROR_PX,
                "global_rmse_px": GLOBAL_MAX_RMSE_PX,
                "global_p95_px": GLOBAL_MAX_P95_PX,
                "global_max_px": GLOBAL_MAX_ERROR_PX,
                "loocv_worst_rmse_px": LOOCV_MAX_RMSE_PX,
                "minimum_valid_pairs": formal_minimum,
                "minimum_provisional_valid_pairs": provisional_minimum,
                "base_short_edge_px": CALIBRATION_THRESHOLD_BASE_SHORT_EDGE,
                "scale": CALIBRATION_THRESHOLD_SCALE,
            },
            "leave_one_out": loocv,
            "projection_consistency": projection,
        },
        "all_pairs": serialized_pairs,
        "physical_limits": [
            "sub-frame exposure timing differences",
            "parallax from fish leaving the calibration plane",
            "camera-baseline disparity cannot be removed by one Homography",
        ],
    }
    os.makedirs(model_output_dir, exist_ok=True)
    yaml_path = os.path.join(model_output_dir, "calibration.yaml")
    with open(yaml_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)

    print("-" * 72)
    print(f"Accepted pairs: {len(accepted)}/{len(pairs)}")
    print(f"Rejected: {document['metadata']['rejected_pairs']}")
    print(f"Model points: {document['metadata']['model_point_count']}")
    print(
        f"Global RMSE/P95/max: {global_metrics['rmse_px']:.3f}/"
        f"{global_metrics['p95_error_px']:.3f}/{global_metrics['max_error_px']:.3f} px"
    )
    print(f"LOOCV worst RMSE: {loocv['worst_rmse_px']:.3f} px")
    print(f"Representative pair: {chosen['name']} (diagnostic only)")
    print(f"Quality: {document['calibration']['quality']}")
    print(f"Saved: {yaml_path}")
    report_path = os.path.join(model_output_dir, "calibration_report.yaml")
    with open(report_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump({
            "metadata": document["metadata"],
            "quality": {
                key: document["calibration"][key] for key in (
                    "geometry_passed", "sample_count_passed",
                    "diversity_warning", "eligible_for_application",
                    "quality_status", "global_rmse_px", "global_p95_error_px",
                    "global_max_error_px", "leave_one_out", "diversity",
                )
            },
        }, stream, allow_unicode=True, sort_keys=False)
    return document


def main():
    return calibrate_group(RGB_PATH, IR_PATH, SAVE_PATH)


if __name__ == "__main__":
    main()
