"""按 day/experiment 一一对应地标定并批量配准当前 RGB/IR 数据集。"""

from argparse import ArgumentParser
from datetime import datetime
import os
import shutil
import sys

import cv2
import yaml

from apply_registration import process_all_pairs
from calibrate import calibrate_group
from config import (
    CALIBRATION_IR_REFERENCE_SIZE,
    CALIBRATION_IR_ROTATION,
    CALIBRATION_RGB_REFERENCE_SIZE,
    CHECKERBOARD,
    DATASET_BATCH_REPORT_PATH,
    DATASET_CALIBRATED_OUTPUT_PATH,
    DATASET_CALIBRATION_INPUT_PATH,
    DATASET_EXPERIMENT_INPUT_PATH,
    DATASET_MODEL_OUTPUT_PATH,
    DATASET_REGISTERED_OUTPUT_PATH,
    IMAGE_EXTENSIONS,
    MIN_PROVISIONAL_VALID_PAIRS,
    MIN_VALID_PAIRS,
)


def _image_map(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    images = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        key = os.path.splitext(name)[0].casefold()
        if key in images:
            raise ValueError(f"目录中存在重复主文件名 {key!r}: {folder}")
        images[key] = path
    return images


def _validate_pair_dirs(rgb_dir, ir_dir, expected_size=None, read_all=False):
    rgb = _image_map(rgb_dir)
    ir = _image_map(ir_dir)
    rgb_only = sorted(rgb.keys() - ir.keys())
    ir_only = sorted(ir.keys() - rgb.keys())
    if rgb_only or ir_only:
        raise ValueError(f"RGB/IR 文件名不一致: rgb_only={rgb_only}, ir_only={ir_only}")
    if not rgb:
        raise ValueError(f"没有同名 RGB/IR 图像: {rgb_dir} | {ir_dir}")
    keys = sorted(rgb)
    selected = keys if read_all else sorted({keys[0], keys[len(keys) // 2], keys[-1]})
    for key in selected:
        for label, path in (("RGB", rgb[key]), ("IR", ir[key])):
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"无法读取 {label} 图像: {path}")
            size = (int(image.shape[1]), int(image.shape[0]))
            if expected_size is not None and size != tuple(expected_size):
                raise ValueError(
                    f"{label} 尺寸 {size} 与预期 {tuple(expected_size)} 不一致: {path}"
                )
    return {"count": len(keys), "names": keys}


def discover_dataset():
    calibration_groups = {}
    experiment_groups = {}
    if not os.path.isdir(DATASET_CALIBRATION_INPUT_PATH):
        raise FileNotFoundError(DATASET_CALIBRATION_INPUT_PATH)
    if not os.path.isdir(DATASET_EXPERIMENT_INPUT_PATH):
        raise FileNotFoundError(DATASET_EXPERIMENT_INPUT_PATH)

    for day in sorted(os.listdir(DATASET_CALIBRATION_INPUT_PATH)):
        day_dir = os.path.join(DATASET_CALIBRATION_INPUT_PATH, day)
        if not os.path.isdir(day_dir):
            continue
        for experiment in sorted(os.listdir(day_dir)):
            group_dir = os.path.join(day_dir, experiment)
            rgb_dir = os.path.join(group_dir, "images", "rgb")
            ir_dir = os.path.join(group_dir, "images", "ir")
            if os.path.isdir(rgb_dir) and os.path.isdir(ir_dir):
                calibration_groups[f"{day}/{experiment}"] = {
                    "root": group_dir, "rgb": rgb_dir, "ir": ir_dir,
                }

    for day in sorted(os.listdir(DATASET_EXPERIMENT_INPUT_PATH)):
        day_dir = os.path.join(DATASET_EXPERIMENT_INPUT_PATH, day)
        if not os.path.isdir(day_dir):
            continue
        for experiment in sorted(os.listdir(day_dir)):
            group_dir = os.path.join(day_dir, experiment)
            clips = []
            for category in sorted(os.listdir(group_dir)):
                category_dir = os.path.join(group_dir, category)
                if not os.path.isdir(category_dir):
                    continue
                for clip in sorted(os.listdir(category_dir)):
                    clip_dir = os.path.join(category_dir, clip)
                    rgb_dir = os.path.join(clip_dir, "rgb")
                    ir_dir = os.path.join(clip_dir, "ir")
                    if os.path.isdir(rgb_dir) and os.path.isdir(ir_dir):
                        clips.append({
                            "category": category, "clip": clip,
                            "root": clip_dir, "rgb": rgb_dir, "ir": ir_dir,
                        })
            if clips:
                experiment_groups[f"{day}/{experiment}"] = clips
    return calibration_groups, experiment_groups


def _group_parts(group):
    parts = group.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"组名必须是 day/experiment: {group!r}")
    return parts


def _clip_output(group, clip):
    day, experiment = _group_parts(group)
    return os.path.join(
        DATASET_REGISTERED_OUTPUT_PATH, day, experiment,
        clip["category"], clip["clip"],
    )


def _model_output(group):
    return os.path.join(DATASET_MODEL_OUTPUT_PATH, *_group_parts(group))


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _complete_clip(output_dir, expected_names, model_path):
    report_path = os.path.join(output_dir, "application_report.yaml")
    if not os.path.isfile(report_path):
        return False
    try:
        report = _load_yaml(report_path)
    except (OSError, yaml.YAMLError):
        return False
    if os.path.abspath(report.get("model_path", "")) != os.path.abspath(model_path):
        return False
    summary = report.get("summary", {})
    if summary.get("successful_pairs") != len(expected_names):
        return False
    expected = set(expected_names)
    for subdir in ("rgb", "ir", "preview"):
        folder = os.path.join(output_dir, subdir)
        if not os.path.isdir(folder):
            return False
        actual = {
            os.path.splitext(name)[0].casefold() for name in os.listdir(folder)
            if name.lower().endswith(".png")
        }
        if actual != expected:
            return False
    return True


def _replace_directory(source, target):
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(source, target)


def _write_report(report):
    os.makedirs(os.path.dirname(DATASET_BATCH_REPORT_PATH), exist_ok=True)
    with open(DATASET_BATCH_REPORT_PATH, "w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)


def run_pipeline(groups=None, dry_run=False, resume=False, force=False):
    calibration_groups, experiment_groups = discover_dataset()
    missing_models = sorted(experiment_groups.keys() - calibration_groups.keys())
    orphan_models = sorted(calibration_groups.keys() - experiment_groups.keys())
    if missing_models:
        raise RuntimeError(f"实验组缺少同名标定组: {missing_models}")
    selected = sorted(experiment_groups)
    if groups:
        requested = {item.replace("\\", "/") for item in groups}
        unknown = sorted(requested - set(selected))
        if unknown:
            raise KeyError(f"不存在的数据组: {unknown}")
        selected = [item for item in selected if item in requested]

    report = {
        "date": datetime.now().isoformat(),
        "status": "dry_run" if dry_run else "running",
        "input": {
            "calibration_root": DATASET_CALIBRATION_INPUT_PATH,
            "experiment_root": DATASET_EXPERIMENT_INPUT_PATH,
        },
        "output_root": DATASET_CALIBRATED_OUTPUT_PATH,
        "selected_groups": selected,
        "orphan_calibration_groups": orphan_models,
        "groups": [],
    }
    print(f"Discovered {len(calibration_groups)} calibration groups, "
          f"{len(experiment_groups)} experiment groups, selected={len(selected)}")

    preflight = {}
    for group in selected:
        calibration = calibration_groups[group]
        cal_info = _validate_pair_dirs(
            calibration["rgb"], calibration["ir"],
            expected_size=CALIBRATION_RGB_REFERENCE_SIZE, read_all=True,
        )
        clips = []
        for clip in experiment_groups[group]:
            info = _validate_pair_dirs(
                clip["rgb"], clip["ir"],
                expected_size=CALIBRATION_RGB_REFERENCE_SIZE, read_all=False,
            )
            clips.append({**clip, **info})
        preflight[group] = {"calibration": cal_info, "clips": clips}
        print(f"  {group}: calibration={cal_info['count']}, "
              f"clips={len(clips)}, experiment_pairs={sum(x['count'] for x in clips)}")
    if dry_run:
        report["status"] = "dry_run_complete"
        report["summary"] = {
            "groups": len(selected),
            "clips": sum(len(preflight[g]["clips"]) for g in selected),
            "experiment_pairs": sum(
                sum(x["count"] for x in preflight[g]["clips"]) for g in selected
            ),
        }
        return report

    for group in selected:
        group_record = {"group": group, "status": "running", "clips": []}
        report["groups"].append(group_record)
        _write_report(report)
        try:
            model_dir = _model_output(group)
            model_path = os.path.join(model_dir, "calibration.yaml")
            use_existing_model = resume and not force and os.path.isfile(model_path)
            if use_existing_model:
                model = _load_yaml(model_path)
                if model.get("metadata", {}).get("num_total_pairs") != preflight[group]["calibration"]["count"]:
                    use_existing_model = False
            if not use_existing_model:
                temporary_model_dir = model_dir + ".tmp"
                if os.path.exists(temporary_model_dir):
                    shutil.rmtree(temporary_model_dir)
                calibration = calibration_groups[group]
                model = calibrate_group(
                    calibration["rgb"], calibration["ir"], temporary_model_dir,
                    settings={
                        "checkerboard": CHECKERBOARD,
                        "rgb_reference_size": CALIBRATION_RGB_REFERENCE_SIZE,
                        "ir_reference_size": CALIBRATION_IR_REFERENCE_SIZE,
                        "ir_rotation": CALIBRATION_IR_ROTATION,
                        "minimum_valid_pairs": MIN_VALID_PAIRS,
                        "minimum_provisional_valid_pairs": MIN_PROVISIONAL_VALID_PAIRS,
                    },
                )
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                _replace_directory(temporary_model_dir, model_dir)
            group_record["model_path"] = os.path.abspath(model_path)
            group_record["model_quality"] = {
                key: model.get("calibration", {}).get(key) for key in (
                    "quality_status", "geometry_passed", "sample_count_passed",
                    "diversity_warning", "eligible_for_application",
                )
            }
            if not model.get("calibration", {}).get("eligible_for_application", False):
                raise RuntimeError("标定模型未达到临时应用所需的最低有效对数")

            for clip in preflight[group]["clips"]:
                output_dir = _clip_output(group, clip)
                clip_record = {
                    "category": clip["category"], "clip": clip["clip"],
                    "input_pairs": clip["count"], "output_dir": output_dir,
                }
                group_record["clips"].append(clip_record)
                if resume and not force and _complete_clip(
                    output_dir, clip["names"], model_path
                ):
                    clip_record["status"] = "reused"
                    continue
                temporary_output = output_dir + ".tmp"
                if os.path.exists(temporary_output):
                    shutil.rmtree(temporary_output)
                clip_report = process_all_pairs(
                    rgb_dir=clip["rgb"], ir_dir=clip["ir"],
                    output_dir=temporary_output, model_path=model_path,
                )
                summary = clip_report["summary"]
                if summary["failed_pairs"] or summary["successful_pairs"] != clip["count"]:
                    raise RuntimeError(
                        f"{group}/{clip['category']}/{clip['clip']} 输出不完整: {summary}"
                    )
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                _replace_directory(temporary_output, output_dir)
                clip_record["status"] = "success"
                clip_record["summary"] = summary
                _write_report(report)
            group_record["status"] = "success"
        except Exception as error:
            group_record["status"] = "failed"
            group_record["error"] = str(error)
            print(f"[FAILED] {group}: {error}", file=sys.stderr)
        _write_report(report)

    statuses = [item["status"] for item in report["groups"]]
    report["status"] = "complete" if all(x == "success" for x in statuses) else "complete_with_failures"
    report["summary"] = {
        "groups": len(report["groups"]),
        "successful_groups": statuses.count("success"),
        "failed_groups": statuses.count("failed"),
        "clips": sum(len(item["clips"]) for item in report["groups"]),
        "successful_or_reused_clips": sum(
            clip.get("status") in ("success", "reused")
            for item in report["groups"] for clip in item["clips"]
        ),
    }
    _write_report(report)
    return report


def _parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="*", metavar="DAY/EXPERIMENT",
        help="只处理指定组；默认处理全部组",
    )
    parser.add_argument("--dry-run", action="store_true", help="只扫描和校验，不写输出")
    parser.add_argument("--resume", action="store_true", help="复用已完整且模型一致的结果")
    parser.add_argument("--force", action="store_true", help="重新生成指定组及其 clip")
    return parser


def main():
    args = _parser().parse_args()
    report = run_pipeline(
        groups=args.groups, dry_run=args.dry_run, resume=args.resume, force=args.force
    )
    print(yaml.safe_dump(report.get("summary", {}), allow_unicode=True, sort_keys=False))
    if report.get("status") == "complete_with_failures":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
