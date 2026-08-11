"""加载 checkpoint，对显式图像对或已审计 test split 执行融合推理。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from data_pipeline.schema import (
    SAMPLE_FIELDS,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    canonical_rows_hash,
    load_dataset_config,
    parse_bool,
    read_csv_rows,
    resolve_data_path,
)
from inference_utils import (
    infer_pair_tensor,
    load_inference_pair,
    load_model_checkpoint,
    save_fused_png,
)
from model import CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class InputPair:
    relative_path: Path
    vis_path: Path
    ir_path: Path
    sample_id: str = ""
    category: str = ""
    leakage_group_id: str = ""


def _png_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise DataContractError(f"输入目录不存在: {root}")
    files = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.png"))
        if path.is_file()
    }
    if not files:
        raise DataContractError(f"输入目录中没有 PNG: {root}")
    return files


def collect_input_pairs(args: argparse.Namespace) -> list[InputPair]:
    if args.vis is not None:
        if args.ir is None or args.vis_dir is not None or args.ir_dir is not None:
            raise DataContractError("单对模式必须同时使用 --vis 和 --ir")
        vis_path = args.vis.resolve()
        ir_path = args.ir.resolve()
        if vis_path.stem != ir_path.stem:
            raise DataContractError(
                f"VIS/IR 文件 stem 必须一致: {vis_path.name} vs {ir_path.name}"
            )
        return [InputPair(Path(vis_path.name), vis_path, ir_path)]
    if args.vis_dir is None or args.ir_dir is None or args.ir is not None:
        raise DataContractError("目录模式必须同时使用 --vis-dir 和 --ir-dir")
    vis_root = args.vis_dir.resolve()
    ir_root = args.ir_dir.resolve()
    vis_files = _png_files(vis_root)
    ir_files = _png_files(ir_root)
    missing_ir = sorted(set(vis_files) - set(ir_files))
    missing_vis = sorted(set(ir_files) - set(vis_files))
    if missing_ir or missing_vis:
        raise DataContractError(
            "IR/RGB 目录未严格同名配对: "
            f"missing_ir={missing_ir[:10]}, missing_vis={missing_vis[:10]}"
        )
    return [
        InputPair(Path(relative), vis_files[relative], ir_files[relative])
        for relative in sorted(vis_files)
    ]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"JSON 根节点必须是对象: {path}")
    return value


def collect_split_input_pairs(
    data_root: Path,
    config: dict[str, Any],
    split_version: str,
    split_name: str,
) -> tuple[list[InputPair], dict[str, Any]]:
    """从已完整审计的正式 split 精确收集推理图像对。"""
    if split_name != "test":
        raise DataContractError("manifest 推理当前只允许 --split test，避免误用训练/验证集")
    manifest_path = data_root / str(config["paths"]["manifests"]) / "samples.csv"
    split_dir = data_root / str(config["paths"]["splits"]) / split_version
    split_path = split_dir / f"{split_name}.csv"
    split_config = _load_json_object(split_dir / "split_config.json")
    audit = _load_json_object(split_dir / "audit.json")
    samples = read_csv_rows(manifest_path, SAMPLE_FIELDS)
    manifest_hash = canonical_rows_hash(samples, SAMPLE_FIELDS)
    for source_name, source in (("split_config.json", split_config), ("audit.json", audit)):
        if source.get("dataset_manifest_sha256") != manifest_hash:
            raise DataContractError(f"{source_name} 的 manifest 哈希与 samples.csv 不一致")
    if split_config.get("split_version") != split_version:
        raise DataContractError("split_config.json 的 split_version 与请求不一致")
    if not split_config.get("formal", False):
        raise DataContractError("manifest 推理只接受 formal split")
    if int(split_config.get("split_schema_version", -1)) != SPLIT_SCHEMA_VERSION:
        raise DataContractError(
            f"manifest 推理只接受 split schema v{SPLIT_SCHEMA_VERSION}"
        )
    if split_config.get("algorithm") != config["splitting"]["algorithm"]:
        raise DataContractError("split 算法与 dataset.yaml 不一致")
    if audit.get("audit_level") != "full":
        raise DataContractError("manifest 推理前必须完成 full audit")
    if int(audit.get("fatal_count", -1)) != 0 or not audit.get("formal_ready", False):
        raise DataContractError(f"split 审计未通过: {audit.get('fatals', audit.get('failures'))}")

    all_ids = [row["sample_id"] for row in samples]
    if len(all_ids) != len(set(all_ids)):
        raise DataContractError("samples.csv 包含重复 sample_id")
    id_to_sample = {row["sample_id"]: row for row in samples}
    split_ids: dict[str, list[str]] = {}
    for current_split in ("train", "val", "test"):
        current_path = split_dir / f"{current_split}.csv"
        ids = [
            row["sample_id"].strip()
            for row in read_csv_rows(current_path, ["sample_id"])
        ]
        if not ids or any(not sample_id for sample_id in ids):
            raise DataContractError(f"split manifest 为空或包含空 sample_id: {current_path}")
        if len(ids) != len(set(ids)):
            raise DataContractError(f"split manifest 包含重复 sample_id: {current_path}")
        unknown = sorted(set(ids) - set(id_to_sample))
        if unknown:
            raise DataContractError(
                f"split 引用了 samples.csv 中不存在的 ID: {unknown[:10]}"
            )
        split_ids[current_split] = ids
    for index, first in enumerate(("train", "val", "test")):
        for second in ("train", "val", "test")[index + 1 :]:
            overlap = sorted(set(split_ids[first]) & set(split_ids[second]))
            if overlap:
                raise DataContractError(
                    f"样本跨 split 重复: {first}/{second}: {overlap[:10]}"
                )
            first_groups = {
                id_to_sample[sample_id]["leakage_group_id"]
                for sample_id in split_ids[first]
            }
            second_groups = {
                id_to_sample[sample_id]["leakage_group_id"]
                for sample_id in split_ids[second]
            }
            group_overlap = sorted(first_groups & second_groups)
            if group_overlap:
                raise DataContractError(
                    f"防泄漏组跨 split 重复: {first}/{second}: {group_overlap[:10]}"
                )
    sample_ids = split_ids[split_name]

    pairs: list[InputPair] = []
    for sample_id in sample_ids:
        row = id_to_sample[sample_id]
        if not parse_bool(row["usable"]):
            raise DataContractError(f"test split 包含 usable=false 样本: {sample_id}")
        vis_path = resolve_data_path(data_root, row["rgb_path"])
        ir_path = resolve_data_path(data_root, row["ir_path"])
        relative_path = Path(
            row["study_day"],
            row["experiment_id"],
            row["category"],
            row["clip_id"],
            vis_path.name,
        )
        pairs.append(
            InputPair(
                relative_path=relative_path,
                vis_path=vis_path,
                ir_path=ir_path,
                sample_id=sample_id,
                category=row["category"],
                leakage_group_id=row["leakage_group_id"],
            )
        )
    return pairs, {
        "mode": "formal_split",
        "data_root": str(data_root),
        "split_version": split_version,
        "split": split_name,
        "dataset_manifest_sha256": manifest_hash,
        "split_manifest_sha256": canonical_rows_hash(
            ({"sample_id": sample_id} for sample_id in sample_ids), ["sample_id"]
        ),
    }


def _make_run_directory(output_root: Path, untrained_smoke: bool) -> Path:
    prefix = "inference-smoke-untrained" if untrained_smoke else "inference"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{prefix}-{timestamp}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def run_inference(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    config_path = (args.config or data_root / "dataset.yaml").resolve()
    config = load_dataset_config(config_path)
    inference_config = config.get("inference", {})
    factor = int(config["preprocessing"]["model_downsample_factor"])
    pad_mode = str(config["preprocessing"].get("pad_mode", "reflect"))
    tile_size = int(args.tile_size or inference_config.get("tile_size", config["preprocessing"]["crop_size"]))
    overlap = int(
        args.overlap if args.overlap is not None else inference_config.get("overlap", 64)
    )
    max_full_tokens = int(
        args.max_full_tokens
        if args.max_full_tokens is not None
        else inference_config.get("max_full_tokens", 12_544)
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DataContractError("请求了 CUDA，但当前 PyTorch 环境不可用 CUDA")
    amp_enabled = (
        bool(config.get("training", {}).get("amp", True))
        and device.type == "cuda"
        and not args.no_amp
    )
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if args.split_version is not None and args.allow_random_weights:
        raise DataContractError("正式 test split 禁止使用随机未训练权重")
    if args.split_version is not None and (args.ir is not None or args.ir_dir is not None):
        raise DataContractError("--split-version 不能与 --ir/--ir-dir 混用")
    model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention)
    if args.checkpoint is not None:
        weight_source = {
            "status": "checkpoint_loaded",
            **load_model_checkpoint(model, args.checkpoint),
        }
        untrained_smoke = False
    elif args.allow_random_weights:
        weight_source = {
            "status": "random_untrained_smoke_only",
            "seed": int(args.seed),
            "warning": "输出只验证代码路径，不代表融合质量，不能用于评估或检测。",
        }
        untrained_smoke = True
    else:
        raise DataContractError("必须提供 --checkpoint，或显式使用 --allow-random-weights")
    model = model.to(device).eval()
    if args.split_version is not None:
        pairs, input_source = collect_split_input_pairs(
            data_root, config, args.split_version, args.split
        )
    else:
        pairs = collect_input_pairs(args)
        input_source = {"mode": "explicit_paths"}
    run_dir = _make_run_directory(args.output_root.resolve(), untrained_smoke)
    image_root = run_dir / "images"
    pair_results: list[dict[str, object]] = []
    for pair in pairs:
        vis, ir = load_inference_pair(pair.vis_path, pair.ir_path, config)
        fused, details = infer_pair_tensor(
            model,
            vis,
            ir,
            device=device,
            requested_mode=args.mode,
            tile_size=tile_size,
            overlap=overlap,
            factor=factor,
            pad_mode=pad_mode,
            max_full_tokens=max_full_tokens,
            force_full=bool(args.force_full),
            amp_enabled=amp_enabled,
        )
        filename = pair.relative_path.name
        if untrained_smoke:
            filename = f"SMOKE_UNTRAINED__{filename}"
        relative_output = pair.relative_path.with_name(filename)
        output_path = image_root / relative_output
        save_fused_png(fused, output_path)
        pair_results.append(
            {
                "sample_id": pair.sample_id or None,
                "category": pair.category or None,
                "leakage_group_id": pair.leakage_group_id or None,
                "vis_path": str(pair.vis_path),
                "ir_path": str(pair.ir_path),
                "output_path": str(output_path),
                **details,
            }
        )
        print(
            json.dumps(
                {
                    "input": pair.relative_path.as_posix(),
                    "output": str(output_path),
                    "mode": details["actual_mode"],
                    "shape": [details["output_height"], details["output_width"]],
                },
                ensure_ascii=False,
            )
        )
    metadata = {
        "schema_version": 1,
        "result_status": (
            "smoke_only_untrained"
            if untrained_smoke
            else "heldout_split_inference"
            if args.split_version is not None
            else "checkpoint_inference"
        ),
        "config_path": str(config_path),
        "input_source": input_source,
        "weights": weight_source,
        "device": str(device),
        "amp": amp_enabled,
        "requested_mode": args.mode,
        "tile_size": tile_size,
        "overlap": overlap,
        "max_full_tokens": max_full_tokens,
        "pair_count": len(pair_results),
        "pairs": pair_results,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--vis", type=Path, help="单张 VIS/RGB PNG")
    inputs.add_argument("--vis-dir", type=Path, help="VIS/RGB PNG 目录，可递归")
    inputs.add_argument("--split-version", help="读取已审计的正式 split 版本")
    parser.add_argument("--ir", type=Path, help="与 --vis 同 stem 的 IR PNG")
    parser.add_argument("--ir-dir", type=Path, help="与 --vis-dir 相对路径严格配对的 IR 目录")
    parser.add_argument("--split", choices=("test",), default="test")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="仅用于代码路径 smoke；输出会显式标记为未训练",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--mode", choices=("auto", "full", "sliding"), default="auto")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--max-full-tokens", type=int)
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = run_inference(args)
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
