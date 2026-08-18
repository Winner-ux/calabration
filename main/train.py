"""基于正式 manifest/split 的最小 IR/RGB 融合训练入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.schema import (
    SAMPLE_FIELDS,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    canonical_rows_hash,
    load_dataset_config,
    read_csv_rows,
)
from data_pipeline.sampling import build_train_sampler
from data_pipeline.transforms import build_paired_transform, build_rgb_transform
from .dataset import PairedFusionDataset
from .inference_utils import load_initial_model_checkpoint, sha256_file
from .loss import FusionLoss
from model import IR_MODES, CrossAttention, DecoderBlock, FusionBlock, Residual, ResNetFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
DEFAULT_LOSS_WEIGHTS = {
    "intensity": 1.0,
    "gradient": 10.0,
    "ssim": 5.0,
    "edge": 2.0,
}


def _loss_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name, default in DEFAULT_LOSS_WEIGHTS.items():
        raw = getattr(args, f"lambda_{name}", None)
        weights[name] = float(default if raw is None else raw)
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0:
            raise DataContractError(f"lambda_{name} 必须是有限非负数，实际为 {value!r}")
    if not any(value > 0 for value in weights.values()):
        raise DataContractError("损失权重不能全部为 0")
    if not math.isclose(sum(weights.values()), 18.0, rel_tol=0.0, abs_tol=1e-6):
        raise DataContractError(
            f"本轮权重消融要求四项权重之和为 18，实际为 {sum(weights.values()):.8g}"
        )
    return weights


def _checkpoint_loss_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
    if not isinstance(raw, dict):
        raise DataContractError("checkpoint loss_weights 必须是对象")
    try:
        return {name: float(raw[name]) for name in DEFAULT_LOSS_WEIGHTS}
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"checkpoint loss_weights 非法: {raw!r}") from exc


def _validate_checkpoint_loss_weights(
    config: dict[str, Any], current: dict[str, float]
) -> None:
    checkpoint_weights = _checkpoint_loss_weights(config)
    if checkpoint_weights != current:
        raise DataContractError(
            "checkpoint 的 loss_weights 与当前训练配置不一致: "
            f"checkpoint={checkpoint_weights!r}, current={current!r}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"JSON 根节点必须是对象: {path}")
    return value


def validate_training_inputs(
    data_root: Path,
    config: dict[str, Any],
    split_version: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """拒绝未经完整审计或与 manifest 不一致的 split。"""
    manifest_dir = data_root / str(config["paths"]["manifests"])
    samples_path = manifest_dir / "samples.csv"
    samples = read_csv_rows(samples_path, SAMPLE_FIELDS)
    split_dir = data_root / str(config["paths"]["splits"]) / split_version
    split_config = _load_json(split_dir / "split_config.json")
    audit = _load_json(split_dir / "audit.json")
    expected_hash = canonical_rows_hash(samples, SAMPLE_FIELDS)
    if split_config.get("dataset_manifest_sha256") != expected_hash:
        raise DataContractError("split_config 的 manifest 哈希与当前 samples.csv 不一致")
    if audit.get("dataset_manifest_sha256") != expected_hash:
        raise DataContractError("audit.json 的 manifest 哈希与当前 samples.csv 不一致")
    if not split_config.get("formal", False):
        raise DataContractError("训练入口只接受 formal split")
    if int(split_config.get("split_schema_version", -1)) != SPLIT_SCHEMA_VERSION:
        raise DataContractError(
            f"训练入口只接受 split schema v{SPLIT_SCHEMA_VERSION}"
        )
    if split_config.get("algorithm") != config["splitting"]["algorithm"]:
        raise DataContractError("split 算法与当前 dataset.yaml 不一致")
    if audit.get("audit_level") != "full":
        raise DataContractError("请先运行 scripts/audit_dataset.py 完成 full audit")
    if int(audit.get("fatal_count", -1)) != 0 or not audit.get("formal_ready", False):
        raise DataContractError(f"数据审计未通过: {audit.get('fatals', audit.get('failures'))}")
    for filename in (
        "group_assignments.csv",
        "experiment_assignments.csv",
        "train_pool.csv",
        "train.csv",
        "val.csv",
        "test.csv",
    ):
        if not (split_dir / filename).is_file():
            raise DataContractError(f"split 缺少 {filename}: {split_dir}")
    return samples_path, split_dir, split_config


def _split_manifests_hash(split_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in (
        "group_assignments.csv",
        "experiment_assignments.csv",
        "train_pool.csv",
        "train.csv",
        "val.csv",
        "test.csv",
    ):
        path = split_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_dataset(
    data_root: Path,
    samples_path: Path,
    split_dir: Path,
    split: str,
    config_path: Path,
    config: dict[str, Any],
    seed: int,
) -> PairedFusionDataset:
    return PairedFusionDataset(
        data_root=data_root,
        samples_manifest=samples_path,
        split_manifest=split_dir / f"{split}.csv",
        paired_transform=build_paired_transform(config, split),
        rgb_transform=build_rgb_transform(config, split),
        ir_conversion="bt601",
        config_path=config_path,
        base_seed=seed,
    )


def _make_run_directory(output_root: Path, smoke_only: bool) -> Path:
    prefix = "smoke" if smoke_only else "train"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = output_root / f"{prefix}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _append_log(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _validate_epoch(
    model: torch.nn.Module,
    criterion: FusionLoss,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "intensity_loss": 0.0,
        "gradient_loss": 0.0,
        "ssim_loss": 0.0,
        "edge_loss": 0.0,
    }
    count = 0
    with torch.inference_mode():
        for batch in loader:
            vis = batch["vis"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused = model(vis, ir)
            loss, components = criterion(fused.float(), ir.float(), vis.float())
            batch_size = vis.shape[0]
            totals["loss"] += float(loss.detach()) * batch_size
            for name in components:
                totals[name] += float(components[name]) * batch_size
            count += vis.shape[0]
    if count == 0:
        raise DataContractError("validation DataLoader 为空")
    return {name: value / count for name, value in totals.items()}


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    return metadata


def run_training(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    config = load_dataset_config(config_path)
    loss_weights = _loss_weights_from_args(args)
    ir_mode = str(getattr(args, "ir_mode", "gray"))
    if ir_mode not in IR_MODES:
        raise DataContractError(f"非法 ir_mode: {ir_mode!r}")
    resume = getattr(args, "resume", None)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if resume is not None and init_checkpoint is not None:
        raise DataContractError("--resume 与 --init-checkpoint 不能同时使用")
    if getattr(args, "best_checkpoint", None) and resume is None:
        raise DataContractError("--best-checkpoint 只能与 --resume 一起使用")
    if getattr(args, "reset_optimizer", False) and resume is None:
        raise DataContractError("--reset-optimizer 只能与 --resume 一起使用")
    seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples_path, split_dir, split_config = validate_training_inputs(
        data_root, config, args.split_version
    )
    train_dataset = _build_dataset(
        data_root, samples_path, split_dir, "train", config_path, config, seed
    )
    val_dataset = _build_dataset(
        data_root, samples_path, split_dir, "val", config_path, config, seed
    )
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else config["training"]["num_workers"])
    if batch_size <= 0 or workers < 0:
        raise DataContractError("batch_size 必须为正，num_workers 不能为负")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    train_sampler = build_train_sampler(train_dataset, config, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    model = ResNetFusion(
        Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode=ir_mode
    ).to(device)
    initialization = None
    if init_checkpoint is not None:
        initialization = load_initial_model_checkpoint(
            model, init_checkpoint, ir_mode=ir_mode
        )
    criterion = FusionLoss(
        lambda_intensity=loss_weights["intensity"],
        lambda_gradient=loss_weights["gradient"],
        lambda_ssim=loss_weights["ssim"],
        lambda_edge=loss_weights["edge"],
    ).to(device)
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else config["training"]["learning_rate"]
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise DataContractError("learning_rate 必须是有限正数")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epochs = int(args.epochs or config["training"]["epochs"])
    max_steps = 1 if args.smoke_only and args.max_steps is None else args.max_steps
    if args.smoke_only and max_steps != 1:
        raise DataContractError("--smoke-only 只允许 --max-steps 1")
    split_manifest_hash = _split_manifests_hash(split_dir)
    run_metadata = {
        "smoke_only": bool(args.smoke_only),
        "split_version": args.split_version,
        "split_schema_version": int(split_config["split_schema_version"]),
        "split_manifest_sha256": split_manifest_hash,
        "seed": seed,
        "sampler": str(config["training"]["sampler"]),
        "device": str(device),
        "amp": amp_enabled,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_workers": workers,
        "epochs": epochs,
        "ir_mode": ir_mode,
        "loss_weights": loss_weights,
        "initialization": initialization,
        "reset_optimizer": bool(getattr(args, "reset_optimizer", False)),
        "runtime": _runtime_metadata(device),
    }
    start_epoch = 0
    best_val = float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("run_config", {})
        for key in ("split_version", "split_schema_version", "split_manifest_sha256", "seed", "sampler"):
            if checkpoint_config.get(key) != run_metadata[key]:
                raise DataContractError(
                    f"checkpoint 的 {key} 与当前训练配置不一致: "
                    f"checkpoint={checkpoint_config.get(key)!r}, current={run_metadata[key]!r}"
                )
        checkpoint_ir_mode = str(checkpoint_config.get("ir_mode", "gray"))
        if checkpoint_ir_mode != ir_mode:
            raise DataContractError(
                f"checkpoint 的 ir_mode 与当前模式不一致: "
                f"checkpoint={checkpoint_ir_mode}, current={ir_mode}"
            )
        _validate_checkpoint_loss_weights(checkpoint_config, loss_weights)
        # 延续最初 warm-start 的可审计来源；resume 本身不应抹掉初始化记录。
        run_metadata["initialization"] = checkpoint_config.get("initialization")
        run_metadata["resumed_from"] = {
            "path": str(Path(resume).resolve()),
            "sha256": sha256_file(resume),
            "epoch": checkpoint.get("epoch"),
        }
        model.load_state_dict(checkpoint["model"])
        if not getattr(args, "reset_optimizer", False):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint.get("scaler", {}))
        if args.learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    run_dir = _make_run_directory(args.output_root.resolve(), args.smoke_only)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if resume and not args.smoke_only:
        previous_metrics = Path(resume).resolve().parent / "metrics.csv"
        if previous_metrics.is_file():
            shutil.copy2(previous_metrics, run_dir / "metrics.csv")
    if resume and not args.smoke_only:
        if getattr(args, "best_checkpoint", None):
            best_payload = torch.load(
                args.best_checkpoint, map_location="cpu", weights_only=False
            )
            if float(best_payload.get("best_val", float("inf"))) != best_val:
                raise DataContractError(
                    "--best-checkpoint 的 best_val 与恢复 checkpoint 不一致"
                )
            shutil.copy2(args.best_checkpoint, run_dir / "best.pt")
        else:
            baseline_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": start_epoch - 1,
                "best_val": best_val,
                "run_config": run_metadata,
            }
            torch.save(baseline_checkpoint, run_dir / "best.pt")
    global_steps = 0
    training_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        epoch_totals = {
            "loss": 0.0,
            "intensity_loss": 0.0,
            "gradient_loss": 0.0,
            "ssim_loss": 0.0,
            "edge_loss": 0.0,
        }
        epoch_count = 0
        grad_norm_count = 0
        grad_norm_sum = 0.0
        grad_norm_square_sum = 0.0
        grad_norm_max = 0.0
        amp_skipped_steps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch in train_loader:
            vis = batch["vis"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused = model(vis, ir)
            loss, components = criterion(fused.float(), ir.float(), vis.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "训练 loss 非有限值: "
                    f"loss={float(loss.detach())}, samples={batch['sample_id']}, "
                    f"fused_finite={bool(torch.isfinite(fused).all())}, "
                    f"components={components}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            )
            if not math.isfinite(grad_norm) and not amp_enabled:
                raise FloatingPointError(
                    f"训练梯度范数非有限值: grad_norm={grad_norm}, "
                    f"samples={batch['sample_id']}"
                )
            if math.isfinite(grad_norm):
                grad_norm_count += 1
                grad_norm_sum += grad_norm
                grad_norm_square_sum += grad_norm * grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
            else:
                amp_skipped_steps += 1
            scaler.step(optimizer)
            scaler.update()
            current_batch_size = vis.shape[0]
            epoch_totals["loss"] += float(loss.detach()) * current_batch_size
            for name in components:
                epoch_totals[name] += float(components[name]) * current_batch_size
            epoch_count += current_batch_size
            global_steps += 1
            if max_steps is not None and global_steps >= max_steps:
                break
        if epoch_count == 0:
            raise DataContractError("training DataLoader 为空")
        train_metrics = {
            name: value / epoch_count for name, value in epoch_totals.items()
        }
        train_loss = train_metrics["loss"]
        grad_norm_mean = (
            grad_norm_sum / grad_norm_count if grad_norm_count else None
        )
        grad_norm_variance = (
            max(
                0.0,
                grad_norm_square_sum / grad_norm_count
                - grad_norm_mean * grad_norm_mean,
            )
            if grad_norm_count and grad_norm_mean is not None
            else None
        )
        grad_norm_std = (
            math.sqrt(grad_norm_variance)
            if grad_norm_variance is not None
            else None
        )
        grad_norm_max_value = grad_norm_max if grad_norm_count else None
        if args.smoke_only:
            result = {
                **run_metadata,
                "steps": global_steps,
                "train_loss": train_loss,
                "output_shape": list(fused.shape),
                "train_grad_norm_mean": grad_norm_mean,
                "train_grad_norm_std": grad_norm_std,
                "train_grad_norm_max": grad_norm_max_value,
                "amp_skipped_steps": amp_skipped_steps,
                "amp_skipped_step_fraction": amp_skipped_steps / global_steps,
            }
            if device.type == "cuda":
                result.update(
                    {
                        "peak_gpu_memory_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_gpu_memory_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                )
            (run_dir / "smoke_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return run_dir

        val_metrics = _validate_epoch(model, criterion, val_loader, device, amp_enabled)
        val_loss = val_metrics["loss"]
        epoch_seconds = time.perf_counter() - epoch_started
        log_row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{name}": train_metrics[name] for name in train_metrics if name != "loss"},
            **{f"val_{name}": val_metrics[name] for name in val_metrics if name != "loss"},
            "train_grad_norm_mean": grad_norm_mean,
            "train_grad_norm_std": grad_norm_std,
            "train_grad_norm_max": grad_norm_max_value,
            "amp_skipped_steps": amp_skipped_steps,
            "amp_skipped_step_fraction": amp_skipped_steps
            / (grad_norm_count + amp_skipped_steps),
            "samples_per_second": epoch_count / epoch_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": time.perf_counter() - training_started,
        }
        if device.type == "cuda":
            log_row["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            log_row["peak_gpu_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
        _append_log(run_dir / "metrics.csv", log_row)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            "run_config": run_metadata,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, run_dir / "best.pt")
        print(json.dumps(log_row, ensure_ascii=False))
        if max_steps is not None and global_steps >= max_steps:
            break
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_ROOT / "dataset.yaml")
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lambda-intensity", type=float)
    parser.add_argument("--lambda-gradient", type=float)
    parser.add_argument("--lambda-ssim", type=float)
    parser.add_argument("--lambda-edge", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="只加载模型权重并从 epoch 0 重新训练，不恢复 optimizer/scaler",
    )
    parser.add_argument("--ir-mode", choices=IR_MODES, default="gray")
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        help="恢复 last.pt 时携带此前真实的 best.pt",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="恢复模型/epoch，但重新初始化 optimizer 和 GradScaler",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-steps", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_training(args)
    print(f"run_dir={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
