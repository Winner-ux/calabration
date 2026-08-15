"""比较正式训练 Dataset 在不同 worker 数下的固定样本吞吐。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.schema import load_dataset_config  # noqa: E402
from data_pipeline.transforms import build_paired_transform  # noqa: E402
from main.dataset import PairedFusionDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()
    root = args.data_root.resolve()
    config = load_dataset_config(root / "dataset.yaml")
    results = {}
    for workers in args.workers:
        dataset = PairedFusionDataset(
            root,
            root / "manifests" / "samples.csv",
            root / "splits" / args.split_version / "train.csv",
            paired_transform=build_paired_transform(config, "train"),
            config_path=root / "dataset.yaml",
            base_seed=42,
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        started = time.perf_counter()
        count = 0
        for batch in loader:
            count += len(batch["sample_id"])
            if count >= args.samples:
                break
        elapsed = time.perf_counter() - started
        results[str(workers)] = {
            "samples": count,
            "seconds": elapsed,
            "samples_per_second": count / elapsed,
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
