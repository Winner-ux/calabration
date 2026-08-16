"""从正式数据集的 paired 目录构建 samples.csv 和 groups.csv。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest import build_manifest_rows, write_manifest_files  # noqa: E402
from data_pipeline.schema import DataContractError, load_dataset_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_ROOT / "dataset.yaml")
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    manifest_dir = args.data_root / str(config["paths"]["manifests"])
    assignments = args.assignments or manifest_dir / "group_assignments.csv"
    samples, groups = build_manifest_rows(args.data_root, config, assignments)
    write_manifest_files(
        manifest_dir / "samples.csv",
        manifest_dir / "groups.csv",
        samples,
        groups,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "samples": len(samples),
                "usable": sum(row["usable"] == "true" for row in samples),
                "groups": len(groups),
                "output": str(manifest_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
