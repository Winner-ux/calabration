"""生成不可覆盖的两阶段版本化 leakage-group split。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "calibrated_v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.schema import (  # noqa: E402
    GROUP_FIELDS,
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    read_csv_rows,
)
from data_pipeline.splitting import create_split_version  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_ROOT / "dataset.yaml")
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    manifest_dir = args.data_root / str(config["paths"]["manifests"])
    samples = read_csv_rows(manifest_dir / "samples.csv", SAMPLE_FIELDS)
    groups = read_csv_rows(manifest_dir / "groups.csv", GROUP_FIELDS)
    seed = int(args.seed if args.seed is not None else config["splitting"]["default_seed"])
    output = create_split_version(args.data_root, args.version, samples, groups, config, seed)
    print(json.dumps({"split_version": args.version, "seed": seed, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
