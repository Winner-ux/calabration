"""审计 manifest、split、重复候选和配准质量。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.audit import audit_dataset, write_audit_outputs  # noqa: E402
from data_pipeline.schema import (  # noqa: E402
    GROUP_FIELDS,
    SAMPLE_FIELDS,
    DataContractError,
    load_dataset_config,
    read_csv_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data" / "dataset.yaml")
    parser.add_argument("--split-version")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    manifest_dir = args.data_root / str(config["paths"]["manifests"])
    samples = read_csv_rows(manifest_dir / "samples.csv", SAMPLE_FIELDS)
    groups = read_csv_rows(manifest_dir / "groups.csv", GROUP_FIELDS)
    split_dir = (
        args.data_root / str(config["paths"]["splits"]) / args.split_version
        if args.split_version
        else None
    )
    audit, registration, duplicates = audit_dataset(
        args.data_root, samples, groups, config, split_dir
    )
    write_audit_outputs(
        args.data_root,
        audit,
        registration,
        duplicates,
        config,
        split_dir=split_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["fatal_count"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileExistsError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

