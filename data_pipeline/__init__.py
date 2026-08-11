"""Manifest 驱动的 IR/RGB 配对数据管理工具。"""

from .schema import (
    EXPERIMENT_ASSIGNMENT_FIELDS,
    GROUP_ASSIGNMENT_FIELDS,
    GROUP_FIELDS,
    GROUP_SPLIT_ASSIGNMENT_FIELDS,
    SAMPLE_FIELDS,
    SPLIT_SCHEMA_VERSION,
    DataContractError,
    load_dataset_config,
)

__all__ = [
    "DataContractError",
    "EXPERIMENT_ASSIGNMENT_FIELDS",
    "GROUP_ASSIGNMENT_FIELDS",
    "GROUP_FIELDS",
    "GROUP_SPLIT_ASSIGNMENT_FIELDS",
    "SAMPLE_FIELDS",
    "SPLIT_SCHEMA_VERSION",
    "load_dataset_config",
]
