"""可复现、按 epoch 改变顺序的训练采样器。"""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, Mapping

import torch
from torch.utils.data import Sampler

from .schema import DataContractError


class EpochShuffleSampler(Sampler[int]):
    """每个 epoch 无放回打乱全部样本，顺序仅由 seed 和 epoch 决定。"""

    def __init__(self, data_source: Sized, *, base_seed: int) -> None:
        self.data_source = data_source
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        seed = (self.base_seed + self.epoch) % (2**63 - 1)
        generator.manual_seed(seed)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def build_train_sampler(
    data_source: Sized,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> EpochShuffleSampler:
    strategy = str(config["training"].get("sampler", "epoch_shuffle"))
    if strategy != "epoch_shuffle":
        raise DataContractError(f"不支持的训练 sampler: {strategy}")
    return EpochShuffleSampler(data_source, base_seed=seed)


__all__ = ["EpochShuffleSampler", "build_train_sampler"]
