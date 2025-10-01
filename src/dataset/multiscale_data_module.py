import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper
from .data_module import DataLoaderCfg, DataLoaderStageCfg


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


def get_data_shim_scaled(encoder: nn.Module, scaling_factor: int = 2) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim_scaled"):
        #shims.append(encoder.get_data_shim())
        shims.append(encoder.get_data_shim_scaled(scaling_factor=scaling_factor))

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim

#@dataclass
#class DataLoaderStageCfg:
#    batch_size: int
#    num_workers: int
#    persistent_workers: bool
#    seed: int | None
#
#
#@dataclass
#class DataLoaderCfg:
#    train: DataLoaderStageCfg
#    test: DataLoaderStageCfg
#    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class MultiScaleDataModule(LightningDataModule):
    #dataset_cfg: DatasetCfg
    dataset_cfg_480p: DatasetCfg
    dataset_cfg_960p: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        #dataset_cfg: DatasetCfg,
        dataset_cfg_480p: DatasetCfg,
        dataset_cfg_960p: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        #self.dataset_cfg = dataset_cfg
        self.dataset_cfg_480p = dataset_cfg_480p
        self.dataset_cfg_960p = dataset_cfg_960p
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        #dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        dataset_480p = get_dataset(self.dataset_cfg_480p, "train", self.step_tracker)
        dataset_960p = get_dataset(self.dataset_cfg_960p, "train", self.step_tracker)
        dataset_480p = self.dataset_shim(dataset_480p, "train")
        dataset_960p = self.dataset_shim(dataset_960p, "train")
        dataloader_480p = DataLoader(
            dataset_480p,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset_480p, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )
        dataloader_960p = DataLoader(
            dataset_480p,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset_480p, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )
        return CombinedLoader({
            "480p": dataloader_480p,
            "960p": dataloader_960p,
        }, mode="sequential")

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self, dataset_cfg=None):
        #dataset = get_dataset(
        #    self.dataset_cfg if dataset_cfg is None else dataset_cfg,
        #    "test",
        #    self.step_tracker,
        #)
        assert dataset_cfg is None
        dataset_480p = get_dataset(
            self.dataset_cfg_480p if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset_960p = get_dataset(
            self.dataset_cfg_960p if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset_480p = self.dataset_shim(dataset_480p, "test")
        dataset_960p = self.dataset_shim(dataset_960p, "test")
        dataloader_480p = DataLoader(
            dataset_480p,
            self.data_loader_cfg.test.batch_size,
            #num_workers=self.data_loader_cfg.test.num_workers,
            num_workers=1,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )
        dataloader_960p = DataLoader(
            dataset_960p,
            self.data_loader_cfg.test.batch_size,
            #num_workers=self.data_loader_cfg.test.num_workers,
            num_workers=1,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )
        return CombinedLoader({
            "480p": dataloader_480p,
            "960p": dataloader_960p,
        }, mode="max_size_cycle")
