from typing import Literal, Mapping

import torch
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tsl import logger
from tsl.data.loader import DisjointGraphLoader, StaticGraphLoader
from tsl.typing import Index

from src.data.spatiotemporal_dataset import SpatioTemporalDataset

StageOptions = Literal["fit", "validate", "test", "predict"]


class ConcatDataModule(LightningDataModule):
    """"""

    def __init__(
        self,
        datasets: list[SpatioTemporalDataset],
        force_batch: bool = True,
        splitter=None,
        scalers: Mapping | None = None,
        mask_scaling: bool = True,
        batch_size: int = 32,
        workers: int = 0,
        pin_memory: bool = False,
    ):
        super(ConcatDataModule, self).__init__()
        self.torch_dataset = ConcatDataset(datasets)
        # splitting
        self.splitter = splitter
        self.trainset = self.valset = self.testset = None
        # scaling
        if scalers is None:
            self.scalers = dict()
        else:
            self.scalers = scalers
        self.mask_scaling = mask_scaling
        # data loaders
        self.batch_size = batch_size
        self.force_batch = force_batch
        self.workers = workers
        self.pin_memory = pin_memory

    def __getattr__(self, item):
        ds = self.__dict__.get("torch_dataset")
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def n_nodes(self):
        tot = 0
        for ds in self.torch_dataset.datasets:
            tot += ds.n_nodes
        return tot

    def __repr__(self):
        return (
            "{}(train_len={}, val_len={}, test_len={}, "
            "scalers=[{}], batch_size={})".format(
                self.__class__.__name__,
                self.train_len,
                self.val_len,
                self.test_len,
                ", ".join(self.scalers.keys()),
                self.batch_size,
            )
        )

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def testset(self):
        return self._testset

    @trainset.setter
    def trainset(self, value):
        self._add_set("train", value)

    @valset.setter
    def valset(self, value):
        self._add_set("val", value)

    @testset.setter
    def testset(self, value):
        self._add_set("test", value)

    @property
    def train_len(self):
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        return len(self.testset) if self.testset is not None else None

    def _add_set(self, split_type, _set):
        assert split_type in ["train", "val", "test"]
        split_type = "_" + split_type
        name = split_type + "set"
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        else:
            indices = _set
            assert isinstance(indices, Index.__args__), (
                f"type {type(indices)} of `{name}` is not a valid type. "
                "It must be a dataset or a sequence of indices."
            )
            _set = Subset(self.torch_dataset, indices)
            setattr(self, name, _set)

    def _fit_scaler(self, key, scaler, stage):

        # set scalers
        if stage == "predict":
            logger.info(f"Set scaler for {key}: {scaler}")
        else:  # fit scalers before training
            data = []
            mask = []
            for d_id, dataset in enumerate(self.torch_dataset.datasets):
                data_ = getattr(dataset, key)
                # get only training slice
                train_slice = dataset.expand_indices(
                    indices=self.splitter.splitters[d_id].train_idxs, merge=True
                )
                if "t" in dataset.patterns[key]:
                    data_ = data_[train_slice]
                ndims = len(data_.shape)
                data_ = rearrange(
                    data_, f'... f -> (...) {" ".join(["1"] * (ndims - 2))} f'
                )

                mask_ = None
                if key == "target" and self.mask_scaling:
                    if dataset.mask is not None:
                        mask_ = dataset.get_mask()[train_slice]
                        ndims = len(mask_.shape)
                        mask_ = rearrange(
                            mask_, f'... f -> (...) {" ".join(["1"] * (ndims - 2))} f'
                        )
                        mask_.append(mask_)
                data.append(data_)

            data = torch.cat(data, 0)
            mask = torch.cat(mask, 0) if len(mask) else None

            scaler = scaler.fit(data, mask=mask, keepdims=True)
            logger.info(f"Fit and set scaler for {key}: {scaler}")

    def setup(self, stage: StageOptions = None):
        # splitting
        if self.splitter is not None:
            self.splitter.split(self.torch_dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs

        for (
            key,
            scaler,
        ) in self.scalers.items():
            self._fit_scaler(key, scaler, stage)
            for dataset in self.torch_dataset.datasets:
                dataset.add_scaler(key, scaler)

    def get_dataset_dataloader(
        self,
        idx: int,
        split: Literal["train", "val", "test"] = None,
        shuffle: bool = False,
        batch_size: int | None = None,
    ):
        dataset = self.torch_dataset.datasets[idx]
        if split is not None:
            assert split in ["train", "val", "test"]
            splitter = self.splitter.splitters[idx]
            dataset = Subset(dataset, splitter.indices[split])
        pin_memory = self.pin_memory if split == "train" else None
        return StaticGraphLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=split == "train",
            num_workers=self.workers,
            pin_memory=pin_memory,
        )

    def get_dataloader(
        self,
        split: Literal["train", "val", "test"] = None,
        idx: int = None,
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> DataLoader | None:
        if idx is not None:
            return self.get_dataset_dataloader(idx, split, shuffle, batch_size)
        if split is None:
            dataset = self.torch_dataset
        elif split in ["train", "val", "test"]:
            dataset = getattr(self, f"{split}set")
        else:
            raise ValueError(
                "Argument `split` must be one of " "'train', 'val', or 'test'."
            )
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == "train" else None
        return DisjointGraphLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=split == "train",
            force_batch=self.force_batch,
            num_workers=self.workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(
        self, idx: int = None, shuffle: bool = True, batch_size: int | None = None
    ) -> DataLoader | None:
        """"""
        return self.get_dataloader("train", idx, shuffle, batch_size)

    def val_dataloader(
        self, idx: int = None, shuffle: bool = False, batch_size: int | None = None
    ) -> DataLoader | None:
        """"""
        return self.get_dataloader("val", idx, shuffle, batch_size)

    def test_dataloader(
        self, idx: int = None, shuffle: bool = False, batch_size: int | None = None
    ) -> DataLoader | None:
        """"""
        return self.get_dataloader("test", idx, shuffle, batch_size)
