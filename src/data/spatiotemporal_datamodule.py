import tsl
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from tsl.data import SpatioTemporalDataModule as _SpatioTemporalDataModule_
from tsl.data.datamodule.spatiotemporal_datamodule import StageOptions
from tsl.typing import Index


class SpatioTemporalDataModule(_SpatioTemporalDataModule_):

    @property
    def split_dim(self) -> str:
        return getattr(self.splitter, "split_dim", "t")

    def _add_set(self, split_type, _set):
        assert split_type in ["train", "val", "test"]
        split_type = "_" + split_type
        name = split_type + "set"
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        elif (
            (isinstance(_set, ndarray) and _set.size == 0)
            or (isinstance(_set, Tensor) and _set.numel() == 0)
            or isinstance(_set, (list, tuple))
            and len(_set) == 0
        ):
            setattr(self, name, None)
        else:
            indices = _set
            assert isinstance(indices, Index.__args__), (
                f"type {type(indices)} of `{name}` is not a valid type. "
                "It must be a dataset or a sequence of indices."
            )
            if self.split_dim == "t":
                _set = self.torch_dataset.reduce(time_index=indices)
            elif self.split_dim == "n":
                _set = self.torch_dataset.reduce(node_index=indices)
            else:
                raise ValueError(f"Unable to split along dimension: {self.split_dim}")
            setattr(self, name, _set)

    def setup(self, stage: StageOptions = None):
        # splitting
        if self.splitter is not None:
            self.splitter.split(self.torch_dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs
        if self.split_dim == "n":
            for name in ["train", "val", "test"]:
                _set = getattr(self, name + "set")
                assert (
                    _set.target[(~_set.eval_mask) & _set.mask]
                    == self.trainset.target[self.trainset.mask]
                ).all()

        # set scalers
        for key, scaler in self.scalers.items():
            if key not in self.torch_dataset:
                raise RuntimeError(
                    "Cannot find a tensor to scale matching " f"key '{key}'."
                )
            if stage == "predict":
                tsl.logger.info(f"Set scaler for {key}: {scaler}")
            else:  # fit scalers before training
                data = getattr(
                    self.trainset if self.trainset is not None else self.torch_dataset,
                    key,
                )

                mask = None
                if key == "target" and self.mask_scaling:
                    d = (
                        self.trainset
                        if self.trainset is not None
                        else self.torch_dataset
                    )
                    if d.has_mask:
                        mask = d.get_mask()
                    if hasattr(d, "eval_mask"):
                        if mask is not None:
                            mask = mask & (~d.eval_mask)
                        else:
                            mask = ~d.eval_mask

                scaler = scaler.fit(data, mask=mask, keepdims=True)
                tsl.logger.info(f"Fit and set scaler for {key}: {scaler}")
            self.torch_dataset.add_scaler(key, scaler)
            for set_name in ["train", "val", "test"]:
                _set = getattr(self, set_name + "set")
                if _set is not None:
                    _set.add_scaler(key, scaler)
