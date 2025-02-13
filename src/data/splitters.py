import numpy as np
from torch.utils.data import ConcatDataset
from tsl.data import Splitter, TemporalSplitter


class TemporalConcatSplitter(Splitter):
    r"""Split the data sequentially with specified lengths."""

    def __init__(self, val_len=None, test_len=None):
        super().__init__()
        self._val_len = val_len
        self._test_len = test_len
        self.splitters = list()

    def fit(self, dataset: ConcatDataset):
        offset = 0
        idxs = []
        for ds in dataset.datasets:
            splitter = TemporalSplitter(val_len=self._val_len, test_len=self._test_len)
            splitter.fit(ds)
            idxs.append(
                (
                    splitter.train_idxs + offset,
                    splitter.val_idxs + offset,
                    splitter.test_idxs + offset,
                )
            )
            offset += len(ds)
            self.splitters.append(splitter)
        idxs = zip(*idxs)
        idxs = [np.concatenate(idx, 0) for idx in idxs]
        self.set_indices(*idxs)
