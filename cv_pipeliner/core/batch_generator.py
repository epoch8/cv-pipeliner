import abc
import copy

from typing import List, Any
import numpy as np


class BatchGenerator(abc.ABC):
    def __init__(self, data: List[Any], batch_size: int, use_not_caught_elements_as_last_batch: bool):
        """
        Base generator class for data.

        Argument use_not_caught_elements_as_last_batch is used
        when len(data) is not divisible by batch_size, but we need to get
        all data by using the generator (e.g. for inference).
        For training use_not_caught_elements_as_last_batch should be set as False.
        """
        self.data = np.array(data, dtype=object)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        if use_not_caught_elements_as_last_batch:
            self._len = int(np.ceil(len(self.data) / self.batch_size))
        else:
            assert len(self.data) // self.batch_size != 0
            self._len = len(self.data) // self.batch_size

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index) -> List:
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch = copy.deepcopy(self.data[indexes])
        return batch

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
