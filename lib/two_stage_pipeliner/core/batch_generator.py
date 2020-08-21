import abc
import copy

from typing import List, Any
import numpy as np


class BatchGenerator(abc.ABC):
    def __init__(self,
                 data: List[Any],
                 batch_size: int):
        self.data = np.array(data, dtype=object)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        super(BatchGenerator, self).__init__()

    def __len__(self) -> int:
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index) -> List:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch = copy.deepcopy(self.data[indexes])
        return batch

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
