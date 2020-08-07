from typing import List
import numpy as np

from two_stage_pipeliner.core.data import BboxData, ImageData


class BatchGenerator:
    def __init__(self,
                 data: List,
                 batch_size: int):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        super(BatchGenerator, self).__init__()

    def __len__(self) -> int:
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index) -> List:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.data[indexes]


class BatchGeneratorImageData(BatchGenerator):
    def __init__(self,
                 data: List[ImageData],
                 batch_size: int):
        assert all(isinstance(d, ImageData) for d in data)
        super(BatchGenerator, self).__init__(data, batch_size)


class BatchGeneratorBboxData(BatchGenerator):
    def __init__(self,
                 data: List[List[BboxData]],
                 batch_size: int):
        assert all(isinstance(d, list) for d in data)
        assert all(isinstance(item, BboxData) for d in data for item in d)
        super(BatchGenerator, self).__init__(data, batch_size)
