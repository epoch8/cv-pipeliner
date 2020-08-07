from typing import List, Union
import numpy as np

from two_stage_pipeliner.core.data import BboxData, ImageData


class BatchGenerator:
    def __init__(self,
                 data: List[Union[ImageData, BboxData]],
                 batch_size: int):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        super(BatchGenerator, self).__init__()

    def __len__(self) -> int:
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index) -> List[Union[ImageData, BboxData]]:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.data[indexes]
