import abc
import copy

from typing import List
import numpy as np

from two_stage_pipeliner.core.data import BboxData, ImageData


class BatchGenerator(abc.ABC):
    def __init__(self,
                 data: List,
                 batch_size: int):
        assert int(np.ceil(len(data) / batch_size)) != 0
        self.data = np.array(data)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        super(BatchGenerator, self).__init__()

    def __len__(self) -> int:
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index) -> List:
        pass

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item


class BatchGeneratorImageData(BatchGenerator):
    def __init__(self,
                 data: List[ImageData],
                 batch_size: int):
        assert all(isinstance(d, ImageData) for d in data)
        BatchGenerator.__init__(self, data, batch_size)

    def __getitem__(self, index) -> List[ImageData]:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch = copy.deepcopy(self.data[indexes])
        for image_data in batch:
            if image_data.image is None:
                image_data.open_image(inplace=True)
        return batch


class BatchGeneratorBboxData(BatchGenerator):
    def __init__(self,
                 data: List[List[BboxData]],
                 batch_size: int):
        assert all(isinstance(d, list) for d in data)
        assert all(isinstance(item, BboxData) for d in data for item in d)
        BatchGenerator.__init__(self, data, batch_size)

    def __getitem__(self, index) -> List[List[BboxData]]:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch = copy.deepcopy(self.data[indexes])
        for bboxes_data in batch:
            for bbox_data in bboxes_data:
                if bbox_data.cropped_image is None:
                    bbox_data.open_cropped_image(inplace=True)
        return batch
