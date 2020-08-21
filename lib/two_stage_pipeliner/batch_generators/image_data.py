from typing import List

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.batch_generator import BatchGenerator


class BatchGeneratorImageData(BatchGenerator):
    def __init__(self,
                 data: List[ImageData],
                 batch_size: int):
        assert all(isinstance(d, ImageData) for d in data)
        BatchGenerator.__init__(self, data, batch_size)

    def __getitem__(self, index) -> List[ImageData]:
        batch = super().__getitem__(index)
        for image_data in batch:
            if image_data.image is None:
                image_data.open_image(inplace=True)
        return batch
