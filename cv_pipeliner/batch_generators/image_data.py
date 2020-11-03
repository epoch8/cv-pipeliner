from typing import List

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.batch_generator import BatchGenerator


class BatchGeneratorImageData(BatchGenerator):
    def __init__(self,
                 data: List[ImageData],
                 batch_size: int,
                 use_not_caught_elements_as_last_batch: bool):
        assert all(isinstance(d, ImageData) for d in data)
        super().__init__(data, batch_size, use_not_caught_elements_as_last_batch)

    def __getitem__(self, index) -> List[ImageData]:
        batch = super().__getitem__(index)
        for image_data in batch:
            if image_data.image is None:
                image_data.open_image(inplace=True)
        return batch