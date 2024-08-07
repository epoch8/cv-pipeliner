from typing import List

import numpy as np

from cv_pipeliner.core.batch_generator import BatchGenerator
from cv_pipeliner.core.data import BboxData


class BatchGeneratorBboxData(BatchGenerator):
    def __init__(
        self,
        data: List[List[BboxData]],
        batch_size: int,
        use_not_caught_elements_as_last_batch: bool = True,
        open_cropped_images: bool = True,
    ):
        assert all(isinstance(d, list) or isinstance(d, np.ndarray) for d in data)
        assert all(isinstance(item, BboxData) for d in data for item in d)
        self._shapes = np.array([len(subdata) for subdata in data])
        data = [item for sublist in data for item in sublist]
        super().__init__(data, batch_size, use_not_caught_elements_as_last_batch)
        self.open_cropped_images = open_cropped_images

    def __getitem__(self, index: int) -> List[BboxData]:
        batch = super().__getitem__(index)
        unique_image_paths, unique_image_idxs = np.unique(
            [bbox_data.image_path for bbox_data in batch if bbox_data.image_path is not None], return_index=True
        )
        unique_image_idx_to_image = {
            unique_image_path: batch[unique_image_idx].open_image()
            for unique_image_path, unique_image_idx in zip(unique_image_paths, unique_image_idxs)
        }
        for bbox_data in batch:
            source_image = unique_image_idx_to_image[bbox_data.image_path] if bbox_data.image_path is not None else None
            if self.open_cropped_images and bbox_data.cropped_image is None:
                bbox_data.open_cropped_image(source_image=source_image, inplace=True)
        return batch

    @property
    def shapes(self):
        return self._shapes

    def __iter__(self):
        for bboxes_data in (self[i] for i in range(len(self))):
            yield bboxes_data
            for bbox_data in bboxes_data:
                bbox_data.image = None
                bbox_data.cropped_image = None
            del bboxes_data
