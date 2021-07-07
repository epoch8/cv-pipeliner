from typing import List
from joblib.parallel import delayed

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.batch_generator import BatchGenerator
from joblib import Parallel


class BatchGeneratorImageData(BatchGenerator):
    def __init__(
        self,
        data: List[ImageData],
        batch_size: int,
        use_not_caught_elements_as_last_batch: bool = True,
        workers: int = 1
    ):
        assert all(isinstance(d, ImageData) for d in data)
        super().__init__(data, batch_size, use_not_caught_elements_as_last_batch)
        self.workers = workers

    def __getitem__(self, index) -> List[ImageData]:
        batch = super().__getitem__(index)

        images = Parallel(n_jobs=self.workers)(
            delayed(lambda image_data: image_data.open_image())(image_data)
            for image_data in batch
        )
        for image_data, image in zip(batch, images):
            image_data.image = image

        return batch
