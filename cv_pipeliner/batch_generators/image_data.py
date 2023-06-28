from typing import List
from joblib import Parallel, delayed

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.batch_generator import BatchGenerator


class BatchGeneratorImageData(BatchGenerator):
    def __init__(
        self,
        data: List[ImageData],
        batch_size: int,
        use_not_caught_elements_as_last_batch: bool = True,
        max_workers: int = 64,
    ):
        self.max_workers = max_workers
        assert all(isinstance(d, ImageData) for d in data)
        super().__init__(data, batch_size, use_not_caught_elements_as_last_batch)

    def __getitem__(self, index) -> List[ImageData]:
        batch = super().__getitem__(index)
        Parallel(n_jobs=self.max_workers, prefer="threads")(
            delayed(lambda x: x.open_image(inplace=True))(image_data) for image_data in batch
        )
        return batch
