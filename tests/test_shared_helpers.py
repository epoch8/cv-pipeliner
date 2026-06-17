import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.batch_utils import (
    call_progress_callback,
    ensure_bbox_data_generator,
    ensure_image_data_generator,
    split_chunks,
)


def test_split_chunks():
    assert split_chunks([1, 2, 3, 4, 5], [2, 0, 3]) == [[1, 2], [], [3, 4, 5]]


def test_progress_callback_accepts_none():
    call_progress_callback(None, 10)

    progress = []
    call_progress_callback(progress.append, 3)

    assert progress == [3]


def test_ensure_image_data_generator_from_list():
    generator = ensure_image_data_generator([ImageData(image=np.zeros((2, 2, 3), dtype=np.uint8))], batch_size_default=1)

    assert len(generator.data) == 1


def test_ensure_bbox_data_generator_from_flat_list():
    generator = ensure_bbox_data_generator(
        [BboxData(image=np.zeros((2, 2, 3), dtype=np.uint8), xmin=0, ymin=0, xmax=1, ymax=1)],
        batch_size_default=1,
    )

    assert generator.shapes.tolist() == [1]
