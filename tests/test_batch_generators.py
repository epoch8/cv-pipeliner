import numpy as np
import pytest

from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.batch_generator import BatchGenerator
from cv_pipeliner.core.data import BboxData, ImageData


def test_batch_generator_keeps_partial_final_batch_and_deepcopies_items():
    data = [{"value": 1}, {"value": 2}, {"value": 3}]
    generator = BatchGenerator(data, batch_size=2, use_not_caught_elements_as_last_batch=True)

    first_batch = generator[0]
    first_batch[0]["value"] = 100

    assert len(generator) == 2
    assert len(generator[1]) == 1
    assert data[0]["value"] == 1


def test_batch_generator_can_drop_partial_final_batch():
    generator = BatchGenerator([1, 2, 3], batch_size=2, use_not_caught_elements_as_last_batch=False)

    assert len(generator) == 1
    assert generator[0].tolist() == [1, 2]


def test_batch_generator_rejects_too_small_data_when_dropping_partial_batch():
    with pytest.raises(AssertionError):
        BatchGenerator([1], batch_size=2, use_not_caught_elements_as_last_batch=False)


def test_batch_generator_image_data_opens_images_and_cleans_iteration_batches(tmp_dir):
    image_path = tmp_dir / "image.png"
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    from PIL import Image

    Image.fromarray(image).save(image_path)
    image_data = ImageData(image_path=image_path)
    generator = BatchGeneratorImageData([image_data], batch_size=1, max_workers=1)

    batch = next(iter(generator))

    assert batch[0].image is not None
    assert batch[0].image.shape == (3, 4, 3)


def test_batch_generator_bbox_data_flattens_groups_preserves_shapes_and_opens_crops():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    grouped_bboxes = [
        [BboxData(image=image, xmin=0, ymin=0, xmax=4, ymax=4)],
        [
            BboxData(image=image, xmin=1, ymin=1, xmax=5, ymax=5),
            BboxData(image=image, xmin=2, ymin=2, xmax=6, ymax=7),
        ],
    ]
    generator = BatchGeneratorBboxData(grouped_bboxes, batch_size=2)

    first_batch = generator[0]

    assert generator.shapes.tolist() == [1, 2]
    assert len(generator.data) == 3
    assert first_batch[0].cropped_image.shape == (4, 4, 3)
