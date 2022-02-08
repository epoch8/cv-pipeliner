import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        yield d


def assert_images_datas_equal(image_data1, image_data2):
    bboxes_data1 = sorted(image_data1.bboxes_data, key=lambda x: x.coords)
    bboxes_data2 = sorted(image_data2.bboxes_data, key=lambda x: x.coords)
    for bbox_data1, bbox_data2 in zip(bboxes_data1, bboxes_data2):
        assert bbox_data1.coords == bbox_data2.coords
        assert bbox_data1.label == bbox_data2.label
