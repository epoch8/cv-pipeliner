import json
from pathlib import Path

import fsspec
import pytest

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.data_converters.json import JSONDataConverter
from cv_pipeliner.data_converters.supervisely import SuperviselyDataConverter
from cv_pipeliner.data_converters.yolo import YOLODataConverter

from .conftest import assert_images_datas_equal

test_data = Path(__file__).parent / "test_data"
coco_imgs = test_data / "coco"

images_data = [
    ImageData(
        image_path=coco_imgs / "000000000009.jpg",
        bboxes_data=[
            BboxData(xmin=1, ymin=187, xmax=612, ymax=473, label="bowl"),
            BboxData(xmin=311, ymin=4, xmax=631, ymax=232, label="bowl"),
            BboxData(xmin=249, ymin=229, xmax=565, ymax=474, label="broccoli"),
            BboxData(xmin=0, ymin=13, xmax=434, ymax=388, label="bowl"),
            BboxData(xmin=376, ymin=40, xmax=451, ymax=86, label="orange"),
            BboxData(xmin=465, ymin=38, xmax=523, ymax=85, label="orange"),
            BboxData(xmin=385, ymin=73, xmax=469, ymax=144, label="orange"),
            BboxData(xmin=364, ymin=2, xmax=458, ymax=73, label="orange"),
        ],
    ),
    ImageData(
        image_path=coco_imgs / "000000000025.jpg",
        bboxes_data=[
            BboxData(xmin=385, ymin=60, xmax=600, ymax=357, label="giraffe"),
            BboxData(xmin=53, ymin=356, xmax=185, ymax=411, label="giraffe"),
        ],
    ),
    ImageData(
        image_path=coco_imgs / "000000000030.jpg",
        bboxes_data=[
            BboxData(xmin=204, ymin=31, xmax=459, ymax=355, label="potted plant"),
            BboxData(xmin=237, ymin=155, xmax=403, ymax=351, label="vase"),
        ],
    ),
    ImageData(
        image_path=coco_imgs / "000000000034.jpg",
        bboxes_data=[BboxData(xmin=0, ymin=20, xmax=442, ymax=399, label="zebra")],
    ),
    ImageData(
        image_path=coco_imgs / "000000000036.jpg",
        bboxes_data=[
            BboxData(xmin=0, ymin=50, xmax=457, ymax=480, label="umbrella"),
            BboxData(xmin=167, ymin=162, xmax=478, ymax=628, label="person"),
        ],
    ),
]

coco_class_names = ["bowl", "broccoli", "giraffe", "orange", "person", "potted plant", "umbrella", "vase", "zebra"]


@pytest.mark.parametrize(
    "data_convertor_cls",
    [YOLODataConverter, BrickitDataConverter, JSONDataConverter, SuperviselyDataConverter],
)
def test_data_convertor(tmp_dir, data_convertor_cls):
    kwargs = {}
    if data_convertor_cls == YOLODataConverter:
        kwargs = {"class_names": coco_class_names}
    data_converter = data_convertor_cls(**kwargs)
    annots = data_converter.get_annot_from_images_data(images_data)

    extenstion = "json"
    images_paths = []
    annots_paths = []
    if data_convertor_cls == YOLODataConverter:
        annots = ["\n".join(annot) for annot in annots]
        extenstion = "txt"

    # Разметка формата N изображений --> N файлов разметки
    if data_convertor_cls != BrickitDataConverter:
        for image_data, annot in zip(images_data, annots):
            image_path = image_data.image_path
            annot_path = tmp_dir / f"{image_data.image_path.name}.{extenstion}"
            with fsspec.open(annot_path, "w") as out:
                if data_convertor_cls == YOLODataConverter:
                    out.write(annot)
                else:
                    json.dump(annot, out)
            images_paths.append(image_path)
            annots_paths.append(annot_path)

        images_data_from_annots = data_converter.get_images_data_from_annots(images_paths, annots_paths)

    # Разметка формата N изображений --> 1 файл разметки
    else:
        annots_path = tmp_dir / f"{coco_imgs.name}-annotations.json"
        with fsspec.open(annots_path, "w") as out:
            json.dump(annots, out)
        images_data_from_annots = data_converter.get_images_data_from_annots(coco_imgs, annots_path)

    assert len(images_data) == len(images_data_from_annots)
    for image_data1, image_data2 in zip(images_data, images_data_from_annots):
        assert_images_datas_equal(image_data1, image_data2)
