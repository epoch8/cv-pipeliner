import json
from pathlib import Path

import fsspec
import numpy as np
import pytest

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.data_converters.coco import COCODataConverter
from cv_pipeliner.data_converters.json import JSONDataConverter
from cv_pipeliner.data_converters.supervisely import SuperviselyDataConverter
from cv_pipeliner.data_converters.yolo import YOLODataConverter, YOLOMasksDataConverter

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


class ConcreteDataConverter(DataConverter):
    def get_image_data_from_annot(self, image_path, annot):
        return ImageData(image_path=image_path)


@pytest.mark.parametrize(
    "data_convertor_cls",
    [YOLODataConverter, JSONDataConverter, SuperviselyDataConverter],
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

    assert len(images_data) == len(images_data_from_annots)
    for image_data1, image_data2 in zip(images_data, images_data_from_annots):
        assert_images_datas_equal(image_data1, image_data2)


def test_data_converter_filter_removes_duplicate_and_invalid_bboxes():
    image_data = ImageData(
        image_path=coco_imgs / "000000000009.jpg",
        bboxes_data=[
            BboxData(xmin=1, ymin=2, xmax=5, ymax=6, label="valid"),
            BboxData(xmin=1, ymin=2, xmax=5, ymax=6, label="duplicate"),
            BboxData(xmin=5, ymin=2, xmax=5, ymax=6, label="zero-width"),
            BboxData(xmin=6, ymin=2, xmax=5, ymax=6, label="inverted"),
        ],
    )

    filtered = ConcreteDataConverter().filter_image_data(image_data)

    assert [bbox_data.label for bbox_data in filtered.bboxes_data] == ["valid"]


def test_yolo_converter_handles_empty_annotation(tmp_dir):
    image_path = tmp_dir / "image.png"
    annot_path = tmp_dir / "image.txt"
    from PIL import Image

    Image.new("RGB", (10, 8)).save(image_path)
    annot_path.write_text("")

    image_data = YOLODataConverter(class_names=["class-a"]).get_image_data_from_annot(image_path, annot_path)

    assert image_data.bboxes_data == []


def test_yolo_converter_rejects_unknown_class_on_export():
    converter = YOLODataConverter(class_names=["known"])
    image_data = ImageData(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        bboxes_data=[BboxData(xmin=1, ymin=1, xmax=5, ymax=5, label="unknown")],
    )

    with pytest.raises(KeyError):
        converter.get_annot_from_image_data(image_data)


def test_yolo_masks_converter_roundtrips_polygon(tmp_dir):
    image_path = tmp_dir / "image.png"
    from PIL import Image

    Image.new("RGB", (10, 10)).save(image_path)
    converter = YOLOMasksDataConverter(class_names=["object"])
    image_data = ImageData(
        image_path=image_path,
        bboxes_data=[
            BboxData(
                xmin=1,
                ymin=2,
                xmax=6,
                ymax=8,
                label="object",
                mask=[[(1, 2), (6, 2), (6, 8), (1, 8)]],
            )
        ],
    )

    annot = converter.get_annot_from_image_data(image_data)
    restored = converter.get_image_data_from_annot(image_path=image_path, annot=annot)

    assert len(restored.bboxes_data) == 1
    assert restored.bboxes_data[0].label == "object"
    assert restored.bboxes_data[0].coords == (1, 2, 6, 8)


def test_coco_converter_reads_bboxes_and_segmentation(tmp_dir):
    image_path = tmp_dir / "000000000009.jpg"
    annot = {
        "categories": [{"id": 51, "name": "bowl"}, {"id": 56, "name": "broccoli"}],
        "annotations": [
            {
                "id": 1038967,
                "image_id": 9,
                "category_id": 51,
                "bbox": [1.08, 187.69, 611.59, 285.84],
                "segmentation": [[500.49, 473.53, 599.73, 419.6, 612.67, 375.37]],
                "iscrowd": 0,
            },
            {
                "id": 1058555,
                "image_id": 9,
                "category_id": 56,
                "bbox": [249.6, 229.27, 316.24, 245.08],
                "segmentation": [[249.6, 348.99, 267.67, 311.72, 291.39, 294.78]],
                "iscrowd": 0,
            },
            {
                "id": 999,
                "image_id": 10,
                "category_id": 51,
                "bbox": [0, 0, 1, 1],
                "segmentation": [[0, 0, 1, 0, 1, 1]],
                "iscrowd": 0,
            },
            {
                "id": 1000,
                "image_id": 9,
                "category_id": 51,
                "bbox": [0, 0, 1, 1],
                "segmentation": {"counts": [], "size": [1, 1]},
                "iscrowd": 1,
            },
        ],
    }

    image_data = COCODataConverter().get_image_data_from_annot(image_path=image_path, annot=annot)

    assert image_data.image_path == image_path
    assert image_data.label == "bowl"
    assert image_data.additional_info == {"coco_image_id": 9}
    assert [bbox_data.label for bbox_data in image_data.bboxes_data] == ["bowl", "broccoli"]
    assert [bbox_data.coords for bbox_data in image_data.bboxes_data] == [(1, 188, 613, 474), (250, 229, 566, 474)]
    assert image_data.bboxes_data[0].mask[0].tolist() == [[500, 474], [600, 420], [613, 375]]
    assert image_data.bboxes_data[0].additional_info == {"coco_annotation_id": 1038967}


def test_coco_converter_reads_one_annotation_file_for_many_images(tmp_dir):
    annot_path = tmp_dir / "instances_train2017.json"
    annot = {
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 9,
                "category_id": 1,
                "bbox": [0, 0, 2, 3],
                "segmentation": [[0, 0, 2, 0, 2, 3]],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 25,
                "category_id": 1,
                "bbox": [1, 1, 4, 5],
                "segmentation": [[1, 1, 5, 1, 5, 6]],
                "iscrowd": 0,
            },
        ],
    }
    annot_path.write_text(json.dumps(annot))

    images_data_from_annots = COCODataConverter().get_images_data_from_annots(
        image_paths=[tmp_dir / "000000000009.jpg", tmp_dir / "000000000025.jpg"],
        annots=annot_path,
        n_jobs=1,
        disable_tqdm=True,
    )

    assert [len(image_data.bboxes_data) for image_data in images_data_from_annots] == [1, 1]
    assert [image_data.bboxes_data[0].coords for image_data in images_data_from_annots] == [(0, 0, 2, 3), (1, 1, 5, 6)]
