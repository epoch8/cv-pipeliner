import io
from pathlib import Path
from typing import Dict, List, Union

import fsspec

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.utils.images_datas import combine_mask_polygons_to_one_polygon
from cv_pipeliner.utils.imagesize import get_image_size


class YOLODataConverter(DataConverter):
    def __init__(self, class_names: List[str]):
        super().__init__()
        assert len(set(class_names)) == len(class_names), "There are duplicates in 'class_names'. Remove them."
        self.class_names = class_names
        self.class_name_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class_name = {idx: class_name for idx, class_name in enumerate(self.class_names)}

    def get_annot_from_image_data(self, image_data: ImageData) -> List[str]:
        image_data = self.filter_image_data(image_data)
        width, height = image_data.get_image_size()
        txt_results = []
        for bbox_data in image_data.bboxes_data:
            w = bbox_data.xmax - bbox_data.xmin
            h = bbox_data.ymax - bbox_data.ymin
            xcenter = bbox_data.xmin + w / 2
            ycenter = bbox_data.ymin + h / 2
            xcenter, w = round(xcenter / width, 6), round(w / width, 6)
            ycenter, h = round(ycenter / height, 6), round(h / height, 6)
            idx = self.class_name_to_idx[bbox_data.label]
            txt_results.append(f"{idx} {xcenter} {ycenter} {w} {h}")
        return txt_results

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile, List[str]],
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, "r", encoding="utf8") as f:
                annots = f.read()
        elif isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annots = f.read()
        elif isinstance(annot, io.IOBase):
            annots = annot.read()
            if isinstance(annots, bytes):
                annots = annots.decode()
        elif isinstance(annot, List):
            annots = "\n".join(annot)

        width, height = get_image_size(image_path)
        bboxes_data = []
        for line in annots.strip().split("\n"):
            if line == "":
                continue
            idx, xcenter, ycenter, w, h = line.split(" ")
            label = self.idx_to_class_name[int(idx)]
            xcenter, ycenter, w, h = float(xcenter), float(ycenter), float(w), float(h)
            xcenter, w = xcenter * width, w * width
            ycenter, h = ycenter * height, h * height
            bboxes_data.append(
                BboxData(
                    xmin=xcenter - w / 2,
                    ymin=ycenter - h / 2,
                    xmax=xcenter + w / 2,
                    ymax=ycenter + h / 2,
                    label=label,
                )
            )

        return ImageData(image_path=image_path, bboxes_data=bboxes_data)


class YOLOMasksDataConverter(DataConverter):
    """
    Converter ImageData to YOLO Keypoints and YOLO Keypoints to ImageData
    """

    def __init__(self, class_names: List[str]):
        super().__init__()
        assert len(set(class_names)) == len(class_names), "There are duplicates in 'class_names'. Remove them."
        self.class_names = class_names
        self.class_name_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class_name = {idx: class_name for idx, class_name in enumerate(self.class_names)}

    def get_annot_from_image_data(self, image_data: ImageData) -> List[str]:
        image_data = self.filter_image_data(image_data)
        width, height = image_data.get_image_size()
        # txt_coors_results = []
        txt_keypoints_results = []
        for idx, bbox_data in enumerate(image_data.bboxes_data):
            idx = self.class_name_to_idx[bbox_data.label]
            mask = combine_mask_polygons_to_one_polygon(bbox_data.mask)
            box_keypoins = f"{idx}"
            for x, y in mask:
                box_keypoins += f" {round(x/width, 5)} {round(y/height, 5)}"
            txt_keypoints_results.append(box_keypoins)

        return txt_keypoints_results  # txt_coors_results,

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile, List[str]],
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, "r", encoding="utf8") as f:
                annots = f.read()
        elif isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annots = f.read()
        elif isinstance(annot, io.IOBase):
            annots = annot.read()
            if isinstance(annots, bytes):
                annots = annots.decode()
        elif isinstance(annot, List):
            annots = "\n".join(annot)

        width, height = get_image_size(image_path)
        bboxes_data = []
        for line in annots.strip().split("\n"):
            if line == "":
                continue

            elements = line.split()
            idx = elements[0]
            label = self.idx_to_class_name[int(idx)]
            points = elements[1:]
            scaled_points = [
                (float(points[i]) * width if i % 2 == 0 else float(points[i]) * height) for i in range(len(points))
            ]
            # Определяем минимальные и максимальные координаты для бокса
            xs, ys = scaled_points[0::2], scaled_points[1::2]  # x и y координаты точек

            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

            mask = [[scaled_points[i], scaled_points[i + 1]] for i in range(0, len(scaled_points), 2)]

            bboxes_data.append(
                BboxData(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    mask=[mask],
                    label=label,
                )
            )

        return ImageData(image_path=image_path, bboxes_data=bboxes_data)
