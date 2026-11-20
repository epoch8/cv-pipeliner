import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import fsspec
from joblib import Parallel, delayed
from tqdm import tqdm

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.core.data_converter import DataConverter


class COCODataConverter(DataConverter):
    """COCO / Roboflow-style annotation converter.

    By default ``category_id`` values are **1-based** (Roboflow / RF-DETR).
    Pass ``category_id_start=0`` for 0-based COCO exports.
    """

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        *,
        category_id_start: int = 1,
        include_segmentation: bool = True,
    ):
        super().__init__()
        self.class_names = list(class_names) if class_names is not None else None
        self.category_id_start = int(category_id_start)
        self.include_segmentation = bool(include_segmentation)
        if self.class_names is not None:
            assert len(set(self.class_names)) == len(self.class_names), (
                "There are duplicates in 'class_names'. Remove them."
            )
            self.class_name_to_id = {
                name: self.category_id_start + idx for idx, name in enumerate(self.class_names)
            }
        else:
            self.class_name_to_id = {}

    def _categories(self) -> List[Dict]:
        if self.class_names is None:
            raise ValueError("COCODataConverter requires class_names for export")
        return [
            {"id": self.category_id_start + idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(self.class_names)
        ]

    @staticmethod
    def _load_annot(annot: Union[Path, str, Dict, fsspec.core.OpenFile]) -> Dict:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, "r", encoding="utf8") as f:
                return json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                return json.load(f)
        return annot

    @staticmethod
    def _polygon_to_points(polygon: List[float]) -> List[List[int]]:
        return [[round(polygon[i]), round(polygon[i + 1])] for i in range(0, len(polygon), 2)]

    @staticmethod
    def _mask_to_coco_segmentation(mask) -> List[List[float]]:
        """Convert polygon-list masks to COCO ``segmentation`` rings.

        Accepts ``None``, an empty sequence, or a sequence of polygons (each a
        sequence of ``[x, y]`` points). Binary / ndarray masks are not converted
        here — callers fall back to a bbox rectangle when this returns ``[]``.
        """
        if mask is None:
            return []
        # Numpy arrays are ambiguous under ``bool(mask)`` / ``if not polygon``.
        try:
            import numpy as np

            if isinstance(mask, np.ndarray):
                return []
        except ImportError:
            pass
        try:
            if len(mask) == 0:
                return []
        except TypeError:
            return []
        segs: List[List[float]] = []
        for polygon in mask:
            if polygon is None:
                continue
            try:
                if len(polygon) == 0:
                    continue
            except TypeError:
                continue
            flat: List[float] = []
            for point in polygon:
                if point is None:
                    continue
                try:
                    if len(point) < 2:
                        continue
                except TypeError:
                    continue
                flat.extend([float(point[0]), float(point[1])])
            if len(flat) >= 6:
                segs.append(flat)
        return segs

    def get_annot_from_image_data(
        self,
        image_data: ImageData,
        *,
        image_id: int = 1,
        file_name: Optional[str] = None,
        annotation_id_start: int = 1,
    ) -> Dict:
        """Export one image (+ its annotations) as COCO fragment fields.

        Returns ``{"image": {...}, "annotations": [...]}`` (no categories).
        """
        if self.class_names is None:
            raise ValueError("COCODataConverter requires class_names for export")
        image_data = self.filter_image_data(image_data)
        width, height = image_data.get_image_size()
        if file_name is None:
            if image_data.image_path is not None:
                file_name = Path(image_data.image_path).name
            else:
                file_name = f"{image_id}.jpg"
        image_entry = {
            "id": int(image_id),
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
        }
        annotations: List[Dict] = []
        ann_id = int(annotation_id_start)
        for bbox_data in image_data.bboxes_data or []:
            assert isinstance(bbox_data, BboxData)
            if bbox_data.label not in self.class_name_to_id:
                raise ValueError(
                    f"Unknown label {bbox_data.label!r}; expected one of {self.class_names}"
                )
            x1, y1 = float(bbox_data.xmin), float(bbox_data.ymin)
            x2, y2 = float(bbox_data.xmax), float(bbox_data.ymax)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if self.include_segmentation:
                segmentation = self._mask_to_coco_segmentation(bbox_data.mask)
                if not segmentation:
                    segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            else:
                segmentation = []
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": self.class_name_to_id[bbox_data.label],
                    "bbox": [x1, y1, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1
        return {"image": image_entry, "annotations": annotations}

    def get_annot_from_images_data(
        self,
        images_data: Sequence[ImageData],
        *,
        file_names: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Build a full COCO annotation dict from many ``ImageData`` rows."""
        if self.class_names is None:
            raise ValueError("COCODataConverter requires class_names for export")
        if file_names is not None and len(file_names) != len(images_data):
            raise ValueError("file_names length must match images_data")
        images: List[Dict] = []
        annotations: List[Dict] = []
        ann_id = 1
        for image_id, image_data in enumerate(images_data, start=1):
            file_name = file_names[image_id - 1] if file_names is not None else None
            fragment = self.get_annot_from_image_data(
                image_data,
                image_id=image_id,
                file_name=file_name,
                annotation_id_start=ann_id,
            )
            images.append(fragment["image"])
            annotations.extend(fragment["annotations"])
            ann_id += len(fragment["annotations"])
        return {
            "images": images,
            "annotations": annotations,
            "categories": self._categories(),
        }

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile],
    ) -> ImageData:
        image_path = Path(image_path)
        annot = self._load_annot(annot)
        category_by_id = {category["id"]: category["name"] for category in annot["categories"]}
        # Prefer matching by file_name; fall back to int(stem) for legacy callers.
        image_id = None
        for image_entry in annot.get("images", []):
            if image_entry.get("file_name") == image_path.name:
                image_id = image_entry["id"]
                break
        if image_id is None:
            try:
                image_id = int(image_path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot resolve COCO image_id for {image_path.name!r}"
                ) from exc
        bboxes_data = []
        for bbox_annot in annot["annotations"]:
            if bbox_annot["image_id"] != image_id:
                continue
            if bbox_annot.get("iscrowd", 0) != 0 or not isinstance(bbox_annot.get("segmentation"), list):
                continue

            xmin, ymin, width, height = bbox_annot["bbox"]
            bboxes_data.append(
                BboxData(
                    image_path=image_path,
                    xmin=round(xmin),
                    ymin=round(ymin),
                    xmax=round(xmin + width),
                    ymax=round(ymin + height),
                    label=category_by_id[bbox_annot["category_id"]],
                    mask=[
                        self._polygon_to_points(polygon)
                        for polygon in bbox_annot["segmentation"]
                        if len(polygon) >= 6
                    ],
                    additional_info={"coco_annotation_id": bbox_annot["id"]},
                )
            )

        return ImageData(
            image_path=image_path,
            label=bboxes_data[0].label if len(bboxes_data) > 0 else None,
            additional_info={"coco_image_id": image_id},
            bboxes_data=bboxes_data,
        )

    @DataConverter.assert_images_data
    def get_images_data_from_annots(
        self,
        image_paths: List[Union[str, Path]],
        annots: Union[Path, str, Dict, fsspec.core.OpenFile],
        n_jobs: int = 8,
        disable_tqdm: bool = False,
    ) -> List[ImageData]:
        annot = self._load_annot(annots)
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.get_image_data_from_annot)(image_path=image_path, annot=annot)
            for image_path in tqdm(image_paths, total=len(image_paths), disable=disable_tqdm)
        )
