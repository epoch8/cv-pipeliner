import copy

from typing import List, Tuple

from cv_pipeliner.core.data import ImageData, BboxData


def get_image_data_filtered_by_labels(
    image_data: ImageData,
    filter_by_labels: List[str] = None
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return image_data

    bboxes_data = [
        bbox_data for bbox_data in image_data.bboxes_data if bbox_data.label in filter_by_labels
    ]
    return ImageData(
        image_path=image_data.image_path,
        image=image_data.image,
        bboxes_data=bboxes_data
    )


def get_n_bboxes_data_filtered_by_labels(
    n_bboxes_data: List[List[BboxData]],
    filter_by_labels: List[str] = None
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return n_bboxes_data

    n_bboxes_data = [
        [bbox_data for bbox_data in bboxes_data if bbox_data.label in filter_by_labels]
        for bboxes_data in n_bboxes_data
    ]
    return n_bboxes_data


def cut_images_data_by_bboxes(
    images_data: List[ImageData],
    bboxes: List[Tuple[int, int, int, int]] = None
) -> List[ImageData]:
    if bboxes is None:
        return images_data

    images_data = copy.deepcopy(images_data)
    for image_data, bbox in zip(images_data, bboxes):
        xmin, ymin, xmax, ymax = bbox
        image_data.bboxes_data = [
            bbox_data
            for bbox_data in image_data.bboxes_data
            if (
                bbox_data.xmin - xmin >= 0 and bbox_data.ymin - ymin >= 0 and
                bbox_data.xmax - xmin <= xmax and bbox_data.ymax - ymin <= ymax
            )
        ]
    return images_data
