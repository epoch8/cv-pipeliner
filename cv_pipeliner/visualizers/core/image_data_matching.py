from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.utils.images import concat_images
from cv_pipeliner.visualizers.core.image_data import visualize_image_data


def get_true_and_pred_images_data_with_visualized_labels(
    image_data_matching: ImageDataMatching, error_type: Literal["detection", "pipeline"], label: str = None
) -> ImageData:
    """
    Create true and pred ImageData with changed label for visualization

    For detection error_type, the label will be one of ["TP", "FP", "FN"]

    For pipeline error_type, the label will be in format "label {matching_error_type}"
    (where matching_error_type is one of ["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"])
    """
    for tag, tag_image_data in [
        ("true", image_data_matching.true_image_data),
        ("pred", image_data_matching.pred_image_data),
    ]:
        tag_bboxes_data_with_visualized_label = []
        for tag_bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if tag == "true":
                tag_bbox_data = tag_bbox_data_matching.true_bbox_data
            elif tag == "pred":
                tag_bbox_data = tag_bbox_data_matching.pred_bbox_data

            if tag_bbox_data is None:
                continue

            if error_type == "detection":
                label_caption = f"[{tag_bbox_data_matching.get_detection_error_type(label=label)}]"
            elif error_type == "pipeline":
                pipeline_error_type = tag_bbox_data_matching.get_pipeline_error_type(label=label)
                if label is None:
                    label_caption = f"{tag_bbox_data.label} [{pipeline_error_type}]"
                else:
                    label_caption = f"{tag_bbox_data.label} [{pipeline_error_type}, cls. '{label}']"

            tag_bbox_data_with_visualized_label = BboxData(
                image_path=tag_bbox_data.image_path,
                cropped_image=tag_bbox_data.cropped_image,
                xmin=tag_bbox_data.xmin,
                ymin=tag_bbox_data.ymin,
                xmax=tag_bbox_data.xmax,
                ymax=tag_bbox_data.ymax,
                detection_score=tag_bbox_data.detection_score,
                label=label_caption,
                classification_score=tag_bbox_data.classification_score,
            )
            tag_bboxes_data_with_visualized_label.append(tag_bbox_data_with_visualized_label)

        tag_image_data_with_visualized_labels = ImageData(
            image_path=tag_image_data.image_path,
            image=tag_image_data.image,
            bboxes_data=tag_bboxes_data_with_visualized_label,
        )
        if tag == "true":
            true_image_data_with_visualized_labels = tag_image_data_with_visualized_labels
        elif tag == "pred":
            pred_image_data_with_visualized_labels = tag_image_data_with_visualized_labels

    return true_image_data_with_visualized_labels, pred_image_data_with_visualized_labels


error_type = Literal["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"]


def visualize_image_data_matching_side_by_side(
    image_data_matching: ImageDataMatching,
    error_type: Literal["detection", "pipeline"],
    true_include_labels: bool = False,
    pred_include_labels: bool = False,
    true_include_mask: bool = False,
    pred_include_mask: bool = False,
    true_score_type: Literal["detection", "classification"] = None,
    pred_score_type: Literal["detection", "classification"] = None,
    true_filter_by_error_types: List[error_type] = ["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"],
    pred_filter_by_error_types: List[error_type] = ["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"],
    known_labels: Optional[List[str]] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None,
    keypoints_radius: int = 5,
    include_additional_bboxes_data: bool = False,
    additional_bboxes_data_depth: Optional[int] = None,
    fontsize: int = 24,
    thickness: int = 4,
    mask_alpha: float = 0.5,
    thumbnail_size: Optional[Union[int, Tuple[int, int]]] = None,
    label: str = None,
    return_as_pil_image: bool = False,
    exif_transpose: bool = False,
) -> Union[np.ndarray, Image.Image]:
    (
        true_image_data_with_visualized_labels,
        pred_image_data_with_visualized_labels,
    ) = get_true_and_pred_images_data_with_visualized_labels(
        image_data_matching=image_data_matching, error_type=error_type, label=label
    )

    true_visualized_labels = [bbox_data.label for bbox_data in true_image_data_with_visualized_labels.bboxes_data]
    pred_visualized_labels = [bbox_data.label for bbox_data in pred_image_data_with_visualized_labels.bboxes_data]
    true_filter_by_labels = [
        label
        for label in true_visualized_labels
        if any(f"[{matching_error_type}]" in label for matching_error_type in true_filter_by_error_types)
    ]
    pred_filter_by_labels = [
        label
        for label in pred_visualized_labels
        if any(f"[{matching_error_type}]" in label for matching_error_type in pred_filter_by_error_types)
    ]
    true_image = visualize_image_data(
        image_data=true_image_data_with_visualized_labels,
        include_labels=true_include_labels,
        score_type=true_score_type,
        filter_by_labels=true_filter_by_labels,
        known_labels=known_labels,
        include_mask=true_include_mask,
        mask_alpha=mask_alpha,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
        keypoints_radius=keypoints_radius,
        include_additional_bboxes_data=include_additional_bboxes_data,
        additional_bboxes_data_depth=additional_bboxes_data_depth,
        fontsize=fontsize,
        thickness=thickness,
        thumbnail_size=thumbnail_size,
        return_as_pil_image=False,
        exif_transpose=exif_transpose,
    )
    pred_image = visualize_image_data(
        image_data=pred_image_data_with_visualized_labels,
        include_labels=pred_include_labels,
        score_type=pred_score_type,
        filter_by_labels=pred_filter_by_labels,
        known_labels=known_labels,
        include_mask=pred_include_mask,
        mask_alpha=mask_alpha,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
        keypoints_radius=keypoints_radius,
        include_additional_bboxes_data=include_additional_bboxes_data,
        additional_bboxes_data_depth=additional_bboxes_data_depth,
        fontsize=fontsize,
        thickness=thickness,
        thumbnail_size=thumbnail_size,
        return_as_pil_image=False,
    )
    image = concat_images(true_image, pred_image, how="horizontally")
    if return_as_pil_image:
        return Image.fromarray(image)
    return image
