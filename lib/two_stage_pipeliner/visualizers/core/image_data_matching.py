from typing import List, Literal, Callable

import numpy as np

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching
from two_stage_pipeliner.visualizers.core.image_data import visualize_images_data_side_by_side


def get_true_and_pred_images_data_with_visualized_labels(
    image_data_matching: ImageDataMatching,
    error_type: Literal['detection', 'pipeline']
) -> ImageData:
    """
    Create true and pred ImageData with changed label for visualization

    For detection error_type, the label will be one of ["TP", "FP", "FN"]

    For pipeline error_type, the label will be in format "label {matching_error_type}"
    (where matching_error_type is one of ["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"])
    """
    for tag, tag_image_data in [('true', image_data_matching.true_image_data),
                                ('pred', image_data_matching.pred_image_data)]:
        tag_bboxes_data_with_visualized_label = []
        for tag_bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if tag == 'true':
                tag_bbox_data = tag_bbox_data_matching.true_bbox_data
            elif tag == 'pred':
                tag_bbox_data = tag_bbox_data_matching.pred_bbox_data

            if tag_bbox_data is None:
                continue

            if error_type == 'detection':
                label = f"[{tag_bbox_data_matching.get_detection_error_type()}]"
            elif error_type == 'pipeline':
                pipeline_error_type = tag_bbox_data_matching.get_pipeline_error_type()
                label = f"{tag_bbox_data.label} [{pipeline_error_type}]"

            tag_bbox_data_with_visualized_label = BboxData(
                image_path=tag_bbox_data.image_path,
                image_bytes=tag_bbox_data.image_bytes,
                cropped_image=tag_bbox_data.cropped_image,
                xmin=tag_bbox_data.xmin,
                ymin=tag_bbox_data.ymin,
                xmax=tag_bbox_data.xmax,
                ymax=tag_bbox_data.ymax,
                detection_score=tag_bbox_data.detection_score,
                label=label,
                classification_score=tag_bbox_data.classification_score
            )
            tag_bboxes_data_with_visualized_label.append(tag_bbox_data_with_visualized_label)

        tag_image_data_with_visualized_labels = ImageData(
            image_path=tag_image_data.image_path,
            image_bytes=tag_image_data.image_bytes,
            image=tag_image_data.image,
            bboxes_data=tag_bboxes_data_with_visualized_label
        )
        if tag == 'true':
            true_image_data_with_visualized_labels = tag_image_data_with_visualized_labels
        elif tag == 'pred':
            pred_image_data_with_visualized_labels = tag_image_data_with_visualized_labels

    return true_image_data_with_visualized_labels, pred_image_data_with_visualized_labels


error_type = Literal['TP', 'FP', 'FN', 'TP (extra bbox)', 'FP (extra bbox)']


def visualize_image_data_matching_side_by_side(
    image_data_matching: ImageDataMatching,
    error_type: Literal['detection', 'pipeline'],
    true_use_labels: bool = False,
    pred_use_labels: bool = False,
    true_score_type: Literal['detection', 'classification'] = None,
    pred_score_type: Literal['detection', 'classification'] = None,
    true_filter_by_error_types: List[error_type] = ['TP', 'FP', 'FN', 'TP (extra bbox)', 'FP (extra bbox)'],
    pred_filter_by_error_types: List[error_type] = ['TP', 'FP', 'FN', 'TP (extra bbox)', 'FP (extra bbox)'],
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None
) -> np.ndarray:

    (true_image_data_with_visualized_labels,
     pred_image_data_with_visualized_labels) = get_true_and_pred_images_data_with_visualized_labels(
        image_data_matching=image_data_matching,
        error_type=error_type
    )

    true_visualized_labels = [bbox_data.label for bbox_data in true_image_data_with_visualized_labels.bboxes_data]
    pred_visualized_labels = [bbox_data.label for bbox_data in pred_image_data_with_visualized_labels.bboxes_data]
    true_filter_by_labels = [
        label for label in true_visualized_labels
        if any(f"[{matching_error_type}]" in label for matching_error_type in true_filter_by_error_types)
    ]
    pred_filter_by_labels = [
        label for label in pred_visualized_labels
        if any(f"[{matching_error_type}]" in label for matching_error_type in pred_filter_by_error_types)
    ]

    image = visualize_images_data_side_by_side(
        image_data1=true_image_data_with_visualized_labels,
        image_data2=pred_image_data_with_visualized_labels,
        use_labels1=true_use_labels, use_labels2=pred_use_labels,
        score_type1=true_score_type, score_type2=pred_score_type,
        filter_by_labels1=true_filter_by_labels,
        filter_by_labels2=pred_filter_by_labels,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image
    )
    return image
