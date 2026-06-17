from typing import List, Optional

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.results import DetectionResult


def build_detection_images_data(
    images_data: List[ImageData],
    detection_result: DetectionResult,
    open_images_in_images_data: bool,
    open_cropped_images_in_bboxes_data: bool,
    labels_top_n: Optional[List[List[List[str]]]] = None,
    classification_scores_top_n: Optional[List[List[List[float]]]] = None,
) -> List[ImageData]:
    labels_top_n = labels_top_n if labels_top_n is not None else detection_result.labels_top_n
    classification_scores_top_n = (
        classification_scores_top_n
        if classification_scores_top_n is not None
        else detection_result.classification_scores_top_n
    )

    pred_images_data = []
    for image_idx, image_data in enumerate(images_data):
        bboxes_data = []
        pred_bboxes = detection_result.bboxes[image_idx]
        pred_keypoints = detection_result.keypoints[image_idx]
        pred_masks = detection_result.masks[image_idx]
        pred_detection_scores = detection_result.detection_scores[image_idx]
        image_labels_top_n = labels_top_n[image_idx] if labels_top_n is not None else None
        image_scores_top_n = classification_scores_top_n[image_idx] if classification_scores_top_n is not None else None

        for bbox_idx, (pred_bbox, keypoints, mask, detection_score) in enumerate(
            zip(pred_bboxes, pred_keypoints, pred_masks, pred_detection_scores)
        ):
            xmin, ymin, xmax, ymax = pred_bbox
            kwargs = {}
            if image_labels_top_n is not None:
                label_top_n = image_labels_top_n[bbox_idx]
                kwargs.update(label=label_top_n[0], top_n=len(label_top_n), labels_top_n=label_top_n)
            if image_scores_top_n is not None:
                score_top_n = image_scores_top_n[bbox_idx]
                kwargs.update(
                    classification_score=score_top_n[0],
                    classification_scores_top_n=score_top_n,
                )
            bboxes_data.append(
                BboxData(
                    image_path=image_data.image_path,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    keypoints=keypoints,
                    mask=mask,
                    detection_score=detection_score,
                    **kwargs,
                )
            )

        if open_cropped_images_in_bboxes_data:
            for bbox_data in bboxes_data:
                bbox_data.open_cropped_image(source_image=image_data.image, inplace=True)

        pred_images_data.append(
            ImageData(
                image_path=image_data.image_path,
                image=image_data.image if open_images_in_images_data else None,
                bboxes_data=bboxes_data,
                label=image_data.label,
                keypoints=image_data.keypoints,
                mask=image_data.mask,
                additional_info=image_data.additional_info,
                meta_width=image_data.meta_width,
                meta_height=image_data.meta_height,
            )
        )
    return pred_images_data
