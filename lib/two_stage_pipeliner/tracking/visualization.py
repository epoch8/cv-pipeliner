import imutils
import cv2
import numpy as np

from typing import List, Literal, Tuple

from two_stage_pipeliner.core.data import BboxData


def draw_label_image(
    image: np.ndarray,
    label_image: np.ndarray,
    bbox_data: BboxData
) -> np.ndarray:
    label_image = imutils.resize(label_image, width=int(max(image.shape) / 20))

    x_offset = bbox_data.xmin - 20
    y_offset = bbox_data.ymax - label_image.shape[0]

    y_min, y_max = y_offset, y_offset + label_image.shape[0]
    x_min, x_max = x_offset, x_offset + label_image.shape[1]

    # Ensure that label image is inside image boundaries
    if y_max > image.shape[0]:
        y_min -= y_max - image.shape[0]
        y_max = image.shape[0]

    if x_max > image.shape[1]:
        x_min -= x_max - image.shape[1]
        x_max = image.shape[1]

    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    alpha_label_image = label_image[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_label_image

    for channel in range(0, 3):
        image[y_min:y_max, x_min:x_max, channel] = (
            alpha_label_image * label_image[:, :, channel]
            + alpha_image * image[y_min:y_max, x_min:x_max, channel]
        )

    return image


def draw_overlay(
    frame: np.ndarray,
    tracked_bboxes: List[Tuple[int, int, int, int]],
    tracked_ids: List[int],
    ready_frames_at_the_moment: List['FrameResult'],
    label_images: List[np.ndarray],
    class_names_to_find: List[Literal[str, 'all classes']]
) -> np.ndarray:
    image = frame.copy()
    tracked_bboxes = tracked_bboxes.astype(int)
    ready_tracks_ids_at_the_moment = [
        ready_frame.track_id for ready_frame in ready_frames_at_the_moment
    ]
    for bbox, track_id in zip(tracked_bboxes, tracked_ids):
        xmin, ymin, xmax, ymax = bbox
        ready_frame = ready_frames_at_the_moment[ready_tracks_ids_at_the_moment.index(track_id)]
        label = ready_frame.bbox_data.label

        if label in class_names_to_find or 'all classes' in class_names_to_find:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            image = draw_label_image(
                image, label_images[label], ready_frame.bbox_data
            )

    return image
