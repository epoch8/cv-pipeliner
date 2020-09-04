import numpy as np


def voc_bboxes_to_coco(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    result = list(map(tuple, bboxes))
    return result


def coco_bboxes_to_voc(bboxes, frame_size):
    bboxes = np.clip(bboxes, 0, max(frame_size))
    bboxes = bboxes.round().astype("int")
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes
