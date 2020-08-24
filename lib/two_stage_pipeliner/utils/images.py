import io
from typing import List, Tuple
from matplotlib.figure import Figure

import cv2
import numpy as np


# WARNING: bbox has to be in this format: (ymin, xmin, ymax, xmax)


def denormalize_bboxes(bboxes: List[Tuple[float, float, float, float]],
                       image_width: int,
                       image_height: int) -> List[Tuple[int, int, int, int]]:
    bboxes = np.array(bboxes.copy())
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * image_height
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * image_width
    bboxes = bboxes.round().astype(int)
    return bboxes


def cut_bboxes_from_image(
    image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
) -> List[np.ndarray]:

    img_boxes = []
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = bbox
        img_boxes.append(image[ymin:ymax, xmin:xmax])
    return img_boxes


def get_img_from_fig(fig: Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
