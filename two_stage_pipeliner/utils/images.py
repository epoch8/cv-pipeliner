import io
import math
from pathlib import Path
from typing import List, Tuple, Union, Callable, Literal

import imageio
from matplotlib.figure import Figure
from PIL import Image

import cv2
import numpy as np


def denormalize_bboxes(bboxes: List[Tuple[float, float, float, float]],
                       image_width: int,
                       image_height: int) -> List[Tuple[int, int, int, int]]:
    """
    Denormalize normalized bboxes coordinates.
    bboxes must have this format: (xmin, ymin, xmax, ymax)
    """
    bboxes = np.array(bboxes.copy())
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * image_width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * image_height
    bboxes = bboxes.round().astype(int)
    return bboxes


def cut_bboxes_from_image(
    image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
) -> List[np.ndarray]:

    img_boxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
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


def rotate_point(
    x: float, y: float, cx: float, cy: float, angle: float
) -> Tuple[float, float]:
    angle = math.radians(angle)
    xnew = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
    ynew = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
    return xnew, ynew


def get_label_to_base_label_image(
    base_labels_images_dir: Union[str, Path]
) -> Callable[[str], np.ndarray]:
    base_labels_images_dir = Path(base_labels_images_dir)
    base_labels_images_paths = list(base_labels_images_dir.glob('*.png')) + list(base_labels_images_dir.glob('*.jp*g'))
    ann_class_names = [base_label_image_path.stem for base_label_image_path in base_labels_images_paths]
    unknown_image_filepath = None
    for candidate in ['unknown.png', 'unknown.jpg', 'unknown.jp*g']:
        if (base_labels_images_dir / candidate).exists():
            unknown_image_filepath = base_labels_images_dir / candidate
            break
    if unknown_image_filepath is None:
        raise ValueError('base_labels_images_dir must have unknown.png, unknown.jpg or unknown.jpeg.')

    unknown_image = np.array(imageio.imread(unknown_image_filepath))
    label_to_base_label_image_dict = {}
    for label in ann_class_names + ['unknown']:
        filepath = base_labels_images_dir / f"{label}.png"
        if filepath.exists():
            render = np.array(imageio.imread(filepath))
        else:
            render = unknown_image
        label_to_base_label_image_dict[label] = render

    def label_to_base_label_image(label: str) -> np.ndarray:
        if label in label_to_base_label_image_dict:
            return label_to_base_label_image_dict[label].copy()
        else:
            return label_to_base_label_image_dict['unknown'].copy()

    return label_to_base_label_image


def concat_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    background_color_a: Tuple[int, int, int, int] = None,
    background_color_b: Tuple[int, int, int, int] = None,
    thumbnail_size_a: Tuple[int, int] = None,
    thumbnail_size_b: Tuple[int, int] = None,
    how: Literal['horizontally', 'vertically'] = 'horizontally'
) -> np.ndarray:
    if image_a.shape[-1] == 3:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2RGBA)
    if image_b.shape[-1] == 3:
        image_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2RGBA)
    if thumbnail_size_a is not None:
        image_a = Image.fromarray(image_a)
        image_a.thumbnail(thumbnail_size_b)
        image_a = np.array(image_a)
    if thumbnail_size_b is not None:
        image_b = Image.fromarray(image_b)
        image_b.thumbnail(thumbnail_size_b)
        image_b = np.array(image_b)

    ha, wa = image_a.shape[:2]
    hb, wb = image_b.shape[:2]

    if how == 'horizontally':
        max_height = np.max([ha, hb])
        total_width = wa + wb

        min_ha = max_height // 2 - ha // 2
        max_ha = max_height // 2 + ha // 2
        min_hb = max_height // 2 - hb // 2
        max_hb = max_height // 2 + hb // 2

        new_image = np.zeros(shape=(max_height, total_width, 4), dtype=np.uint8)
        new_image[min_ha:max_ha, :wa, :] = image_a[0:(max_ha-min_ha), :]
        new_image[min_hb:max_hb, wa:wa+wb, :] = image_b[0:(max_hb-min_hb), :]

        if background_color_a is not None:
            new_image[:3, :wa, :] = background_color_a
            new_image[-3:, :wa, :] = background_color_a
            new_image[:, :3, :] = background_color_a
            new_image[:, wa-2:wa, :] = background_color_a
        if background_color_b is not None:
            new_image[:3, wa:, :] = background_color_b
            new_image[-3:, wa:, :] = background_color_b
            new_image[:, -3:, :] = background_color_b
            new_image[:, wa:wa+2, :] = background_color_b
    elif how == 'vertically':
        max_width = np.max([wa, wb])
        total_height = ha + hb

        min_wa = max_width // 2 - wa // 2
        max_wa = max_width // 2 + wa // 2
        min_wb = max_width // 2 - wb // 2
        max_wb = max_width // 2 + wb // 2

        new_image = np.zeros(shape=(total_height, max_width, 4), dtype=np.uint8)
        new_image[:ha, min_wa:max_wa, :] = image_a[:, 0:(max_wa-min_wa)]
        new_image[ha:ha+hb, min_wb:max_wb, :] = image_b[:, 0:(max_wb-min_wb)]

        if background_color_a is not None:
            new_image[:ha, :3, :] = background_color_a
            new_image[:ha, -3:, :] = background_color_a
            new_image[:3, :, :] = background_color_a
            new_image[ha-2:ha:, :, :] = background_color_a
        if background_color_b is not None:
            new_image[ha:, :3, :] = background_color_b
            new_image[ha:, -3:, :] = background_color_b
            new_image[-3:, :, :] = background_color_b
            new_image[ha:ha+2, :, :] = background_color_b
    else:
        raise ValueError(
            "Parametr how must be 'horizontally' or 'vertically'"
        )

    return new_image
