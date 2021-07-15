import base64
import io
import json
import math
from pathlib import Path
from typing import List, Tuple, Union, Literal, Dict
from collections import defaultdict

import imageio
from matplotlib.figure import Figure
from PIL import Image, ImageFont, ImageDraw

import cv2
import numpy as np
import fsspec
import imutils
import PIL
from pathy import Pathy
from tqdm import tqdm

from cv_pipeliner.logging import logger


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
    xnew, ynew = int(xnew), int(ynew)
    return xnew, ynew


def is_base64(s: str):
    try:
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except Exception:
        return False


def open_image(
    image: Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image],
    open_as_rgb: bool = False
) -> np.ndarray:
    if isinstance(image, str) or isinstance(image, Path):
        if is_base64(str(image)):
            image_bytes = base64.b64decode(str(image))
        else:
            with fsspec.open(str(image), 'rb') as src:
                image_bytes = src.read()
    elif isinstance(image, fsspec.core.OpenFile):
        with image as src:
            image_bytes = src.read()
    elif isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, io.BytesIO):
        image_bytes = image.getvalue()
    elif isinstance(image, PIL.Image.Image):
        image_bytes = np.array(image)
    else:
        raise ValueError(f'Got unknown type: {type(image)}.')
    if not isinstance(image_bytes, np.ndarray):
        image = np.array(imageio.imread(image_bytes))
    else:
        image = image_bytes
    if open_as_rgb:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def concat_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    background_color_a: Tuple[int, int, int, int] = None,
    background_color_b: Tuple[int, int, int, int] = None,
    thumbnail_size_a: Tuple[int, int] = None,
    thumbnail_size_b: Tuple[int, int] = None,
    how: Literal['horizontally', 'vertically'] = 'horizontally',
    mode: Literal['L', 'RGB', 'RGBA'] = 'RGBA',
    background_edge_width: int = 3,
    between_edge_width: int = 0
) -> np.ndarray:
    if image_a is None and image_b is not None:
        return image_b
    if image_a is not None and image_b is None:
        return image_a
    if len(image_a.shape) == 2 or image_a.shape[-1] == 1:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2RGBA)
    elif image_a.shape[-1] == 3:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2RGBA)
    if len(image_b.shape) == 2 or image_b.shape[-1] == 1:
        image_b = cv2.cvtColor(image_b, cv2.COLOR_GRAY2RGBA)
    elif image_b.shape[-1] == 3:
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
        total_width = wa + wb + between_edge_width

        min_ha = max_height // 2 - ha // 2
        max_ha = max_height // 2 + ha // 2
        min_hb = max_height // 2 - hb // 2
        max_hb = max_height // 2 + hb // 2

        new_image = np.zeros(shape=(max_height, total_width, 4), dtype=np.uint8)
        new_image[min_ha:max_ha, :wa, :] = image_a[0:(max_ha-min_ha), :]
        new_image[min_hb:max_hb, wa+between_edge_width:wa+between_edge_width+wb, :] = image_b[0:(max_hb-min_hb), :]

        if background_color_a is not None:
            new_image[:background_edge_width, :wa, :] = background_color_a
            new_image[-background_edge_width:, :wa, :] = background_color_a
            new_image[:, :background_edge_width, :] = background_color_a
            new_image[:, wa+between_edge_width-(background_edge_width-1):wa+between_edge_width, :] = background_color_a
        if background_color_b is not None:
            new_image[:background_edge_width, wa:, :] = background_color_b
            new_image[-background_edge_width:, wa:, :] = background_color_b
            new_image[:, -background_edge_width:, :] = background_color_b
            new_image[:, wa+between_edge_width:wa+between_edge_width+(background_edge_width-1), :] = background_color_b
    elif how == 'vertically':
        max_width = np.max([wa, wb])
        total_height = ha + hb + between_edge_width

        min_wa = max_width // 2 - wa // 2
        max_wa = max_width // 2 + wa // 2
        min_wb = max_width // 2 - wb // 2
        max_wb = max_width // 2 + wb // 2

        new_image = np.zeros(shape=(total_height, max_width, 4), dtype=np.uint8)
        new_image[:ha, min_wa:max_wa, :] = image_a[:, 0:(max_wa-min_wa)]
        new_image[ha+between_edge_width:ha+between_edge_width+hb, min_wb:max_wb, :] = image_b[:, 0:(max_wb-min_wb)]

        if background_color_a is not None:
            new_image[:ha, :background_edge_width, :] = background_color_a
            new_image[:ha, -background_edge_width:, :] = background_color_a
            new_image[:background_edge_width, :, :] = background_color_a
            new_image[ha+between_edge_width-(background_edge_width-1):ha+between_edge_width, :, :] = background_color_a
        if background_color_b is not None:
            new_image[ha:, :background_edge_width, :] = background_color_b
            new_image[ha:, -background_edge_width:, :] = background_color_b
            new_image[-background_edge_width:, :, :] = background_color_b
            new_image[ha+between_edge_width:ha+between_edge_width+(background_edge_width-1), :, :] = background_color_b
    else:
        raise ValueError(
            "Parametr how must be 'horizontally' or 'vertically'"
        )

    if mode == 'RGBA':
        pass
    elif mode == 'RGB':
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)
    elif mode == 'L':
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError(
            "Parametr mode must be 'RGBA' or 'RGB' or 'L'"
        )

    return new_image


def put_text_on_image(
    image: np.ndarray,
    text: str,
    fontsize: int,
    ymax: int,
    maximum_width: int,
    font: str = 'arial.ttf',
    fill: str = 'black'
) -> np.ndarray:
    texts = [text[i:i+maximum_width] for i in range(0, len(text), maximum_width)]
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(font, fontsize)
    except IOError:
        font = ImageFont.load_default()
    width, height = image_pil.size
    for i, subtext in enumerate(texts):
        text_width, text_height = font.getsize(subtext)
        left = (width - text_width) // 2
        text_bottom = ymax + (i+1)*fontsize
        margin = np.ceil(0.05 * text_height)
        draw.text(
            xy=(left + margin, text_bottom - text_height - margin),
            text=subtext,
            fill=fill,
            font=font
        )
    image = np.array(image_pil)
    return image


def draw_rectangle(
    image: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float
):
    image_original = image.copy()
    image = image.copy()

    image[max(0, ymin-thickness):ymin, max(0, xmin-thickness):(xmax+thickness), :] = np.uint8(
        (1-alpha) * image_original[max(0, ymin-thickness):ymin, max(0, xmin-thickness):(xmax+thickness), :] + alpha * np.array(color)  # noqa: E501
    )
    image[ymax:ymax+thickness, max(0, xmin-thickness):(xmax+thickness), :] = np.uint8(
        (1-alpha) * image_original[ymax:ymax+thickness, max(0, xmin-thickness):(xmax+thickness), :] + alpha * np.array(color)  # noqa: E501
    )
    image[max(0, ymin-thickness):(ymax+thickness), xmax:xmax+thickness, :] = np.uint8(
        (1-alpha) * image_original[max(0, ymin-thickness):(ymax+thickness), xmax:xmax+thickness, :] + alpha * np.array(color)  # noqa: E501
    )
    image[max(0, ymin-thickness):ymax+thickness, max(0, xmin-thickness):xmin, :] = np.uint8(
        (1-alpha) * image_original[max(0, ymin-thickness):ymax+thickness, max(0, xmin-thickness):xmin, :] + alpha * np.array(color)  # noqa: E501
    )
    return image


# TODO: Remove 4D padding as it's very slow.
def get_base_label_image_with_description(
    base_label_image: np.ndarray,
    label: str,
    description: str,
    pad_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    pad_resize: int = 90,
    pad_width: int = 5,
    base_resize: int = 150,
    maximum_text_width: int = 20
) -> np.ndarray:
    if len(base_label_image.shape) == 2 or base_label_image.shape[-1] == 1:
        base_label_image = cv2.cvtColor(base_label_image, cv2.COLOR_GRAY2RGBA)
    elif base_label_image.shape[-1] == 3:
        base_label_image = cv2.cvtColor(base_label_image, cv2.COLOR_RGB2RGBA)
    base_label_image = imutils.resize(base_label_image, width=base_resize, height=base_resize)
    height, width, _ = base_label_image.shape
    base_label_image = np.pad(
        base_label_image,
        pad_width=(
            (max(0, pad_resize-height//2), max(0, pad_resize-height//2)),
            (max(0, pad_resize-width//2), max(0, pad_resize-width//2)),
            (0, 0)
        ),
        constant_values=((pad_color, pad_color), (pad_color, pad_color), (0, 0)),
        mode='constant'
    )
    how_many = 30 * len(description) // maximum_text_width
    base_label_image = np.pad(
        base_label_image,
        pad_width=((how_many, 0), (0, 0), (0, 0)),
        constant_values=((pad_color, 0), (0, 0), (0, 0)),
        mode='constant'
    )
    base_label_image = np.pad(
        base_label_image,
        pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
        constant_values=((0, 0), (0, 0), (0, 0)),
        mode='constant'
    )
    base_label_image = np.pad(
        base_label_image,
        pad_width=((60, 0), (0, 0), (0, 0)),
        constant_values=(((255, 255, 255, 255), 0), (0, 0), (0, 0)),
        mode='constant'
    )
    fontsize1, fontsize2 = 30, 17
    ymax1, ymax2 = 25, 10
    base_label_image = put_text_on_image(
        image=base_label_image,
        text=label,
        fontsize=fontsize1,
        ymax=ymax1,
        maximum_width=maximum_text_width
    )
    base_label_image = put_text_on_image(
        image=base_label_image,
        text=description,
        fontsize=fontsize2,
        ymax=fontsize1+ymax1+ymax2,
        maximum_width=maximum_text_width
    )

    return base_label_image


def get_label_to_description(
    label_to_description_dict: Union[str, Path, Dict, None],
    default_description: str = 'No description.'
) -> Dict[str, str]:
    if label_to_description_dict is None:
        label_to_description_dict = {}
    elif isinstance(label_to_description_dict, str) or isinstance(label_to_description_dict, Path):
        with fsspec.open(label_to_description_dict, 'r') as src:
            label_to_description_dict = json.load(src)

    label_to_description = defaultdict(lambda: default_description)
    label_to_description['unknown'] = default_description
    for k in label_to_description_dict:
        label_to_description[k] = label_to_description_dict[k]

    return label_to_description


def get_label_to_base_label_image(
    base_labels_images: Union[str, Path],
    label_to_description: Union[str, Path, Dict[str, str]] = None,
    add_label_to_image: bool = False,
    make_labels_for_these_class_names_too: List[str] = []  # add known description to classes without base images
) -> Dict[str, np.ndarray]:
    if base_labels_images is None:
        return None

    base_labels_images_files = fsspec.open_files(str(base_labels_images))
    ann_class_names_files = [
        Pathy(base_label_image_file.path).stem for base_label_image_file in base_labels_images_files
    ]
    unique_ann_class_names = set(ann_class_names_files)
    if 'unknown' not in unique_ann_class_names:
        raise ValueError(
            f'"{base_labels_images}" must have image with name "unknown.*"'
        )
    unknown_image_path = base_labels_images_files[ann_class_names_files.index('unknown')]
    label_to_base_label_image = defaultdict(lambda: unknown_image_path)
    label_to_base_label_image['unknown'] = unknown_image_path
    logger.info(f"Loading base labels images from {base_labels_images}...")
    for label in tqdm(list(unique_ann_class_names) + list(set(make_labels_for_these_class_names_too))):
        if label in unique_ann_class_names:
            base_label_image = base_labels_images_files[ann_class_names_files.index(label)]
        else:
            base_label_image = label_to_base_label_image['unknown']
        label_to_base_label_image[label] = base_label_image

    def label_to_base_label_image_func(
        label: str,
        label_to_description: Union[str, Path, Dict[str, str]] = label_to_description,
        add_label_to_image: bool = add_label_to_image
    ):
        base_label_image = open_image(label_to_base_label_image[label])
        if label_to_description is not None:
            if isinstance(label_to_description, str) or isinstance(label_to_description, Path):
                label_to_description = get_label_to_description(label_to_description_dict=label_to_description)
            base_label_image = get_base_label_image_with_description(
                base_label_image=base_label_image,
                label=label,
                description=label_to_description[label]
            )
        elif add_label_to_image:
            base_label_image = get_base_label_image_with_description(
                base_label_image=base_label_image,
                label=label,
                description=''
            )
        return base_label_image

    return label_to_base_label_image_func


def draw_n_base_labels_images(
    labels: List[str],
    label_to_base_label_image: Dict[str, np.ndarray],
    label_to_description: Union[str, Path, Dict[str, str]] = None,
    images_per_row: int = 7
) -> np.ndarray:
    rows = np.array_split(labels, max(1, len(labels) // images_per_row))
    total_image = None
    i = 0
    for row in rows:
        total_image_row = None
        for j, label in enumerate(row):
            image_b = label_to_base_label_image(label)
            if label_to_description is not None:
                image_b = get_base_label_image_with_description(
                    base_label_image=image_b,
                    label=label,
                    description=label_to_description[label]
                )
            image_b = np.pad(
                image_b,
                pad_width=((60, 0), (0, 0), (0, 0)),
                constant_values=((0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
            image_b = put_text_on_image(
                image=image_b,
                text=str(i + 1),
                fontsize=60,
                ymax=5,
                maximum_width=14
            )
            if total_image_row is None:
                total_image_row = image_b
            else:
                total_image_row = concat_images(
                    image_a=total_image_row,
                    image_b=image_b,
                    how='horizontally',
                    background_color_a=[0, 0, 0, 255] if j == 1 else None,
                    background_color_b=[0, 0, 0, 255],
                    thumbnail_size_a=(300, 300) if j == 1 else None,
                    thumbnail_size_b=(300, 300)
                )
            i += 1
        if total_image is None:
            total_image = total_image_row
        else:
            total_image = concat_images(
                image_a=total_image,
                image_b=total_image_row,
                how='vertically'
            )
    return total_image


def get_image_b64(
    image: np.ndarray,
    format: str
) -> str:
    image_io = io.BytesIO()
    image = np.array(image, dtype=np.uint8)
    Image.fromarray(image).save(image_io, format=format)
    image_format = image_io.getvalue()
    image_format_b64 = base64.b64encode(image_format).decode('utf-8')
    return image_format_b64


def draw_quadrangle_on_image(
    image: np.ndarray,
    points: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    color: Tuple[int, int, int] = None,
    thickness: int = 3,
    alpha: float = 0.3,
) -> np.ndarray:
    image = image.copy()
    rect = cv2.minAreaRect(np.array(points))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cropped_image_zeros = np.ones_like(image)
    cv2.drawContours(
        image=cropped_image_zeros,
        contours=[box],
        contourIdx=0,
        color=color,
        thickness=thickness
    )
    colored_regions = (cropped_image_zeros == color)
    image[colored_regions] = (
        (1 - alpha) * image[colored_regions] + alpha * cropped_image_zeros[colored_regions]
    )
    return image
