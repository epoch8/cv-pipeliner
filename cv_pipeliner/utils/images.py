import io
import math
from pathlib import Path
from typing import List, Tuple, Union, Callable, Literal

import imageio
from matplotlib.figure import Figure
from PIL import Image, ImageFont, ImageDraw

import cv2
import numpy as np
import fsspec
import zipfile
import tempfile
from pathy import Pathy
from tqdm import tqdm

from cv_pipeliner.logging import logger
from cv_pipeliner.utils.files import fixed_fsspec_glob


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


def open_image_with_fsspec(
    image_path: Union[str, Path],
    open_as_rgb: bool = False
) -> np.ndarray:
    image_path = Pathy(image_path)
    if image_path.scheme == 'gs':
        fs = fsspec.filesystem('gcs')
        with fs.open(image_path, 'rb') as src:
            image_bytes = src.read()
        image = np.array(imageio.imread(image_bytes))
    else:
        image = np.array(imageio.imread(image_path, pilmode="RGB"))

    if open_as_rgb:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def get_label_to_base_label_image(
    base_labels_images_dir: Union[str, Path],
    fs: fsspec.filesystem = fsspec.filesystem('file')
) -> Callable[[str], np.ndarray]:
    base_labels_images_dir = Pathy(base_labels_images_dir)

    if base_labels_images_dir.suffix == '.zip':  # loading one-by-one from GCS is very slow
        logger.info('Zip file detected. Extracting to temp folder...')
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)

        temp_zip_file = temp_dir_path / base_labels_images_dir.name
        with open(temp_zip_file, 'wb') as out:
            with fs.open(str(base_labels_images_dir), 'rb') as src:
                out.write(src.read())
        with zipfile.ZipFile(temp_zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_path)
        temp_zip_file.unlink()
        base_labels_images_dir = temp_dir_path
        fs = fsspec.filesystem('file')
    else:
        temp_dir = None

    base_labels_images_paths = (
        list(fixed_fsspec_glob(fs, str(base_labels_images_dir / '*.png'))) +
        list(fixed_fsspec_glob(fs, str(base_labels_images_dir / '*.jp*g')))
    )
    ann_class_names = [Pathy(base_label_image_path).stem for base_label_image_path in base_labels_images_paths]
    unknown_image_candidates = list(fixed_fsspec_glob(fs, str(base_labels_images_dir / 'unknown.*')))
    if len(unknown_image_candidates) == 0:
        raise ValueError(
            f'Folder "{base_labels_images_dir}" must have image with name "unknown.*"'
        )
    else:
        unknown_image_filepath = unknown_image_candidates[0]
    with fs.open(unknown_image_filepath, 'rb') as src:
        unknown_image = np.array(imageio.imread(src.read()))
    label_to_base_label_image_dict = {}
    logger.info(f"Loading base labels images from {base_labels_images_dir}...")
    for label in tqdm(ann_class_names + ['unknown']):
        filepath_candidates = list(fixed_fsspec_glob(fs, str(base_labels_images_dir / f"{label}.*")))
        if len(filepath_candidates) > 0:
            filepath = filepath_candidates[0]
            with fs.open(filepath, 'rb') as src:
                render = np.array(imageio.imread(src.read()))
        else:
            render = unknown_image
        label_to_base_label_image_dict[label] = render

    if temp_dir is not None:
        temp_dir.cleanup()

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
    font = ImageFont.truetype(font, fontsize)
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
