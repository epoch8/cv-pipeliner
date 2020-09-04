from pathlib import Path
from typing import Tuple, Callable, Literal

import cv2
import imageio
import numpy as np
from PIL import Image

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching

import streamlit as st


@st.cache(show_spinner=False)
def get_label_to_base_label_image(base_labels_images_dir) -> Callable[[str], np.ndarray]:
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
            return label_to_base_label_image_dict[label]
        else:
            return label_to_base_label_image_dict['unknown']

    return label_to_base_label_image


def concat_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    background_color_a: Tuple[int, int, int, int] = None,
    background_color_b: Tuple[int, int, int, int] = None,
    thumbnail_size_a: Tuple[int, int] = None,
    thumbnail_size_b: Tuple[int, int] = None
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

    return new_image


@st.cache(show_spinner=False)
def get_illustrated_bboxes_data(
    true_image_data: ImageData,
    minimum_iou: float,
    label_to_base_label_image: Callable[[str], np.ndarray],
    pred_image_data: ImageData = None,
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None
):
    cropped_images_and_renders = []

    if pred_image_data is not None:
        image_data_matching = ImageDataMatching(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou
        )
        bboxes_data_matchings = [
            bbox_data_matching for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.pred_bbox_data is not None
        ]
        true_bboxes_data = [bbox_data_matching.true_bbox_data for bbox_data_matching in bboxes_data_matchings]
        pred_bboxes_data = [bbox_data_matching.pred_bbox_data for bbox_data_matching in bboxes_data_matchings]
        pred_cropped_images = [bbox_data.open_cropped_image() for bbox_data in pred_bboxes_data]
        true_labels = [bbox_data.label if bbox_data is not None else "unknown" for bbox_data in true_bboxes_data]
        pred_labels = [bbox_data.label for bbox_data in pred_bboxes_data]
        true_renders = [label_to_base_label_image(true_label) for true_label in true_labels]
        pred_renders = [label_to_base_label_image(pred_label) for pred_label in pred_labels]
        caption = [f"{pred_label}/{true_label}" for pred_label, true_label in zip(pred_labels, true_labels)]
        for cropped_image, true_render, pred_render in zip(pred_cropped_images, true_renders, pred_renders):
            height, width, _ = cropped_image.shape
            size = max(height, width, 50)
            thumbnail_size_b = (size, size)
            cropped_image_and_render = concat_images(
                image_a=cropped_image,
                image_b=pred_render,
                background_color_a=background_color_a,
                background_color_b=pred_background_color_b,
                thumbnail_size_b=thumbnail_size_b
            )
            cropped_image_and_render = concat_images(
                image_a=cropped_image_and_render,
                image_b=true_render,
                background_color_b=true_background_color_b,
                thumbnail_size_b=thumbnail_size_b
            )

            cropped_images_and_renders.append(cropped_image_and_render)
    else:
        cropped_images = [bbox_data.open_cropped_image() for bbox_data in true_image_data.bboxes_data]
        labels = [bbox_data.label for bbox_data in true_image_data.bboxes_data]
        renders = [label_to_base_label_image(label) for label in labels]
        caption = labels
        for cropped_image, render in zip(cropped_images, renders):
            height, width, _ = cropped_image.shape
            size = max(height, width, 50)
            thumbnail_size_b = (size, size)
            cropped_image_and_render = concat_images(
                image_a=cropped_image,
                image_b=render,
                background_color_a=background_color_a,
                background_color_b=true_background_color_b,
                thumbnail_size_b=thumbnail_size_b
            )
            cropped_images_and_renders.append(cropped_image_and_render)

    return cropped_images_and_renders, caption


def illustrate_bboxes_data(
    true_image_data: ImageData,
    minimum_iou: float,
    label_to_base_label_image: Callable[[str], np.ndarray],
    mode: Literal['many', 'one-by-one'],
    pred_image_data: ImageData = None,
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None,
):
    cropped_images_and_renders, caption = get_illustrated_bboxes_data(
        true_image_data=true_image_data,
        minimum_iou=minimum_iou,
        label_to_base_label_image=label_to_base_label_image,
        pred_image_data=pred_image_data,
        background_color_a=background_color_a,
        true_background_color_b=true_background_color_b,
        pred_background_color_b=pred_background_color_b
    )
    if mode == "one-by-one":
        for cropped_image_and_render, label in zip(cropped_images_and_renders, caption):
            st.image(image=cropped_image_and_render)
            st.markdown(label)
            st.markdown('----')
    elif mode == "many":
        st.image(image=cropped_images_and_renders, caption=caption)
        st.markdown('----')
