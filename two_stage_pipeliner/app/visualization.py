from typing import Tuple, Callable, Literal, List

import numpy as np
import imutils

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.metrics.image_data_matching import BboxDataMatching, ImageDataMatching
from two_stage_pipeliner.utils.images import concat_images

import streamlit as st


@st.cache(show_spinner=False)
def get_illustrated_bboxes_data(
    source_image: np.ndarray,
    bboxes_data: List[BboxData],
    label_to_base_label_image: Callable[[str], np.ndarray],
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    max_images_size: int = None
) -> Tuple[List[np.ndarray], List[str]]:
    cropped_images_and_renders = []
    cropped_images = [bbox_data.open_cropped_image(source_image=source_image) for bbox_data in bboxes_data]
    labels = [bbox_data.label for bbox_data in bboxes_data]
    renders = [label_to_base_label_image(label) for label in labels]
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
        if max_images_size is not None:
            height, width, _ = cropped_image_and_render.shape
            if max(height, width) > max_images_size:
                if height <= width:
                    cropped_image_and_render = imutils.resize(cropped_image_and_render, width=max_images_size)
                else:
                    cropped_image_and_render = imutils.resize(cropped_image_and_render, height=max_images_size)

        cropped_images_and_renders.append(cropped_image_and_render)

    return cropped_images_and_renders, labels


@st.cache(show_spinner=False)
def get_illustrated_bboxes_data_matchings(
    source_image: np.ndarray,
    bboxes_data_matchings: List[BboxDataMatching],
    label_to_base_label_image: Callable[[str], np.ndarray],
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None,
    max_images_size: int = None
) -> Tuple[List[np.ndarray], List[str]]:
    cropped_images_and_renders = []

    true_bboxes_data = [bbox_data_matching.true_bbox_data for bbox_data_matching in bboxes_data_matchings]
    pred_bboxes_data = [bbox_data_matching.pred_bbox_data for bbox_data_matching in bboxes_data_matchings]
    pred_cropped_images = [bbox_data.open_cropped_image(source_image=source_image) for bbox_data in pred_bboxes_data]
    true_labels = [bbox_data.label if bbox_data is not None else "unknown" for bbox_data in true_bboxes_data]
    pred_labels = [bbox_data.label for bbox_data in pred_bboxes_data]
    true_renders = [label_to_base_label_image(true_label) for true_label in true_labels]
    pred_renders = [label_to_base_label_image(pred_label) for pred_label in pred_labels]
    labels = [f"{pred_label}/{true_label}" for pred_label, true_label in zip(pred_labels, true_labels)]
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

        if max_images_size is not None:
            height, width, _ = cropped_image_and_render.shape
            if max(height, width) > max_images_size:
                if height <= width:
                    cropped_image_and_render = imutils.resize(cropped_image_and_render, width=max_images_size)
                else:
                    cropped_image_and_render = imutils.resize(cropped_image_and_render, height=max_images_size)

        cropped_images_and_renders.append(cropped_image_and_render)

    return cropped_images_and_renders, labels


@st.cache(show_spinner=False)
def get_image_data_matching(
    true_image_data: ImageData,
    pred_image_data: ImageData,
    minimum_iou: float
) -> ImageDataMatching:
    image_data_matching = ImageDataMatching(
        true_image_data=true_image_data,
        pred_image_data=pred_image_data,
        minimum_iou=minimum_iou
    )
    return image_data_matching


def illustrate_bboxes_data(
    true_image_data: ImageData,
    label_to_base_label_image: Callable[[str], np.ndarray],
    mode: Literal['many', 'one-by-one'],
    pred_image_data: ImageData = None,
    minimum_iou: float = None,
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None,
    average_maximum_images_per_page: int = 50,
    max_images_size: int = 400
):
    source_image = true_image_data.open_image()

    if pred_image_data is None:
        bboxes_data = true_image_data.bboxes_data
    else:
        assert minimum_iou is not None
        assert pred_background_color_b is not None
        image_data_matching = get_image_data_matching(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou
        )
        bboxes_data = [
            bbox_data_matching for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.pred_bbox_data is not None
        ]

    if len(bboxes_data) == 0:
        return

    if pred_image_data is not None:
        st.text(f"Found {len(pred_image_data.bboxes_data)} bricks!")
        st.text(f"""
True Positives: {image_data_matching.get_pipeline_TP()}
False Positives: {image_data_matching.get_pipeline_FP()}
False Negatives: {image_data_matching.get_pipeline_FN()}

Extra bboxes counts: {image_data_matching.get_detection_FP()}
True Positives on extra bboxes: {image_data_matching.get_pipeline_TP_extra_bbox()}
False Positives on extra bboxes: {image_data_matching.get_pipeline_FP_extra_bbox()}""")
    else:
        st.text(f"Found {len(true_image_data.bboxes_data)} bricks!")

    n_split = int(np.ceil(len(bboxes_data) / average_maximum_images_per_page))
    splitted_bboxes_data = np.array_split(bboxes_data, n_split)

    if n_split >= 2:
        page = st.slider(
            label="",
            min_value=1,
            max_value=n_split
        )
        page_bboxes_data = splitted_bboxes_data[page-1]
    else:
        page_bboxes_data = splitted_bboxes_data[0]

    if pred_image_data is None:
        cropped_images_and_renders, labels = get_illustrated_bboxes_data(
            source_image=source_image,
            bboxes_data=page_bboxes_data,
            label_to_base_label_image=label_to_base_label_image,
            background_color_a=background_color_a,
            true_background_color_b=true_background_color_b,
            max_images_size=max_images_size
        )
    else:
        cropped_images_and_renders, labels = get_illustrated_bboxes_data_matchings(
            source_image=source_image,
            bboxes_data_matchings=page_bboxes_data,
            label_to_base_label_image=label_to_base_label_image,
            background_color_a=background_color_a,
            true_background_color_b=true_background_color_b,
            pred_background_color_b=pred_background_color_b,
            max_images_size=max_images_size
        )

    if mode == "one-by-one":
        for cropped_image_and_render, label in zip(cropped_images_and_renders, labels):
            st.image(image=cropped_image_and_render)
            st.markdown(label)
            st.markdown('----')
    elif mode == "many":
        st.image(image=cropped_images_and_renders, caption=labels)
        st.markdown('----')
