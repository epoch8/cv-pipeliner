from typing import Tuple, Callable, Literal, List, Dict

import numpy as np
import imutils

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.metrics.image_data_matching import BboxDataMatching, ImageDataMatching
from cv_pipeliner.utils.images import concat_images, open_image
from cv_pipeliner.visualizers.core.image_data import visualize_image_data

import streamlit as st


def get_illustrated_bboxes_data(
    source_image: np.ndarray,
    bboxes_data: List[BboxData],
    label_to_base_label_image: Callable[[str], np.ndarray],
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    max_images_size: int = None,
    bbox_offset: int = 0,
    draw_rectangle_with_color: Tuple[int, int, int, int] = None,
    alpha: float = 0.3
) -> Tuple[List[np.ndarray], List[str]]:
    cropped_images_and_renders = []
    cropped_images = [
        bbox_data.open_cropped_image(
            source_image=source_image,
            xmin_offset=bbox_offset,
            ymin_offset=bbox_offset,
            xmax_offset=bbox_offset,
            ymax_offset=bbox_offset,
            draw_rectangle_with_color=draw_rectangle_with_color,
            alpha=alpha
        )
        for bbox_data in bboxes_data
    ]
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


def get_illustrated_bboxes_data_matchings(
    source_image: np.ndarray,
    bboxes_data_matchings: List[BboxDataMatching],
    label_to_base_label_image: Callable[[str], np.ndarray],
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None,
    max_images_size: int = None,
    bbox_offset: int = 0,
    draw_rectangle_with_color: Tuple[int, int, int] = None
) -> Tuple[List[np.ndarray], List[str]]:
    cropped_images_and_renders = []

    true_bboxes_data = [bbox_data_matching.true_bbox_data for bbox_data_matching in bboxes_data_matchings]
    pred_bboxes_data = [bbox_data_matching.pred_bbox_data for bbox_data_matching in bboxes_data_matchings]
    pred_cropped_images = [bbox_data.open_cropped_image(
        source_image=source_image,
        xmin_offset=bbox_offset,
        ymin_offset=bbox_offset,
        xmax_offset=bbox_offset,
        ymax_offset=bbox_offset,
        draw_rectangle_with_color=draw_rectangle_with_color
    ) for bbox_data in pred_bboxes_data]
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


class PageSession:
    pass


@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_page_session():
    page_session = PageSession()
    page_session.counter = 1
    return page_session


@st.cache(allow_output_mutation=True, show_spinner=False)
def fetch_image_data(image_data: ImageData):
    page_session = PageSession()
    page_session.bboxes_data_fast_annotations = {
        (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax): None
        for bbox_data in image_data.bboxes_data
    }
    return page_session


def illustrate_bboxes_data(
    true_image_data: ImageData,
    label_to_base_label_image: Callable[[str], np.ndarray],
    label_to_description: Dict[str, str],
    mode: Literal['many', 'one-by-one'],
    pred_image_data: ImageData = None,
    minimum_iou: float = None,
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    pred_background_color_b: Tuple[int, int, int, int] = None,
    average_maximum_images_per_page: int = 50,
    max_images_size: int = 400,
    bbox_offset: int = 0,
    draw_rectangle_with_color: Tuple[int, int, int] = None,
    change_annotation: Callable[[BboxData], None] = None,
    show_top_n: bool = False,
    fast_annotation_mode: bool = False
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
        st.text(f"Found {len(pred_image_data.bboxes_data)} bboxes!")
        st.text(f"""
True Positives: {image_data_matching.get_pipeline_TP()}
False Positives: {image_data_matching.get_pipeline_FP()}
False Negatives: {image_data_matching.get_pipeline_FN()}

Extra bboxes counts: {image_data_matching.get_detection_FP()}
True Positives on extra bboxes: {image_data_matching.get_pipeline_TP_extra_bbox()}
False Positives on extra bboxes: {image_data_matching.get_pipeline_FP_extra_bbox()}""")
        st.markdown('---')
    else:
        st.text(f"Found {len(true_image_data.bboxes_data)} bboxes!")

    n_split = int(np.ceil(len(bboxes_data) / average_maximum_images_per_page))
    splitted_bboxes_data = np.array_split(bboxes_data, n_split)

    if change_annotation is not None:
        page_session = fetch_page_session()
        current_page = page_session.counter
        if current_page > n_split:
            current_page = 1
            page_session.counter = 1
    else:
        current_page = 1

    if n_split >= 2:
        page = st.slider(
            label="",
            min_value=1,
            max_value=n_split,
            value=current_page
        )
        page_bboxes_data = splitted_bboxes_data[page-1]
    else:
        page = 1
        page_bboxes_data = splitted_bboxes_data[0]

    if change_annotation is not None and page != current_page:
        page_session.counter = page

    if pred_image_data is None:
        cropped_images_and_renders, labels = get_illustrated_bboxes_data(
            source_image=source_image,
            bboxes_data=page_bboxes_data,
            label_to_base_label_image=label_to_base_label_image,
            background_color_a=background_color_a,
            true_background_color_b=true_background_color_b,
            max_images_size=max_images_size,
            bbox_offset=bbox_offset,
            draw_rectangle_with_color=draw_rectangle_with_color
        )
    else:
        cropped_images_and_renders, labels = get_illustrated_bboxes_data_matchings(
            source_image=source_image,
            bboxes_data_matchings=page_bboxes_data,
            label_to_base_label_image=label_to_base_label_image,
            background_color_a=background_color_a,
            true_background_color_b=true_background_color_b,
            pred_background_color_b=pred_background_color_b,
            max_images_size=max_images_size,
            bbox_offset=bbox_offset,
            draw_rectangle_with_color=draw_rectangle_with_color
        )

    if mode == "one-by-one":
        for cropped_image_and_render, label, bbox_data in zip(cropped_images_and_renders, labels, page_bboxes_data):
            st.image(image=cropped_image_and_render)
            if isinstance(bbox_data, BboxData):
                if fast_annotation_mode:
                    col1, col2 = st.beta_columns(2)
                else:
                    col1, _ = st.beta_columns([1000, 1])
                with col1:
                    st.markdown(f"'{label}'")
                    st.markdown(label_to_description[label])
                    st.text(f'Bbox: {[bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax]}')
                    if bbox_data.detection_score is not None:
                        st.text(f'Detection score: {bbox_data.detection_score}')
                    if bbox_data.classification_score is not None:
                        st.text(f'Classification score: {bbox_data.classification_score}')
                    if show_top_n:
                        st.text(f'Top-n labels: {bbox_data.labels_top_n}')
                        st.text(f'Top-n scores: {bbox_data.classification_scores_top_n}')
                if fast_annotation_mode:
                    with col2:
                        fast_annotation = st.radio(
                            label='Annotate:',
                            options=[None, 'OK', 'Detection error', 'Classification error'],
                            index=0,
                            key=f"{(bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)}"
                        )
                    page_session = fetch_image_data(image_data=true_image_data)
                    page_session.bboxes_data_fast_annotations[
                        (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)
                    ] = fast_annotation

            elif isinstance(bbox_data, BboxDataMatching):
                true_bbox_data = bbox_data.true_bbox_data
                pred_bbox_data = bbox_data.pred_bbox_data
                if pred_bbox_data is not None:
                    st.markdown(f"Prediction: '{pred_bbox_data.label}'")
                    st.markdown(label_to_description[pred_bbox_data.label])
                    st.text(f'Bbox: {[pred_bbox_data.xmin, pred_bbox_data.ymin, pred_bbox_data.xmax, pred_bbox_data.ymax]}')
                    if pred_bbox_data.detection_score is not None:
                        st.text(f'Detection score: {pred_bbox_data.detection_score}')
                    if pred_bbox_data.classification_score is not None:
                        st.text(f'Classification score: {pred_bbox_data.classification_score}')
                    if show_top_n:
                        st.text(f'Top-n labels: {pred_bbox_data.labels_top_n}')
                        st.text(f'Top-n scores: {pred_bbox_data.classification_scores_top_n}')
                    st.markdown('--')
                if true_bbox_data is not None:
                    st.markdown(f"Ground Truth: '{true_bbox_data.label}'")
                    st.markdown(label_to_description[true_bbox_data.label])
                    st.text(f'Bbox: {[true_bbox_data.xmin, true_bbox_data.ymin, true_bbox_data.xmax, true_bbox_data.ymax]}')
                    st.markdown('--')
                st.markdown(f'Pipeline error type: {bbox_data.get_pipeline_error_type()}')
            if change_annotation is not None:
                change_annotation(
                    bbox_data=bbox_data,
                    max_page=n_split
                )
            st.markdown('----')
    elif mode == "many":
        st.image(image=cropped_images_and_renders, caption=labels)
        st.markdown('----')


def illustrate_n_bboxes_data(
    n_bboxes_data: List[List[BboxData]],
    label_to_base_label_image: Callable[[str], np.ndarray],
    label_to_description: Dict[str, str],
    mode: Literal['many', 'one-by-one'],
    minimum_iou: float = None,
    background_color_a: Tuple[int, int, int, int] = None,
    true_background_color_b: Tuple[int, int, int, int] = None,
    average_maximum_images_per_page: int = 50,
    max_images_size: int = 400,
    bbox_offset: int = 0,
    draw_rectangle_with_color: Tuple[int, int, int] = None,
    change_annotation: Callable[[BboxData], None] = None,
    page_: st.empty = None
):
    bboxes_data = [bbox_data for bboxes_data in n_bboxes_data for bbox_data in bboxes_data]

    if len(bboxes_data) == 0:
        return

    st.markdown(f"Found {len(bboxes_data)} bboxes!")

    n_split = int(np.ceil(len(bboxes_data) / average_maximum_images_per_page))
    splitted_bboxes_data = np.array_split(bboxes_data, n_split)

    if change_annotation is not None:
        page_session = fetch_page_session()
        current_page = page_session.counter
        if current_page > n_split:
            current_page = 1
            page_session.counter = 1
    else:
        current_page = 1

    if n_split >= 2:
        page = st.slider(
            label="",
            min_value=1,
            max_value=n_split,
            value=current_page
        )
        page_bboxes_data = splitted_bboxes_data[page-1]
    else:
        page = 1
        page_bboxes_data = splitted_bboxes_data[0]

    if change_annotation is not None and page != current_page:
        page_session.counter = page

    page_bboxes_data = np.array(page_bboxes_data)
    page_image_paths = np.array([
        bbox_data.image_path for bbox_data in page_bboxes_data
        if bbox_data.image_path is not None
    ])
    unique_image_paths = set(page_image_paths)
    cropped_images_and_renders, labels = [], []
    indexes = []
    for image_path in unique_image_paths:
        indexes_by_image_path = np.where(page_image_paths == image_path)[0]
        bboxes_data_by_image_path = page_bboxes_data[indexes_by_image_path]
        source_image = open_image(image=image_path, open_as_rgb=True)
        if '2020_12_08_validation_v3_mini' in str(image_path):  # DELETE ME LATER
            xmin, ymin, xmax, ymax = eval(str(image_path).split('crop_')[1].split('.jp')[0])
            image_data_with_crop = ImageData(
                image_path=image_path,
                bboxes_data=[BboxData(image_path=image_path, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)]
            )
            source_image = visualize_image_data(image_data=image_data_with_crop)
        cropped_images_and_renders_by_image_path, labels_by_image_path = get_illustrated_bboxes_data(
            source_image=source_image,
            bboxes_data=bboxes_data_by_image_path,
            label_to_base_label_image=label_to_base_label_image,
            background_color_a=background_color_a,
            true_background_color_b=true_background_color_b,
            max_images_size=max_images_size,
            bbox_offset=bbox_offset,
            draw_rectangle_with_color=draw_rectangle_with_color
        )
        indexes.extend(indexes_by_image_path)
        cropped_images_and_renders.extend(cropped_images_and_renders_by_image_path)
        labels.extend(labels_by_image_path)

    inv = np.empty_like(indexes)
    inv[indexes] = np.arange(len(inv), dtype=inv.dtype)
    cropped_images_and_renders = list(np.array(cropped_images_and_renders)[inv])
    labels = list(np.array(labels)[inv])

    if mode == "one-by-one":
        for cropped_image_and_render, label, bbox_data in zip(cropped_images_and_renders, labels, page_bboxes_data):
            st.image(image=cropped_image_and_render)
            st.markdown(label)
            st.markdown(label_to_description[label])
            st.text(f"From image '{bbox_data.image_name}'")
            st.text(f'Bbox: {[bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax]}')
            if change_annotation is not None:
                change_annotation(
                    bbox_data=bbox_data,
                    max_page=n_split
                )
            st.markdown('----')
    elif mode == "many":
        st.image(image=cropped_images_and_renders, caption=labels)
        st.markdown('----')
