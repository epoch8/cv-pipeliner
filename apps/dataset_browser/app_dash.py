import os
import datetime
import json
import fsspec
import copy
import time
import re
from typing import Dict, List, Literal, Tuple
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from pathy import Pathy

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels, get_n_bboxes_data_filtered_by_labels
from cv_pipeliner.utils.images import (
    get_label_to_base_label_image, open_image,
    draw_n_base_labels_images
)
from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.dash.data import get_images_data_from_dir
# from cv_pipeliner.utils.streamlit.visualization import (
#     get_illustrated_bboxes_data,
#     illustrate_bboxes_data, illustrate_n_bboxes_data, fetch_page_session
# )
from apps.config import get_cfg_defaults, merge_cfg_from_file_fsspec, merge_cfg_from_string

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from yacs.config import CfgNode

config_file = os.environ['CV_PIPELINER_APP_CONFIG']
cfg, current_config_str = None, None
label_to_base_label_image, label_to_description, label_to_category = None, None, None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def read_config_file() -> bool:
    global cfg, current_config_str
    global label_to_base_label_image, label_to_description, label_to_category
    with fsspec.open(config_file, 'r') as src:
        config_str = src.read()
    if config_str != current_config_str:
        if current_config_str is not None:
            app.logger.info("Loading config...")
        else:
            app.logger.info("Config change detected. Reloading...")
        cfg = get_cfg_defaults()
        merge_cfg_from_string(cfg, config_str)
        current_config_str = config_str

        label_to_base_label_image = get_label_to_base_label_image(
            base_labels_images=cfg.data.base_labels_images
        )
        label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)
        label_to_category = get_label_to_description(
            label_to_description_dict=cfg.data.label_to_category,
            default_description='No category'
        )

        return True
    else:
        return False

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "4rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    children=[
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "Images from"
        ),
        dcc.Dropdown(
            id='images_from',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
        ),
        html.P(
            "Annotation filepath"
        ),
        dcc.Dropdown(
            id='annotation_filepath',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
        ),
        html.Hr(),
        html.P(
            "View"
        ),
        dcc.RadioItems(
            id='view',
            options=[
                {'label': 'Detection', 'value': 'Detection'},
                {'label': 'Classification', 'value': 'Classification'},
            ],
            value='Detection',
            labelStyle={'display': 'block'}
        ),
        html.Hr(),
        html.P(
            "Maximum images per page"
        ),
        dcc.Slider(
            id='average_maximum_images_per_page',
            min=1,
            max=100,
            step=1,
            value=20,
            marks={i: {'label': str(i)} for i in range(10, 101, 10)}
        ),
        dcc.Checklist(
            id='use_labels',
            options=[
                {'label': 'Write labels on image', 'value': 'true'},
            ],
            value=['true']
        ),
        dcc.Checklist(
            id='draw_label_images',
            options=[
                {'label': 'Draw base labels images', 'value': 'true'},
            ],
            value=[]
        ),
    ],
    style=SIDEBAR_STYLE
)

stores = html.Div(
    [
        dcc.Interval(
            id='interval-component',
            interval=3*1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Store(id='config', data=current_config_str),
        dcc.Store(id='images_data', storage_type='session'),
        dcc.Store(id='images_data_captions', storage_type='session'),
    ]
)

detection_page_content = html.Div(
    children=[],
    id="detection_page_content",
    style=CONTENT_STYLE
)
classification_page_content = html.Div(
    children=[],
    id="classification_page_content",
    style=CONTENT_STYLE
)

app.layout = html.Div([
    dcc.Location(id="url"), sidebar, stores,
    detection_page_content, classification_page_content
])


@app.callback(Output('config', 'data'),
              Input('interval-component', 'n_intervals'))
def update_config(n):
    read_config_file()
    return current_config_str


@app.callback(Output("images_from", "options"), [Input("config", "data")])
def render_images_dirs(data):
    # with fsspec.open(cfg.data.ann_class_names, 'r') as src:
    #     ann_class_names = json.load(src)
    images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
    image_dir_to_annotation_filepaths = {
        image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
    }
    images_dirs = [image_dir for image_dir in images_dirs if len(image_dir_to_annotation_filepaths[image_dir]) > 0]
    dropdown_options = [{'label': f"../{Pathy(image_dir).name}", 'value': image_dir} for image_dir in images_dirs]
    return dropdown_options


@app.callback(Output("annotation_filepath", "options"), [Input("images_from", "value")])
def render_annotation_paths(images_from):
    # with fsspec.open(cfg.data.ann_class_names, 'r') as src:
    #     ann_class_names = json.load(src)
    if images_from is not None:
        images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
        image_dir_to_annotation_filepaths = {
            image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
        }
        dropdown_options = [
            {
                'label': f"../{Pathy(filepath).name}",
                'value': filepath
            } for filepath in image_dir_to_annotation_filepaths[images_from]
        ]
    else:
        dropdown_options = [{'label': 'None', 'value': 'None'}]

    return dropdown_options


@app.callback(
    Output("images_data", "data"),
    Output("images_data_captions", "data"),
    Input("images_from", "value"),
    Input("annotation_filepath", "value"),
)
def get_images_data(images_from: str, annotation_filepath: str):
    images_data, images_data_captions = None, None
    if images_from is not None and annotation_filepath is not None:
        images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=cfg.data.images_annotation_type,
            images_dir=images_from,
            annotation_filepath=annotation_filepath,
        )
        if annotation_success:
            images_data_captions = [
                f"[{i}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
                for i, image_data in enumerate(images_data)
            ]
        images_data = [image_data.asdict() for image_data in images_data]

    return images_data, images_data_captions


@app.callback(
    [
        Output("detection_page_content", "children"),
        Output("classification_page_content", "children"),
    ],
    [
        Input("images_from", "value"),
        Input("images_data_captions", "data"),
        Input("view", "value")
    ]
)
def render_current_view(
    images_data: Dict,
    images_data_captions: List[str],
    view: str
):
    if images_data is None:
        return [], []
    if view == 'detection':
        st.markdown("Choose an image:")
        images_data_selected_caption = st.selectbox(
            label='Image',
            options=[None] + images_data_captions
        )
        if images_data_selected_caption is not None:
            image_data_index = images_data_captions.index(images_data_selected_caption)
            image_data = images_data[image_data_index]
            st.text(images_data_selected_caption)

            labels = [bbox_data.label for bbox_data in image_data.bboxes_data]
        else:
            image_data = None
            labels = None
    return [], []


    # if view == 'detection':
    #     st.markdown("Choose an image:")
    #     images_data_selected_caption = st.selectbox(
    #         label='Image',
    #         options=[None] + images_data_captions
    #     )
    #     if images_data_selected_caption is not None:
    #         image_data_index = images_data_captions.index(images_data_selected_caption)
    #         image_data = images_data[image_data_index]
    #         st.text(images_data_selected_caption)

    #         labels = [bbox_data.label for bbox_data in image_data.bboxes_data]
    #     else:
    #         image_data = None
    #         labels = None

    # elif view == 'classification':
    #     bboxes_data = [bbox_data for image_data in images_data for bbox_data in image_data.bboxes_data]
    #     labels = [bbox_data.label for bbox_data in bboxes_data]
    # else:
    #     image_data = None
    #     bboxes_data = None

    # st.sidebar.markdown('---')
    # if not annotation_mode:
    #     average_maximum_images_per_page = st.sidebar.slider(
    #         label='Maximum images per page',
    #         min_value=1,
    #         max_value=100,
    #         value=50
    #     )
    # else:
    #     average_maximum_images_per_page = 1

    # if view == 'detection':
    #     st.sidebar.title("Visualization")
    #     use_labels = st.sidebar.checkbox(
    #         'Write labels',
    #         value=True
    #     )
    #     draw_label_images = st.sidebar.checkbox(
    #         'Draw base labels images',
    #         value=False
    #     )
    #     draw_base_labels_with_given_label_to_base_label_image = (
    #         label_to_base_label_image if draw_label_images else None
    #     )

    # class_names_counter = Counter(labels) if labels is not None else {}
    # class_names = sorted(
    #     ann_class_names,
    #     key=lambda x: int(re.sub('\D', '', x)) if re.sub('\D', '', x).isdigit() else 0
    # )

    # classes_col1, classes_col2 = st.beta_columns(2)
    # with classes_col1:
    #     label_to_description['non_default_class'] = "Not from ann_class_names.json"
    #     format_func_filter = lambda class_name: (
    #         f"{class_name} [{label_to_description[class_name]}]"
    #     )
    #     filter_by_labels = st.multiselect(
    #         label="Classes to find",
    #         options=['non_default_class'] + class_names,
    #         default=[],
    #         format_func=format_func_filter
    #     )
    #     st.markdown(f'Classes chosen: {", ".join([format_func_filter(class_name) for class_name in filter_by_labels])}')
    # with classes_col2:
    #     sorted_class_names = sorted(
    #         class_names, key=lambda class_name: class_names_counter.get(class_name, 0), reverse=True
    #     )
    #     show_df = st.checkbox(
    #         label='Show count df'
    #     )
    #     if show_df:
    #         df = pd.DataFrame({
    #             'class_name': sorted_class_names,
    #             'count': list(map(lambda class_name: class_names_counter.get(class_name, 0), sorted_class_names))
    #         })
    #         st.dataframe(data=df, width=1000)
    # filter_by_labels = [
    #     chosen_class_name
    #     for chosen_class_name in filter_by_labels
    # ]
    # if 'non_default_class' in filter_by_labels and labels is not None:
    #     filter_by_labels = sorted(set(labels) - set(ann_class_names))
    #     if len(filter_by_labels) == 0:
    #         filter_by_labels = ['non_default_class']
    # categories_by_class_names = [label_to_category[class_name] for class_name in class_names]
    # categories_counter = Counter(categories_by_class_names)
    # categories = sorted([
    #     category
    #     for category in set(label_to_category.values())
    #     if categories_counter[category] > 0
    # ])



# @app.callback(Output("dd-output-container", "options"), [Input("demo-dropdown", "value")])
# def render_page_content(value: str):
#     return f'You have selected "{value}"'


if __name__ == "__main__":
    read_config_file()
    app.run_server(debug=True)
