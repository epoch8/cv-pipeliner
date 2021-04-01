import os
import json
import fsspec
from typing import Dict, List
from collections import Counter

import numpy as np
from pathy import Pathy

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels, get_n_bboxes_data_filtered_by_labels
from cv_pipeliner.utils.images import (
    get_image_b64, get_label_to_base_label_image
)
from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.dash.data import get_images_data_from_dir
# from cv_pipeliner.utils.streamlit.visualization import (
#     get_illustrated_bboxes_data,
#     illustrate_bboxes_data, illustrate_n_bboxes_data, fetch_page_session
# )
from apps.config import get_cfg_defaults, merge_cfg_from_string

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from yacs.config import CfgNode

from cv_pipeliner.utils.dash.visualization import illustrate_bboxes_data, illustrate_n_bboxes_data

config_file = os.environ['CV_PIPELINER_APP_CONFIG']
cfg, current_config_str = None, None
label_to_base_label_image, label_to_description, label_to_category = None, None, None
ann_class_names = None


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def read_config_file() -> bool:
    global cfg, current_config_str
    global label_to_base_label_image, label_to_description, label_to_category
    global ann_class_names
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

        if cfg.data.ann_class_names is not None:
            with fsspec.open(cfg.data.ann_class_names, 'r') as src:
                ann_class_names = json.load(src)

        return True
    else:
        return False


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "45rem",
    "margin-right": "4rem",
    "padding": "2rem 1rem",
}


sidebar = html.Div(
    children=[
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
        html.Center(
            id='average_maximum_images_per_page_output',
            children=[]
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
        html.Hr(),
        html.Center(
            children=[
                html.P(
                    children='Page 1/1',
                    id='current_page_text'
                )
            ],
            style={
                'font-size': '30px'
            }
        ),
        html.Center(
            children=[
                html.Button(
                    children='Back',
                    id='back_button',
                    style={
                        'width': '100px'
                    }
                ),
                html.Button(
                    children='Next',
                    id='next_button',
                    style={
                        'width': '100px'
                    }
                )
            ],
        )
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
        dcc.Store(id='current_ann_class_names', storage_type='session'),
        dcc.Store(id='filter_by_labels', storage_type='session'),
        dcc.Store(id='current_image_data', storage_type='session'),
        dcc.Store(id='current_page', storage_type='session', data=1),
        dcc.Store(id='maximum_page', storage_type='session', data=1),
    ]
)

main_page_content = html.Div(
    children=[
        html.Div(
            children=[
                html.P(
                    "Classes to find"
                ),
                dcc.Dropdown(
                    id='find_labels',
                    options=[
                        {'label': 'None', 'value': 'None'},
                    ],
                    multi=True
                ),
                html.Br(),
                html.P(
                    "Classes to hide"
                ),
                dcc.Dropdown(
                    id='hide_labels',
                    options=[
                        {'label': 'None', 'value': 'None'},
                    ],
                    multi=True
                ),
                html.Br(),
                html.Div(
                    id='images_data_selected_caption_view',
                    children=[
                        html.P(
                            "Choose an image:"
                        ),
                        dcc.Dropdown(
                            id='images_data_selected_caption',
                            options=[
                                {'label': 'None', 'value': 'None'},
                            ]
                        ),
                    ]
                ),
                html.Br(),
                html.Div(
                    id='page_content_image',
                    children=[]
                ),
                html.Div(
                    id='page_content_bboxes',
                    children=[]
                )
            ],
            style={
                "max-width": "600px",
            }
        )
    ],
    id="main_page_content",
    style=CONTENT_STYLE
)


app.layout = html.Div([
    stores,
    dcc.Location(id="url"),
    sidebar,
    main_page_content
])


@app.callback(
    Output('config', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_config(n):
    read_config_file()
    return current_config_str


@app.callback(
    dash.dependencies.Output('average_maximum_images_per_page_output', 'children'),
    [dash.dependencies.Input('average_maximum_images_per_page', 'value')])
def update_dcc_slider_output(value):
    return [
        html.P(f"{value}")
    ]


@app.callback(
    Output("current_page_text", "children"),
    [
        Input("current_page", "data"),
        Input("maximum_page", "data")
    ]
)
def render_current_page_text(
    current_page,
    maximum_page,
):
    return f"Page {current_page}/{maximum_page}"


@app.callback(
    Output("current_page", "data"),
    [
        Input("current_page", "data"),
        Input("maximum_page", "data"),
        Input("back_button", "n_clicks"),
        Input("next_button", "n_clicks")
    ]
)
def on_click_back_button(
    current_page,
    maximum_page,
    back_button,
    next_button
):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "back_button" in changed_id:
        current_page -= 1
    elif "next_button" in changed_id:
        current_page += 1

    if current_page > maximum_page:
        current_page = 1
    elif current_page < 1:
        current_page = maximum_page

    return current_page


@app.callback(
    Output("images_from", "options"),
    [
        Input("config", "data")
    ]
)
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


@app.callback(
    Output("annotation_filepath", "options"),
    [
        Input("images_from", "value")
    ]
)
def render_annotation_paths(images_from):
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
    Output("images_data_selected_caption_view", "style"),
    [
        Input("view", "value"),
    ]
)
def render_images_data_selected_caption(
    view
):
    if view == "Detection":
        return {}
    else:
        return {'display': 'none'}


@app.callback(
    [
        Output("images_data", "data"),
        Output("images_data_selected_caption", "options")
    ],
    [
        Input("images_from", "value"),
        Input("annotation_filepath", "value")
    ]
)
def get_images_data(images_from: str, annotation_filepath: str):
    images_data = None
    images_data_options = [{'label': 'None', 'value': 'None'}]
    if images_from is not None:
        images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=cfg.data.images_annotation_type,
            images_dir=images_from,
            annotation_filepath=annotation_filepath,
        )
        images_data = sorted(images_data, key=lambda image_data: image_data.image_name)
        idxs_images_data = [(idx, image_data) for idx, image_data in enumerate(images_data)]
        if annotation_success:
            idxs_images_data = sorted(
                idxs_images_data,
                key=lambda pair: len(pair[1].bboxes_data),
                reverse=True
            )
            images_data_captions = [
                f"[{idx}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
                for idx, image_data in idxs_images_data
            ]
        else:
            images_data_captions = [
                f"[{idx}] {image_data.image_name}"
                for idx, image_data in idxs_images_data
            ]
        images_data_options = [
            {
                'label': image_data_caption,
                'value': idx
            } for (idx, _), image_data_caption in zip(idxs_images_data, images_data_captions)
        ]
        images_data = [image_data.asdict() for image_data in images_data]

    return images_data, images_data_options

@app.callback(
    [
        Output("current_image_data", "data"),
        Output("maximum_page", "data"),
        Output("current_ann_class_names", "data"),
    ],
    [
        Input("images_data", "data"),
        Input("images_data_selected_caption", "value"),
        Input("average_maximum_images_per_page", "value"),
        Input("view", "value"),
        Input("find_labels", "value"),
        Input("hide_labels", "value")
    ]
)
def update_current_image_data_and_maximum_page(
    images_data,
    images_data_selected_caption,
    average_maximum_images_per_page,
    view,
    find_labels,
    hide_labels
):
    global ann_class_names
    current_ann_class_names = ann_class_names

    if images_data is None:
        return None, 1, current_ann_class_names

    if view == 'Detection':
        if images_data_selected_caption is None:
            return None, 1, current_ann_class_names
        current_image_data = images_data[images_data_selected_caption]
        image_data = ImageData.from_dict(current_image_data)
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=find_labels,
            include=True
        )
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=hide_labels,
            include=False
        )
        bboxes_data = image_data.bboxes_data
    elif view == 'Classification':
        current_image_data = None
        images_data = [ImageData.from_dict(image_data) for image_data in images_data]
        bboxes_data = [bbox_data for image_data in images_data for bbox_data in image_data.bboxes_data]
        n_bboxes_data = [bboxes_data]
        n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
            n_bboxes_data=n_bboxes_data,
            include=True,
            filter_by_labels=find_labels
        )
        n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
            n_bboxes_data=n_bboxes_data,
            include=False,
            filter_by_labels=hide_labels
        )
        bboxes_data = n_bboxes_data[0]

    maximum_page = max(1, int(np.ceil(len(bboxes_data) / average_maximum_images_per_page)))

    if current_ann_class_names is None:
        labels = [bbox_data.label for bbox_data in bboxes_data]
        labels_counter = Counter(labels)
        current_ann_class_names = sorted(
            set(labels), key=lambda label: labels_counter[label]
        )

    return current_image_data, maximum_page, current_ann_class_names


@app.callback(
    [
        Output("find_labels", "options"),
        Output("hide_labels", "options")
    ],
    [
        Input("current_ann_class_names", "data"),
    ]
)
def update_find_and_hide_labels(
    current_ann_class_names
):
    if current_ann_class_names is None:
        res_options = [{'label': 'None', 'value': 'None'}]
    else:
        res_options = [
            {
                'label': f'{class_name} [{label_to_description[class_name]}]',
                'value': class_name
            }
            for class_name in current_ann_class_names
        ]
    return res_options, res_options


@app.callback(
    Output("page_content_image", "children"),
    [
        Input("current_image_data", "data"),
        Input('use_labels', "value"),
        Input('draw_label_images', "value"),
        Input("view", "value"),
        Input("find_labels", "value"),
        Input("hide_labels", "value")
    ]
)
def render_main_image(
    current_image_data,
    use_labels,
    draw_label_images,
    view,
    find_labels,
    hide_labels
):

    div_children_result = []

    if view == "Detection":
        if current_image_data is None:
            return None
        image_data = ImageData.from_dict(current_image_data)
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=find_labels,
            include=True
        )
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=hide_labels,
            include=False
        )
        use_labels = True if 'true' in use_labels else False
        draw_label_images = True if 'true' in draw_label_images else False
        global label_to_base_label_image
        draw_base_labels_with_given_label_to_base_label_image = (
            label_to_base_label_image if draw_label_images else None
        )

        image = visualize_image_data(
            image_data=image_data,
            use_labels=use_labels,
            draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
        )
        div_children_result = [
            html.Hr(),
            html.Img(
                src=f"data:image/png;base64,{get_image_b64(image, format='png')}",
                style={
                    'max-width': '100%',
                    'height': 'auto'
                }
            ),
            html.Hr(),
            html.Hr()
        ]

    return div_children_result

@app.callback(
    Output("page_content_bboxes", "children"),
    [
        Input("images_data", "data"),
        Input("current_image_data", "data"),
        Input("view", "value"),
        Input("average_maximum_images_per_page", "value"),
        Input("current_page", "data"),
        Input("find_labels", "value"),
        Input("hide_labels", "value")
    ]
)
def render_bboxes(
    images_data,
    current_image_data,
    view,
    average_maximum_images_per_page,
    current_page,
    find_labels,
    hide_labels
):
    if view == 'Detection':
        if current_image_data is None:
            return None
        image_data = ImageData.from_dict(current_image_data)
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=find_labels,
            include=True
        )
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=hide_labels,
            include=False
        )
        return illustrate_bboxes_data(
            true_image_data=image_data,
            label_to_base_label_image=label_to_base_label_image,
            label_to_description=label_to_description,
            background_color_a=[0, 0, 0, 255],
            true_background_color_b=[0, 255, 0, 255],
            bbox_offset=100,
            draw_rectangle_with_color=[0, 255, 0],
            average_maximum_images_per_page=average_maximum_images_per_page,
            current_page=current_page
        )
    elif view == 'Classification':
        images_data = [ImageData.from_dict(image_data) for image_data in images_data]
        bboxes_data = [bbox_data for image_data in images_data for bbox_data in image_data.bboxes_data]
        n_bboxes_data = [bboxes_data]
        n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
            n_bboxes_data=n_bboxes_data,
            include=True,
            filter_by_labels=find_labels
        )
        n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
            n_bboxes_data=n_bboxes_data,
            include=False,
            filter_by_labels=hide_labels
        )
        return illustrate_n_bboxes_data(
            n_bboxes_data=n_bboxes_data,
            label_to_base_label_image=label_to_base_label_image,
            label_to_description=label_to_description,
            background_color_a=[0, 0, 0, 255],
            true_background_color_b=[0, 255, 0, 255],
            bbox_offset=100,
            draw_rectangle_with_color=[0, 255, 0],
            average_maximum_images_per_page=average_maximum_images_per_page,
            current_page=current_page
        )
    return None


if __name__ == "__main__":
    read_config_file()
    app.run_server(debug=True)
