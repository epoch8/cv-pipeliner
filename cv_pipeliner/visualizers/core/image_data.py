import collections
import hashlib
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import imutils
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils.images import rotate_point
from cv_pipeliner.utils.images_datas import (
    flatten_additional_bboxes_data_in_image_data,
    get_image_data_filtered_by_labels,
)

# Taken from object_detection.utils.visualization_utils
STANDARD_COLORS = [
    "AliceBlue",
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "BlanchedAlmond",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]
STANDARD_COLORS_RGB = ["Red", "Green", "Blue", "Yellow"]
STANDARD_COLORS_RGB = STANDARD_COLORS_RGB + [x for x in STANDARD_COLORS if x not in STANDARD_COLORS_RGB]


def draw_mask_on_image(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    if len(mask.shape) == 2:
        mask = np.array(Image.fromarray(mask).convert("RGB"))
    norm_mask = mask.astype(np.float32) / 255.0
    np_where = np.where(norm_mask > 0)
    if len(np_where[0]) > 0:
        colored_mask_np = np.zeros_like(mask, dtype=np.float32)
        recolor_value = np.array(color)
        colored_mask_np[np_where] = norm_mask[np_where] * np.tile(recolor_value, len(np_where[0]) // 3)
        colored_mask = colored_mask_np.astype(np.uint8)
        colored_np_where = np.where(colored_mask > 0)
        image[colored_np_where] = (1 - alpha) * image[colored_np_where] + alpha * colored_mask[colored_np_where]
    return image


# Taken from object_detection.utils.visualization_utils
def draw_bounding_box_on_image(
    image: Image,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    keypoints: np.ndarray = [],
    angle: int = 0,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
    keypoints_radius: int = 5,
    fontsize: int = 24,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    angle: angle of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                        (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    points = [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
    rotated_points = [rotate_point(x=x, y=y, cx=left, cy=top, angle=angle) for (x, y) in points]
    if thickness > 0:
        draw.line(rotated_points, width=thickness, fill=color)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_bboxes = [font.getbbox(ds) for ds in display_str_list]
    display_str_heights = [f_ymax - f_ymin for (_, f_ymin, _, f_ymax) in display_str_bboxes]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        f_xmin, f_ymin, f_xmax, f_ymax = font.getbbox(display_str)
        text_width, text_height = f_xmax - f_xmin, f_ymax - f_ymin
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

    keypoints = np.array(keypoints).reshape(len(keypoints), 2)
    for idx, (x, y) in enumerate(keypoints):
        draw.pieslice(
            [(x - keypoints_radius, y - keypoints_radius), (x + keypoints_radius, y + keypoints_radius)],
            start=0,
            end=360,
            fill=STANDARD_COLORS_RGB[idx % len(STANDARD_COLORS_RGB)],
        )


# Taken from object_detection.utils.visualization_utils
def visualize_boxes_and_labels_on_image_array(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    angles: List[int],
    labels: List[str],
    scores: List[float],
    k_keypoints: List[List[Tuple[int, int]]],
    use_normalized_coordinates=False,
    groundtruth_box_visualization_color="black",
    known_labels: List[str] = [],
    skip_scores=False,
    skip_labels=False,
    keypoints_radius: int = 5,
    fontsize: int = 24,
    thickness: int = 4,
    label_to_color: Optional[Dict[str, str]] = None,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      angles: a numpy array of shape [N].
      labels: a numpy array of shape [N]. Note that class indices are 1-based.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      known_labels: a list of known labels. If given, bboxes colors will be chosen by this list.
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    scores = np.array(scores)

    if len(known_labels) > 0:
        known_labels = set(known_labels)
    label_to_id = {label: int(hashlib.md5(str(label).encode()).hexdigest(), 16) for id_, label in enumerate(labels)}
    bbox_to_display_str = collections.defaultdict(list)
    bbox_to_color = collections.defaultdict(str)

    for i in range(len(bboxes)):
        bbox = tuple(bboxes[i].tolist())
        if skip_labels and len(known_labels) == 0:
            bbox_to_color[bbox] = groundtruth_box_visualization_color
        else:
            display_str = ""
            if not skip_labels:
                display_str = str(labels[i])
            if not skip_scores:
                if not display_str:
                    display_str = f"{round(100*scores[i])}%"
                else:
                    display_str = f"{display_str}: {round(100*scores[i])}%"
            bbox_to_display_str[bbox].append(display_str)
            if len(known_labels) > 0 and labels[i] in known_labels:
                if label_to_color is not None:
                    bbox_to_color[bbox] = label_to_color[labels[i]]
                else:
                    bbox_to_color[bbox] = STANDARD_COLORS[label_to_id[labels[i]] % len(STANDARD_COLORS)]
            else:
                bbox_to_color[bbox] = groundtruth_box_visualization_color

    # Draw all boxes onto image.
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    for bbox, angle, keypoints in zip(bboxes, angles, k_keypoints):
        bbox = tuple(bbox.tolist())
        xmin, ymin, xmax, ymax = bbox
        draw_bounding_box_on_image(
            image=image_pil,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            keypoints=keypoints,
            angle=angle,
            color=bbox_to_color[bbox],
            thickness=thickness,
            display_str_list=bbox_to_display_str[bbox],
            use_normalized_coordinates=use_normalized_coordinates,
            keypoints_radius=keypoints_radius,
            fontsize=fontsize,
        )
    image = np.array(image_pil)
    return image


def draw_label_image(
    image: np.ndarray, base_label_image: np.ndarray, bbox_data: BboxData, inplace: bool = False
) -> np.ndarray:
    if not inplace:
        image = image.copy()

    bbox_data_size = max(bbox_data.xmax - bbox_data.xmin, bbox_data.ymax - bbox_data.ymin)
    resize = min(int(bbox_data_size / 1.5), int(max(image.shape) / 20))

    height, width, _ = base_label_image.shape
    if height <= width:
        label_image = imutils.resize(base_label_image, width=resize)
    else:
        label_image = imutils.resize(base_label_image, height=resize)

    x_offset = bbox_data.xmin - 20
    y_offset = bbox_data.ymax - label_image.shape[0]

    y_min, y_max = y_offset, y_offset + label_image.shape[0]
    x_min, x_max = x_offset, x_offset + label_image.shape[1]

    # Ensure that label image is inside image boundaries
    if y_max > image.shape[0]:
        y_min -= y_max - image.shape[0]
        y_max = image.shape[0]

    if x_max > image.shape[1]:
        x_min -= x_max - image.shape[1]
        x_max = image.shape[1]

    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    alpha_label_image = label_image[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_label_image

    for channel in range(0, 3):
        image[y_min:y_max, x_min:x_max, channel] = (
            alpha_label_image * label_image[:, :, channel] + alpha_image * image[y_min:y_max, x_min:x_max, channel]
        )

    if not inplace:
        return image


def visualize_image_data(
    image_data: ImageData,
    include_labels: bool = False,
    score_type: Literal["detection", "classification"] = None,
    filter_by_labels: List[str] = None,
    known_labels: Optional[List[str]] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None,
    keypoints_radius: int = 5,
    include_additional_bboxes_data: bool = False,
    additional_bboxes_data_depth: Optional[int] = None,
    include_keypoints: bool = False,
    include_mask: bool = False,
    mask_alpha: float = 0.5,
    label_to_color: Optional[Dict[str, str]] = None,
    fontsize: int = 24,
    thickness: int = 4,
    thumbnail_size: Optional[Union[int, Tuple[int, int]]] = None,
    return_as_pil_image: bool = False,
    default_color: str = "lime",
    exif_transpose: bool = False,
    xmin_offset: Union[int, float] = 0,
    ymin_offset: Union[int, float] = 0,
    xmax_offset: Union[int, float] = 0,
    ymax_offset: Union[int, float] = 0,
    use_labels: Optional[bool] = None,
) -> Union[np.ndarray, Image.Image]:
    if use_labels is not None:
        print("WARNING: argument use_labels= is deprecated and will be removed. Use include_labels= instead.")
    if thumbnail_size is not None:
        from cv_pipeliner.utils.images_datas import thumbnail_image_data

        image_data = thumbnail_image_data(image_data, thumbnail_size)

    image_data = get_image_data_filtered_by_labels(image_data=image_data, filter_by_labels=filter_by_labels)
    image = image_data.open_image(exif_transpose=exif_transpose)
    if include_additional_bboxes_data:
        bboxes_data = flatten_additional_bboxes_data_in_image_data(
            image_data, additional_bboxes_data_depth=additional_bboxes_data_depth
        ).bboxes_data
    else:
        bboxes_data = image_data.bboxes_data
    labels = [bbox_data.label for bbox_data in bboxes_data] + [image_data.label]
    if known_labels is None:
        known_labels = list(set(labels))
    k_keypoints = [bbox_data.keypoints for bbox_data in bboxes_data]
    bboxes = np.array(
        [
            bbox_data.coords_with_offset(xmin_offset, ymin_offset, xmax_offset, ymax_offset, source_image=image)
            for bbox_data in bboxes_data
        ]
    )
    angles = np.array([0.0 for _ in bboxes_data])
    if score_type == "detection":
        scores = np.array([bbox_data.detection_score for bbox_data in bboxes_data])
        skip_scores = False
    elif score_type == "classification":
        scores = np.array([bbox_data.classification_score for bbox_data in bboxes_data])
        skip_scores = False
    else:
        scores = None
        skip_scores = True

    image = visualize_boxes_and_labels_on_image_array(
        image=image,
        bboxes=bboxes,
        angles=angles,
        scores=scores,
        k_keypoints=k_keypoints,
        labels=labels,
        use_normalized_coordinates=False,
        skip_scores=skip_scores,
        skip_labels=not include_labels,
        groundtruth_box_visualization_color=default_color,
        known_labels=known_labels,
        keypoints_radius=keypoints_radius,
        fontsize=fontsize,
        thickness=thickness,
        label_to_color=label_to_color,
    )
    if include_keypoints and len(image_data.keypoints) > 0:
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        for idx, (x, y) in enumerate(image_data.keypoints):
            draw.pieslice(
                [(x - keypoints_radius, y - keypoints_radius), (x + keypoints_radius, y + keypoints_radius)],
                start=0,
                end=360,
                fill=STANDARD_COLORS_RGB[idx % len(STANDARD_COLORS_RGB)],
            )
        image = np.array(image_pil)

    if include_mask:
        label_to_id = {label: int(hashlib.md5(str(label).encode()).hexdigest(), 16) for id_, label in enumerate(labels)}
        for bbox_data in bboxes_data:
            if label_to_color is None:
                bbox_color = ImageColor.getrgb(STANDARD_COLORS[label_to_id[bbox_data.label] % len(STANDARD_COLORS)])
            else:
                bbox_color = ImageColor.getrgb(label_to_color.get(bbox_data.label, default_color))
            bbox_mask_np = bbox_data.open_mask(exif_transpose=exif_transpose)
            image = draw_mask_on_image(
                image=image,
                mask=bbox_mask_np,
                color=bbox_color,
                alpha=mask_alpha,
            )
        mask_np = image_data.open_mask(exif_transpose=exif_transpose, include_bboxes_data=False)
        if label_to_color is None:
            color = ImageColor.getrgb(STANDARD_COLORS[label_to_id[image_data.label] % len(STANDARD_COLORS)])
        else:
            color = ImageColor.getrgb(label_to_color.get(image_data.label, default_color))
        image = draw_mask_on_image(
            image=image,
            mask=mask_np,
            color=color,
            alpha=mask_alpha,
        )

    if draw_base_labels_with_given_label_to_base_label_image is not None:
        for bbox_data in bboxes_data:
            base_label_image = draw_base_labels_with_given_label_to_base_label_image(bbox_data.label)
            draw_label_image(image=image, base_label_image=base_label_image, bbox_data=bbox_data, inplace=True)

    if return_as_pil_image:
        return Image.fromarray(image)

    return image
