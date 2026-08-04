import collections
import hashlib
import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import imutils
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from cv_pipeliner.core.data import BboxData, ImageData, KeypointVisibility
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


def _normalize_keypoints_visibility(
    keypoints_visibility: Optional[List[Union[int, KeypointVisibility]]],
    n_keypoints: int,
) -> List[Optional[KeypointVisibility]]:
    if keypoints_visibility is None:
        return [None] * n_keypoints
    if len(keypoints_visibility) != n_keypoints:
        raise ValueError(
            f"Len keypoints_visibility={len(keypoints_visibility)} not equal to len keypoints={n_keypoints}"
        )
    return [None if value is None else KeypointVisibility(int(value)) for value in keypoints_visibility]


def _normalize_keypoints_scores(
    keypoints_scores: Optional[List[Optional[float]]],
    n_keypoints: int,
) -> List[Optional[float]]:
    if keypoints_scores is None:
        return [None] * n_keypoints
    if len(keypoints_scores) != n_keypoints:
        raise ValueError(f"Len keypoints_scores={len(keypoints_scores)} not equal to len keypoints={n_keypoints}")
    return [None if score is None else float(score) for score in keypoints_scores]


def _normalize_keypoints_labels(
    keypoints_labels: Optional[List[Optional[str]]],
    n_keypoints: int,
) -> List[Optional[str]]:
    if keypoints_labels is None:
        return [None] * n_keypoints
    if len(keypoints_labels) != n_keypoints:
        raise ValueError(f"Len keypoints_labels={len(keypoints_labels)} not equal to len keypoints={n_keypoints}")
    return [None if label is None else str(label) for label in keypoints_labels]


def _resolve_keypoint_color(
    idx: int,
    label: Optional[str],
    *,
    label_to_color: Optional[Dict[str, str]] = None,
    default_color: Optional[str] = None,
) -> str:
    """Resolve a keypoint draw color.

    Preference when ``label_to_color`` is provided:
    1. ``label_to_color[label]`` for the keypoint label
    2. ``default_color`` (parent bbox / image color)
    3. palette color by keypoint index

    Without ``label_to_color``, keep the legacy index palette.
    """
    if label_to_color is not None:
        if label is not None and label in label_to_color:
            return label_to_color[label]
        if default_color is not None:
            return default_color
    return STANDARD_COLORS_RGB[idx % len(STANDARD_COLORS_RGB)]


def _rects_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Axis-aligned rects as (left, top, right, bottom)."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _keypoint_circle_rect(x: float, y: float, radius: float) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds of a drawn keypoint circle (slightly padded)."""
    pad = 1.0
    r = float(radius) + pad
    return (x - r, y - r, x + r, y + r)


def _bbox_display_str_rects(
    *,
    left: float,
    top: float,
    bottom: float,
    display_str_list: List[str],
    font,
) -> List[Tuple[float, float, float, float]]:
    """Caption boxes for bbox labels — same geometry as ``draw_bounding_box_on_image``."""
    if not display_str_list:
        return []
    display_str_bboxes = [font.getbbox(ds) for ds in display_str_list]
    display_str_heights = [f_ymax - f_ymin for (_, f_ymin, _, f_ymax) in display_str_bboxes]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = float(top)
    else:
        text_bottom = float(bottom + total_display_str_height)
    rects: List[Tuple[float, float, float, float]] = []
    for display_str in display_str_list[::-1]:
        f_xmin, f_ymin, f_xmax, f_ymax = font.getbbox(display_str)
        text_width, text_height = f_xmax - f_xmin, f_ymax - f_ymin
        margin = float(np.ceil(0.05 * text_height))
        rects.append((left, text_bottom - text_height - 2 * margin, left + text_width, text_bottom))
        text_bottom -= text_height - 2 * margin
    return rects


def _caption_center(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    left, top, right, bottom = rect
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _clamp_caption_rect(
    left: float,
    text_bottom: float,
    *,
    text_width: float,
    box_h: float,
    image_width: float,
    image_height: float,
) -> Tuple[float, float, float, float, float]:
    left = min(max(0.0, left), max(0.0, image_width - text_width))
    text_bottom = min(max(box_h, text_bottom), image_height)
    return (left, text_bottom - box_h, left + text_width, text_bottom, text_bottom)


def _place_keypoint_caption_box(
    *,
    x: float,
    y: float,
    keypoints_radius: int,
    text_width: float,
    text_height: float,
    margin: float,
    image_width: float,
    image_height: float,
    placed_rects: List[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float, float]:
    """Pick the closest free caption slot around a keypoint.

    Candidates stay near the point (right/left/above/below rings) instead of
    stacking far away. Avoids ``placed_rects`` (other captions, circles, bbox labels).
    Returns ``(left, top, right, bottom, text_bottom)``.
    """
    box_h = text_height + 2 * margin
    gap = float(keypoints_radius + 2)
    half_w = text_width / 2.0

    # (left, text_bottom) anchors, preferred order later sorted by distance.
    raw_anchors: List[Tuple[float, float]] = [
        (x + gap, y + box_h / 2.0),  # right, vertically centered
        (x + gap, y - gap),  # right-above
        (x + gap, y + gap + box_h),  # right-below
        (x - gap - text_width, y + box_h / 2.0),  # left, centered
        (x - gap - text_width, y - gap),  # left-above
        (x - gap - text_width, y + gap + box_h),  # left-below
        (x - half_w, y - gap),  # above
        (x - half_w, y + gap + box_h),  # below
    ]
    # Small local stack around the preferred right/left slots (keep close).
    for k in (1, 2, 3):
        dy = k * box_h
        raw_anchors.extend(
            [
                (x + gap, y - gap - dy),
                (x + gap, y + gap + box_h + dy),
                (x - gap - text_width, y - gap - dy),
                (x - gap - text_width, y + gap + box_h + dy),
            ]
        )

    candidates: List[Tuple[float, float, float, float, float]] = []
    seen = set()
    for left0, tb0 in raw_anchors:
        placed = _clamp_caption_rect(
            left0,
            tb0,
            text_width=text_width,
            box_h=box_h,
            image_width=image_width,
            image_height=image_height,
        )
        key = (round(placed[0], 1), round(placed[1], 1), round(placed[2], 1), round(placed[3], 1))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(placed)

    def distance(item: Tuple[float, float, float, float, float]) -> float:
        cx, cy = _caption_center(item[:4])
        return (cx - x) ** 2 + (cy - y) ** 2

    candidates.sort(key=distance)

    free = [
        item
        for item in candidates
        if not any(_rects_overlap(item[:4], placed) for placed in placed_rects)
    ]
    if free:
        return free[0]
    # Fallback: stay as close as possible even if slightly overlapping.
    return candidates[0]


def _axis_aligned_segment_hits_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rect: Tuple[float, float, float, float],
    *,
    pad: float = 2.0,
) -> bool:
    """True if a horizontal/vertical segment intersects an expanded axis-aligned rect."""
    left, top, right, bottom = rect
    left -= pad
    top -= pad
    right += pad
    bottom += pad
    if abs(y0 - y1) <= 1e-6:
        y = y0
        if y < top or y > bottom:
            return False
        seg_l, seg_r = (x0, x1) if x0 <= x1 else (x1, x0)
        return not (seg_r < left or seg_l > right)
    if abs(x0 - x1) <= 1e-6:
        x = x0
        if x < left or x > right:
            return False
        seg_t, seg_b = (y0, y1) if y0 <= y1 else (y1, y0)
        return not (seg_b < top or seg_t > bottom)
    # Fallback for rare non-axis segments: AABB overlap of the segment bbox with rect.
    seg_l, seg_r = (x0, x1) if x0 <= x1 else (x1, x0)
    seg_t, seg_b = (y0, y1) if y0 <= y1 else (y1, y0)
    return not (seg_r < left or seg_l > right or seg_b < top or seg_t > bottom)


def _polyline_text_hit_count(
    points: List[Tuple[float, float]],
    text_rects: List[Tuple[float, float, float, float]],
    *,
    pad: float = 2.0,
) -> int:
    """How many polyline segments intersect at least one text rect."""
    if len(points) < 2 or not text_rects:
        return 0
    hits = 0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if abs(x0 - x1) < 0.5 and abs(y0 - y1) < 0.5:
            continue
        if any(_axis_aligned_segment_hits_rect(x0, y0, x1, y1, rect, pad=pad) for rect in text_rects):
            hits += 1
    return hits


def _polyline_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _build_leader_polyline(
    x: float,
    y: float,
    radius: float,
    *waypoints: Tuple[float, float],
) -> List[Tuple[float, float]]:
    if not waypoints:
        return []
    first = waypoints[0]
    dx, dy = first[0] - x, first[1] - y
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    start = (x + dx / length * float(radius), y + dy / length * float(radius))
    points: List[Tuple[float, float]] = [start]
    for point in waypoints:
        prev = points[-1]
        if abs(point[0] - prev[0]) > 0.5 or abs(point[1] - prev[1]) > 0.5:
            points.append(point)
    return points


def _leader_clear_channels(
    text_rects: List[Tuple[float, float, float, float]],
    *,
    margin: float = 8.0,
) -> Tuple[List[float], List[float]]:
    """X/Y coordinates that run just outside known text boxes (for Z-detours)."""
    clear_xs: List[float] = []
    clear_ys: List[float] = []
    for left, top, right, bottom in text_rects:
        clear_xs.extend([left - margin, right + margin])
        clear_ys.extend([top - margin, bottom + margin])
    return clear_xs, clear_ys


def _keypoint_caption_leader_polyline(
    x: float,
    y: float,
    radius: float,
    caption_rect: Tuple[float, float, float, float],
    text_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Pick an L/Z leader that prefers not to cross other caption/bbox text."""
    left, top, right, bottom = caption_rect
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    attaches = [
        (left, mid_y),
        (right, mid_y),
        (mid_x, top),
        (mid_x, bottom),
        (left, top),
        (right, top),
        (left, bottom),
        (right, bottom),
    ]
    candidates: List[List[Tuple[float, float]]] = []
    for ax, ay in attaches:
        # Simple L routes.
        candidates.append(_build_leader_polyline(x, y, radius, (ax, y), (ax, ay)))
        candidates.append(_build_leader_polyline(x, y, radius, (x, ay), (ax, ay)))

    avoid = list(text_rects or [])
    clear_xs, clear_ys = _leader_clear_channels(avoid)
    # Also try channels just outside the own caption (useful when blockers sit under it).
    clear_xs.extend([left - 8.0, right + 8.0])
    clear_ys.extend([top - 8.0, bottom + 8.0])

    for ax, ay in attaches:
        for clear_x in clear_xs:
            # Z: horizontal out → vertical clear channel → horizontal into attach.
            candidates.append(_build_leader_polyline(x, y, radius, (clear_x, y), (clear_x, ay), (ax, ay)))
        for clear_y in clear_ys:
            # Z: vertical out → horizontal clear channel → vertical into attach.
            candidates.append(_build_leader_polyline(x, y, radius, (x, clear_y), (ax, clear_y), (ax, ay)))

    # Preferred default (same as before): closer axis first.
    cx, cy = mid_x, mid_y
    if abs(cx - x) >= abs(cy - y):
        preferred_attach = (left, mid_y) if cx >= x else (right, mid_y)
        preferred = _build_leader_polyline(x, y, radius, (preferred_attach[0], y), preferred_attach)
    else:
        preferred_attach = (mid_x, top) if cy >= y else (mid_x, bottom)
        preferred = _build_leader_polyline(x, y, radius, (x, preferred_attach[1]), preferred_attach)
    candidates.insert(0, preferred)

    stroke_pad = 3.0
    scored = [
        (_polyline_text_hit_count(points, avoid, pad=stroke_pad), _polyline_length(points), i, points)
        for i, points in enumerate(candidates)
        if len(points) >= 2
    ]
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def _draw_keypoint_caption_leader(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    keypoints_radius: int,
    caption_rect: Tuple[float, float, float, float],
    color: str,
    text_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> None:
    """Draw an L/Z leader in the keypoint/caption color."""
    points = _keypoint_caption_leader_polyline(
        x,
        y,
        float(keypoints_radius),
        caption_rect,
        text_rects=text_rects,
    )
    if len(points) < 2:
        return
    width = max(2, keypoints_radius // 3)
    draw.line(points, fill=color, width=width)


def draw_keypoints_on_image(
    draw: ImageDraw.ImageDraw,
    keypoints: np.ndarray,
    *,
    keypoints_radius: int = 5,
    keypoints_visibility: Optional[List[Union[int, KeypointVisibility]]] = None,
    keypoints_scores: Optional[List[Optional[float]]] = None,
    keypoints_labels: Optional[List[Optional[str]]] = None,
    include_keypoint_scores: bool = False,
    include_keypoints_labels: bool = False,
    include_keypoints_visibility: bool = True,
    keypoints_fontsize: int = 12,
    label_to_color: Optional[Dict[str, str]] = None,
    default_color: Optional[str] = None,
    occupied_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> None:
    """Draw keypoints with optional COCO visibility, labels, and confidence scores.

    When ``include_keypoints_visibility`` is True (default):
    - ``None`` / missing: draw filled (legacy behavior)
    - ``NOT_LABELED`` (0): skip
    - ``LABELED_NOT_VISIBLE`` (1): outline only
    - ``LABELED_AND_VISIBLE`` (2): filled

    When ``include_keypoints_visibility`` is False, every keypoint is drawn
    filled as-is (visibility flags are ignored).

    Colors: ``label_to_color`` is applied to ``keypoints_labels`` when present;
    otherwise ``default_color`` (parent box/image), else index palette.

    Captions use the same readability pattern as bbox labels: colored background,
    black text, and a nearby free slot around the point (not far stacks). Every
    caption is linked with an L/Z leader in the keypoint color that
    prefers routes which do not cross other caption/bbox text.
    Pass a shared ``occupied_rects`` list to reserve already-drawn bbox labels
    (and to accumulate new caption/circle rects).
    """
    keypoints = np.asarray(keypoints).reshape(-1, 2)
    visibility = (
        _normalize_keypoints_visibility(keypoints_visibility, len(keypoints))
        if include_keypoints_visibility
        else [None] * len(keypoints)
    )
    scores = _normalize_keypoints_scores(keypoints_scores, len(keypoints))
    labels = _normalize_keypoints_labels(keypoints_labels, len(keypoints))

    try:
        font = ImageFont.truetype("arial.ttf", max(1, keypoints_fontsize))
    except IOError:
        font = ImageFont.load_default()

    image = getattr(draw, "_image", None)
    if image is not None:
        image_width, image_height = image.size
    else:
        image_width, image_height = 10**9, 10**9

    # Placement avoids circles + text; leaders only avoid foreign text boxes.
    # ``occupied_rects`` is text-only (bbox labels + captions) across draw calls —
    # never put keypoint circles there, or leader routing treats them as blockers
    # and may prefer a short path that still crosses real text.
    text_rects: List[Tuple[float, float, float, float]] = list(occupied_rects or [])
    avoid_rects: List[Tuple[float, float, float, float]] = list(occupied_rects or [])
    drawable: List[Tuple[int, float, float, Optional[KeypointVisibility], Optional[float], Optional[str]]] = []
    for idx, ((x, y), visibility_flag, score, label) in enumerate(zip(keypoints, visibility, scores, labels)):
        if visibility_flag is KeypointVisibility.NOT_LABELED:
            continue
        fx, fy = float(x), float(y)
        drawable.append((idx, fx, fy, visibility_flag, score, label))
        circle_rect = _keypoint_circle_rect(fx, fy, keypoints_radius)
        avoid_rects.append(circle_rect)

    for idx, x, y, visibility_flag, score, label in drawable:
        color = _resolve_keypoint_color(
            idx,
            label,
            label_to_color=label_to_color,
            default_color=default_color,
        )
        point_bbox = [(x - keypoints_radius, y - keypoints_radius), (x + keypoints_radius, y + keypoints_radius)]
        if visibility_flag is KeypointVisibility.LABELED_NOT_VISIBLE:
            draw.arc(point_bbox, start=0, end=360, fill=color, width=max(1, keypoints_radius // 2))
        else:
            draw.pieslice(point_bbox, start=0, end=360, fill=color)
        caption_parts = []
        if include_keypoints_labels and label is not None:
            caption_parts.append(label)
        if include_keypoint_scores and score is not None:
            caption_parts.append(f"{round(100 * score)}%")
        if not caption_parts:
            continue

        display_str = " ".join(caption_parts)
        f_xmin, f_ymin, f_xmax, f_ymax = font.getbbox(display_str)
        text_width, text_height = f_xmax - f_xmin, f_ymax - f_ymin
        margin = float(np.ceil(0.05 * text_height))
        left, top, right, bottom, text_bottom = _place_keypoint_caption_box(
            x=x,
            y=y,
            keypoints_radius=keypoints_radius,
            text_width=float(text_width),
            text_height=float(text_height),
            margin=margin,
            image_width=float(image_width),
            image_height=float(image_height),
            placed_rects=avoid_rects,
        )
        caption_rect = (left, top, right, bottom)

        _draw_keypoint_caption_leader(
            draw,
            x=x,
            y=y,
            keypoints_radius=keypoints_radius,
            caption_rect=caption_rect,
            color=color,
            text_rects=text_rects,
        )
        draw.rectangle([(left, top), (right, bottom)], fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )

        avoid_rects.append(caption_rect)
        text_rects.append(caption_rect)
        if occupied_rects is not None:
            occupied_rects.append(caption_rect)



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
    keypoints_fontsize: Optional[int] = None,
    keypoints_visibility: Optional[List[Union[int, KeypointVisibility]]] = None,
    keypoints_scores: Optional[List[Optional[float]]] = None,
    keypoints_labels: Optional[List[Optional[str]]] = None,
    include_keypoint_scores: bool = False,
    include_keypoints_labels: bool = False,
    include_keypoints_visibility: bool = True,
    label_to_color: Optional[Dict[str, str]] = None,
    occupied_rects: Optional[List[Tuple[float, float, float, float]]] = None,
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
    occupied_rects: mutable list of already-occupied caption/keypoint regions;
        bbox labels are appended, then passed to keypoint caption placement.
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
    caption_rects = _bbox_display_str_rects(
        left=float(left),
        top=float(top),
        bottom=float(bottom),
        display_str_list=list(display_str_list),
        font=font,
    )
    for display_str, (caption_left, caption_top, caption_right, caption_bottom) in zip(
        display_str_list[::-1], caption_rects
    ):
        f_xmin, f_ymin, f_xmax, f_ymax = font.getbbox(display_str)
        text_height = f_ymax - f_ymin
        margin = float(np.ceil(0.05 * text_height))
        draw.rectangle([(caption_left, caption_top), (caption_right, caption_bottom)], fill=color)
        draw.text(
            (caption_left + margin, caption_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )

    if occupied_rects is None:
        occupied_rects = list(caption_rects)
    else:
        occupied_rects.extend(caption_rects)

    resolved_keypoints_fontsize = max(8, fontsize // 2) if keypoints_fontsize is None else keypoints_fontsize
    draw_keypoints_on_image(
        draw,
        keypoints,
        keypoints_radius=keypoints_radius,
        keypoints_visibility=keypoints_visibility,
        keypoints_scores=keypoints_scores,
        keypoints_labels=keypoints_labels,
        include_keypoint_scores=include_keypoint_scores,
        include_keypoints_labels=include_keypoints_labels,
        include_keypoints_visibility=include_keypoints_visibility,
        keypoints_fontsize=resolved_keypoints_fontsize,
        label_to_color=label_to_color,
        default_color=color,
        occupied_rects=occupied_rects,
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
    keypoints_fontsize: Optional[int] = None,
    thickness: int = 4,
    label_to_color: Optional[Dict[str, str]] = None,
    k_keypoints_visibility: Optional[List[Optional[List[Union[int, KeypointVisibility]]]]] = None,
    k_keypoints_scores: Optional[List[Optional[List[Optional[float]]]]] = None,
    k_keypoints_labels: Optional[List[Optional[List[Optional[str]]]]] = None,
    include_keypoint_scores: bool = False,
    include_keypoints_labels: bool = False,
    include_keypoints_visibility: bool = True,
    occupied_rects: Optional[List[Tuple[float, float, float, float]]] = None,
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

    if k_keypoints_visibility is None:
        k_keypoints_visibility = [None] * len(k_keypoints)
    if k_keypoints_scores is None:
        k_keypoints_scores = [None] * len(k_keypoints)
    if k_keypoints_labels is None:
        k_keypoints_labels = [None] * len(k_keypoints)

    # Draw all boxes onto image.
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    if occupied_rects is None:
        occupied_rects = []
    for bbox, angle, keypoints, keypoints_visibility, keypoints_scores, keypoints_labels in zip(
        bboxes, angles, k_keypoints, k_keypoints_visibility, k_keypoints_scores, k_keypoints_labels
    ):
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
            keypoints_fontsize=keypoints_fontsize,
            keypoints_visibility=keypoints_visibility,
            keypoints_scores=keypoints_scores,
            keypoints_labels=keypoints_labels,
            include_keypoint_scores=include_keypoint_scores,
            include_keypoints_labels=include_keypoints_labels,
            include_keypoints_visibility=include_keypoints_visibility,
            label_to_color=label_to_color,
            occupied_rects=occupied_rects,
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
    include_keypoint_scores: bool = False,
    include_keypoints_labels: bool = False,
    include_keypoints_visibility: bool = True,
    include_mask: bool = False,
    mask_alpha: float = 0.5,
    label_to_color: Optional[Dict[str, str]] = None,
    fontsize: int = 24,
    keypoints_fontsize: Optional[int] = None,
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
    k_keypoints_visibility = [bbox_data.keypoints_visibility for bbox_data in bboxes_data]
    k_keypoints_scores = [bbox_data.keypoints_scores for bbox_data in bboxes_data]
    k_keypoints_labels = [bbox_data.keypoints_labels for bbox_data in bboxes_data]
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

    occupied_rects: List[Tuple[float, float, float, float]] = []
    image = visualize_boxes_and_labels_on_image_array(
        image=image,
        bboxes=bboxes,
        angles=angles,
        scores=scores,
        k_keypoints=k_keypoints,
        k_keypoints_visibility=k_keypoints_visibility,
        k_keypoints_scores=k_keypoints_scores,
        k_keypoints_labels=k_keypoints_labels,
        include_keypoint_scores=include_keypoint_scores,
        include_keypoints_labels=include_keypoints_labels,
        include_keypoints_visibility=include_keypoints_visibility,
        labels=labels,
        use_normalized_coordinates=False,
        skip_scores=skip_scores,
        skip_labels=not include_labels,
        groundtruth_box_visualization_color=default_color,
        known_labels=known_labels,
        keypoints_radius=keypoints_radius,
        fontsize=fontsize,
        keypoints_fontsize=keypoints_fontsize,
        thickness=thickness,
        label_to_color=label_to_color,
        occupied_rects=occupied_rects,
    )
    if include_keypoints and len(image_data.keypoints) > 0:
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        resolved_keypoints_fontsize = max(8, fontsize // 2) if keypoints_fontsize is None else keypoints_fontsize
        draw_keypoints_on_image(
            draw,
            image_data.keypoints,
            keypoints_radius=keypoints_radius,
            keypoints_visibility=image_data.keypoints_visibility,
            keypoints_scores=image_data.keypoints_scores,
            keypoints_labels=image_data.keypoints_labels,
            include_keypoint_scores=include_keypoint_scores,
            include_keypoints_labels=include_keypoints_labels,
            include_keypoints_visibility=include_keypoints_visibility,
            keypoints_fontsize=resolved_keypoints_fontsize,
            label_to_color=label_to_color,
            default_color=(
                label_to_color.get(image_data.label, default_color)
                if label_to_color is not None and image_data.label is not None
                else default_color
            ),
            occupied_rects=occupied_rects,
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
