import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
from cv_pipeliner.metrics.image_data_matching import (
    ImageDataMatching, BboxDataMatching, pairwise_intersection_over_union, intersection_over_union
)
from cv_pipeliner.core.data import ImageData

EPS = 1e-9

def _as_xy(poly: np.ndarray) -> np.ndarray:
    a = np.asarray(poly)
    if a.ndim == 3 and a.shape[1] == 1 and a.shape[2] == 2:
        a = a[:, 0, :]
    return a.astype(np.float64).reshape(-1, 2)

def _oriented_area_cv(poly_xy: np.ndarray) -> float:
    return float(cv2.contourArea(poly_xy.astype(np.float32), oriented=True))

def _polygon_area_xy(poly_xy: np.ndarray) -> float:
    return abs(_oriented_area_cv(poly_xy))

def _is_ccw(poly_xy: np.ndarray) -> bool:
    return _oriented_area_cv(poly_xy) > 0

def _cross(o, a, b) -> float:
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_triangle(p, a, b, c) -> bool:
    c1 = _cross(a, b, p); c2 = _cross(b, c, p); c3 = _cross(c, a, p)
    has_neg = (c1 < -EPS) or (c2 < -EPS) or (c3 < -EPS)
    has_pos = (c1 >  EPS) or (c2 >  EPS) or (c3 >  EPS)
    return not (has_neg and has_pos)

def _is_convex(a, b, c) -> bool:
    return _cross(a, b, c) > EPS

def _orient_ccw(poly_xy: np.ndarray) -> np.ndarray:
    return poly_xy if _is_ccw(poly_xy) else poly_xy[::-1].copy()

def _triangulate_earclip(poly_xy: np.ndarray) -> List[np.ndarray]:
    n = len(poly_xy)
    if n < 3:
        return []
    P = _orient_ccw(poly_xy)
    idx = list(range(n))
    tris: List[np.ndarray] = []

    def ear(i: int) -> bool:
        i0, i1, i2 = idx[(i-1) % len(idx)], idx[i], idx[(i+1) % len(idx)]
        a, b, c = P[i0], P[i1], P[i2]
        if not _is_convex(a, b, c):
            return False
        tri = np.array([a, b, c], dtype=np.float64)
        for j in idx:
            if j in (i0, i1, i2):
                continue
            if _point_in_triangle(P[j], *tri):
                return False
        return True

    i = 0
    guard = 0
    while len(idx) > 3 and guard < 10000:
        if ear(i):
            i0, i1, i2 = idx[(i-1) % len(idx)], idx[i], idx[(i+1) % len(idx)]
            tris.append(np.array([P[i0], P[i1], P[i2]], dtype=np.float64))
            del idx[i]
            i = 0
        else:
            i = (i + 1) % len(idx)
        guard += 1
    if len(idx) == 3:
        tris.append(np.array([P[idx[0]], P[idx[1]], P[idx[2]]], dtype=np.float64))
    return tris

# exact intersection area of two simple polygons via triangle×triangle
def _triangles_intersection_area_cv2(tri1: np.ndarray, tri2: np.ndarray) -> float:
    area, _poly = cv2.intersectConvexConvex(tri1.astype(np.float32), tri2.astype(np.float32))
    return float(area) if area is not None else 0.0

def _simple_polygons_intersection_area(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    if len(A_xy) < 3 or len(B_xy) < 3:
        return 0.0
    minAx, minAy = A_xy.min(axis=0); maxAx, maxAy = A_xy.max(axis=0)
    minBx, minBy = B_xy.min(axis=0); maxBx, maxBy = B_xy.max(axis=0)
    if maxAx < minBx - EPS or maxBx < minAx - EPS or maxAy < minBy - EPS or maxBy < minAy - EPS:
        return 0.0
    TA = _triangulate_earclip(A_xy)
    TB = _triangulate_earclip(B_xy)
    if not TA or not TB:
        return 0.0
    inter = 0.0
    # Предварительные bbox’ы треугольников для отсечки
    bboxA = [ (t.min(axis=0), t.max(axis=0)) for t in TA ]
    bboxB = [ (t.min(axis=0), t.max(axis=0)) for t in TB ]
    for i, ta in enumerate(TA):
        mina, maxa = bboxA[i]
        for j, tb in enumerate(TB):
            minb, maxb = bboxB[j]
            if maxa[0] < minb[0]-EPS or maxb[0] < mina[0]-EPS or maxa[1] < minb[1]-EPS or maxb[1] < mina[1]-EPS:
                continue
            inter += _triangles_intersection_area_cv2(ta, tb)
    return inter

def iou_rings_with_holes(rings_a: List[np.ndarray], rings_b: List[np.ndarray]) -> float:
    A = []
    B = []
    for p in rings_a or []:
        poly = _as_xy(p)
        if len(poly) >= 3 and _polygon_area_xy(poly) > EPS:
            A.append(poly)
    for p in rings_b or []:
        poly = _as_xy(p)
        if len(poly) >= 3 and _polygon_area_xy(poly) > EPS:
            B.append(poly)

    if not A and not B:
        return 0.0

    areaA_signed = sum(_oriented_area_cv(p) for p in A)
    areaB_signed = sum(_oriented_area_cv(p) for p in B)
    areaA = abs(areaA_signed)  # on case of «pure» orientation inversion of components
    areaB = abs(areaB_signed)

    # If both are empty
    if areaA <= EPS and areaB <= EPS:
        return 0.0

    # double sum of pairwise intersections with signs of rings
    inter = 0.0
    if A and B:
        signsA = [np.sign(_oriented_area_cv(p)) for p in A]
        signsB = [np.sign(_oriented_area_cv(p)) for p in B]
        # bbox’es for clipping
        bboxA = [ (p.min(axis=0), p.max(axis=0)) for p in A ]
        bboxB = [ (p.min(axis=0), p.max(axis=0)) for p in B ]
        for i, pa in enumerate(A):
            mina, maxa = bboxA[i]
            sa = signsA[i]
            for j, pb in enumerate(B):
                minb, maxb = bboxB[j]
                if maxa[0] < minb[0]-EPS or maxb[0] < mina[0]-EPS or maxa[1] < minb[1]-EPS or maxb[1] < mina[1]-EPS:
                    continue
                sb = signsB[j]
                inter_ij = _simple_polygons_intersection_area(pa, pb)
                if inter_ij > 0.0:
                    inter += (sa * sb) * inter_ij

    union = areaA + areaB - inter
    if union <= EPS:
        # complete match of zero area — define as 0
        return 0.0
    # clip on [0,1] from numerical noise
    iou = inter / union
    return float(max(0.0, min(1.0, iou)))

# NxK matrix mask IoU for lists BboxData
def pairwise_mask_iou_matrix_with_holes(true_bboxes, pred_bboxes) -> Optional[np.ndarray]:
    """
    Returns NxK IoU matrix by masks (considering holes) without rasterization.
    If some bbox mask is not a list of polygons -> return None (let the calling code solve fallback).
    """
    N, K = len(true_bboxes), len(pred_bboxes)
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float64)

    def to_rings(bb) -> Optional[List[np.ndarray]]:
        m = bb.mask
        if isinstance(m, list):
            rings = []
            for poly in m:
                arr = _as_xy(poly)
                if len(arr) >= 3 and _polygon_area_xy(arr) > EPS:
                    rings.append(arr)
            return rings
        return None

    A = [to_rings(bb) for bb in true_bboxes]
    B = [to_rings(bb) for bb in pred_bboxes]
    if any(a is None for a in A) or any(b is None for b in B):
        return None

    out = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        for j in range(K):
            out[i, j] = iou_rings_with_holes(A[i], B[j])
    return out


@dataclass(init=False)
class ImageDataMatchingSegmentation(ImageDataMatching):
    mask_minimum_iou: Optional[float] = None

    def __init__(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float,
        extra_bbox_label: str = None,
        bboxes_data_matchings: List[BboxDataMatching] = None,
        mask_minimum_iou: Optional[float] = None,
    ):
        self.true_image_data = true_image_data
        self.pred_image_data = pred_image_data
        self.minimum_iou = minimum_iou
        self.extra_bbox_label = extra_bbox_label
        self.mask_minimum_iou = mask_minimum_iou

        if bboxes_data_matchings is None:
            self.bboxes_data_matchings = self._get_bboxes_data_matchings(
                true_image_data=true_image_data,
                pred_image_data=pred_image_data,
                minimum_iou=minimum_iou,
                extra_bbox_label=extra_bbox_label,
                mask_minimum_iou=mask_minimum_iou,
            )
        else:
            self.bboxes_data_matchings = bboxes_data_matchings

    def _get_bboxes_data_matchings(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float,
        extra_bbox_label: str,
        mask_minimum_iou: Optional[float] = None,   # <== НОВОЕ
    ) -> List[BboxDataMatching]:
        true_bboxes_data = true_image_data.bboxes_data
        pred_bboxes_data = pred_image_data.bboxes_data
        remained_pred_bboxes_data = set(range(len(pred_bboxes_data)))
        bboxes_data_matchings: List[BboxDataMatching] = []

            # sanity: уникальность координат и валидность
        for tag, bboxes_list in [("true", true_bboxes_data), ("pred", pred_bboxes_data)]:
            bboxes_coords = set()
            for bbox_data in bboxes_list:
                assert all(x is not None for x in [bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax])
                if not (bbox_data.xmin <= bbox_data.xmax and bbox_data.ymin <= bbox_data.ymax):
                    continue
                if bbox_data.coords in bboxes_coords:
                    pass
                bboxes_coords.add(bbox_data.coords)

        if len(true_bboxes_data) > 0 and len(pred_bboxes_data) > 0:
            pairwise_ious = pairwise_intersection_over_union(true_bboxes_data, pred_bboxes_data)

            # --- НОВОЕ: матрица IoU по маскам и масочный порог ---
            pairwise_mask_ious = None
            if mask_minimum_iou is not None:
                pairwise_mask_ious = pairwise_mask_iou_matrix_with_holes(true_bboxes_data, pred_bboxes_data)
                # если масок не оказалось вообще (None), то форсим провал по масочному порогу
                if pairwise_mask_ious is None:
                    # nobody matches the mask condition
                    pairwise_ious[:] = -1.0
                else:
                    # turn off pairs that do not pass the mask threshold
                    mask_fail = pairwise_mask_ious < float(mask_minimum_iou)
                    pairwise_ious = np.where(mask_fail, -1.0, pairwise_ious)

            for idx, true_bbox_data in enumerate(true_bboxes_data):
                best_pred_bbox_column = int(np.argmax(pairwise_ious[idx, :]))
                if pairwise_ious[idx, best_pred_bbox_column] >= minimum_iou:
                    pairwise_ious[:, best_pred_bbox_column] = -1
                    remained_pred_bboxes_data.discard(best_pred_bbox_column)
                    matched = BboxDataMatching(
                        true_bbox_data=true_bbox_data,
                        pred_bbox_data=pred_bboxes_data[best_pred_bbox_column],
                        extra_bbox_label=extra_bbox_label,
                        _iou=intersection_over_union(true_bbox_data, pred_bboxes_data[best_pred_bbox_column]),
                    )
                    # проставим mask_iou, если считали матрицу
                    if pairwise_mask_ious is not None:
                        matched._mask_iou = float(pairwise_mask_ious[idx, best_pred_bbox_column])
                    bboxes_data_matchings.append(matched)
                else:
                    # не найден подходящий пред-бокс
                    bboxes_data_matchings.append(
                        BboxDataMatching(true_bbox_data=true_bbox_data, pred_bbox_data=None, extra_bbox_label=extra_bbox_label)
                    )

            for pred_col in remained_pred_bboxes_data:
                bboxes_data_matchings.append(
                    BboxDataMatching(
                        true_bbox_data=None,
                        pred_bbox_data=pred_bboxes_data[pred_col],
                        extra_bbox_label=extra_bbox_label,
                    )
                )

        elif len(true_bboxes_data) > 0:
            for true_bbox_data in true_bboxes_data:
                bboxes_data_matchings.append(
                    BboxDataMatching(true_bbox_data=true_bbox_data, pred_bbox_data=None, extra_bbox_label=extra_bbox_label)
                )
        elif len(pred_bboxes_data) > 0:
            for pred_bbox_data in pred_bboxes_data:
                bboxes_data_matchings.append(
                    BboxDataMatching(true_bbox_data=None, pred_bbox_data=pred_bbox_data, extra_bbox_label=extra_bbox_label)
                )

        return bboxes_data_matchings

