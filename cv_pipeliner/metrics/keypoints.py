from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from cv_pipeliner.core.data import BboxData, ImageData, KeypointVisibility

df_keypoints_metrics_columns = ["value"]

_IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def _require_ultralytics():
    try:
        from ultralytics.utils.metrics import OKS_SIGMA, Metric, ap_per_class, kpt_iou
        from ultralytics.utils.ops import xyxy2xywh
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "get_df_keypoints_metrics requires ultralytics. Install with: pip install 'cv_pipeliner[torch]'"
        ) from e
    return OKS_SIGMA, Metric, ap_per_class, kpt_iou, xyxy2xywh


def _bbox_xyxy(bbox_data: BboxData) -> np.ndarray:
    return np.array([bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax], dtype=np.float32)


def _keypoints_nx3(bbox_data: BboxData) -> np.ndarray:
    keypoints = np.asarray(bbox_data.keypoints, dtype=np.float32).reshape(-1, 2)
    if len(keypoints) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if bbox_data.keypoints_visibility is not None:
        visibility = np.asarray([int(v) for v in bbox_data.keypoints_visibility], dtype=np.float32)
        if len(visibility) != len(keypoints):
            raise ValueError(
                f"keypoints_visibility length ({len(visibility)}) != keypoints length ({len(keypoints)})"
            )
    else:
        visibility = np.full(len(keypoints), float(KeypointVisibility.LABELED_AND_VISIBLE), dtype=np.float32)
    return np.concatenate([keypoints, visibility[:, None]], axis=1)


def _class_id(label: Optional[str], class_names: Optional[List[str]]) -> int:
    if class_names is None:
        return 0
    if label is None:
        raise ValueError("bbox label is None but class_names were provided")
    return class_names.index(label)


def _extract_instances(
    images_data: List[ImageData],
    class_names: Optional[List[str]],
    *,
    is_pred: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Per image: boxes (N,4) xyxy, keypoints (N,K,3), classes (N,), confs (N,)."""
    boxes_list, kpts_list, cls_list, conf_list = [], [], [], []
    for image_data in images_data:
        boxes, kpts, classes, confs = [], [], [], []
        for bbox_data in image_data.bboxes_data:
            keypoints = _keypoints_nx3(bbox_data)
            if len(keypoints) == 0:
                continue
            boxes.append(_bbox_xyxy(bbox_data))
            kpts.append(keypoints)
            classes.append(_class_id(bbox_data.label, class_names))
            if is_pred:
                confs.append(1.0 if bbox_data.detection_score is None else float(bbox_data.detection_score))
            else:
                confs.append(1.0)
        if boxes:
            boxes_list.append(np.stack(boxes, axis=0))
            kpts_list.append(np.stack(kpts, axis=0))
            cls_list.append(np.asarray(classes, dtype=np.float32))
            conf_list.append(np.asarray(confs, dtype=np.float32))
        else:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            kpts_list.append(np.zeros((0, 0, 3), dtype=np.float32))
            cls_list.append(np.zeros((0,), dtype=np.float32))
            conf_list.append(np.zeros((0,), dtype=np.float32))
    return boxes_list, kpts_list, cls_list, conf_list


def _infer_nkpt(true_kpts_list: List[np.ndarray], pred_kpts_list: List[np.ndarray]) -> int:
    for kpts in list(true_kpts_list) + list(pred_kpts_list):
        if kpts.shape[0] > 0:
            return int(kpts.shape[1])
    return 0


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor = _IOU_THRESHOLDS,
) -> torch.Tensor:
    """Match predictions to GT by OKS/IoU (ultralytics BaseValidator.match_predictions)."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


def get_df_keypoints_metrics(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    class_names: Optional[List[str]] = None,
    sigma: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Returns YOLO-style pose keypoints metrics (precision, recall, mAP50, mAP50-95)
    using Object Keypoint Similarity (OKS), via ultralytics primitives.
    """
    OKS_SIGMA, Metric, ap_per_class, kpt_iou, xyxy2xywh = _require_ultralytics()

    assert len(true_images_data) == len(pred_images_data)

    true_boxes, true_kpts, true_cls, _ = _extract_instances(true_images_data, class_names, is_pred=False)
    pred_boxes, pred_kpts, pred_cls, pred_conf = _extract_instances(pred_images_data, class_names, is_pred=True)

    nkpt = _infer_nkpt(true_kpts, pred_kpts)
    support = int(sum(len(c) for c in true_cls))

    empty_df = pd.DataFrame(
        {
            "images_support": len(true_images_data),
            "support": [support],
            "pose_P": [0.0],
            "pose_R": [0.0],
            "pose_mAP50": [0.0],
            "pose_mAP50_95": [0.0],
        },
        dtype=object,
    ).T
    empty_df.columns = df_keypoints_metrics_columns

    if nkpt == 0:
        return empty_df

    if sigma is None:
        sigma_arr = np.asarray(OKS_SIGMA, dtype=np.float32) if nkpt == 17 else (np.ones(nkpt, dtype=np.float32) / nkpt)
    else:
        sigma_arr = np.asarray(sigma, dtype=np.float32).flatten()
        if len(sigma_arr) != nkpt:
            raise ValueError(f"sigma length ({len(sigma_arr)}) must equal number of keypoints ({nkpt})")

    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    for i in range(len(true_images_data)):
        gt_k = true_kpts[i]
        pr_k = pred_kpts[i]
        gt_cls = true_cls[i]
        pr_cls = pred_cls[i]
        pr_conf = pred_conf[i]
        gt_boxes = true_boxes[i]

        if gt_k.shape[0] > 0 and gt_k.shape[1] != nkpt:
            raise ValueError(f"Inconsistent GT keypoint count: expected {nkpt}, got {gt_k.shape[1]}")
        if pr_k.shape[0] > 0 and pr_k.shape[1] != nkpt:
            raise ValueError(f"Inconsistent pred keypoint count: expected {nkpt}, got {pr_k.shape[1]}")

        n_pred = pr_cls.shape[0]
        conf_list.append(pr_conf)
        pred_cls_list.append(pr_cls)
        target_cls_list.append(gt_cls)

        if gt_cls.shape[0] == 0 or n_pred == 0:
            tp_list.append(np.zeros((n_pred, len(_IOU_THRESHOLDS)), dtype=bool))
            continue

        # Pad empty-keypoint images already skipped; reshape (0,0,3) → skip above
        area = xyxy2xywh(torch.from_numpy(gt_boxes))[:, 2:].prod(1) * 0.53
        iou = kpt_iou(
            torch.from_numpy(gt_k),
            torch.from_numpy(pr_k),
            sigma=sigma_arr.tolist(),
            area=area,
        )
        tp_list.append(
            _match_predictions(
                torch.from_numpy(pr_cls),
                torch.from_numpy(gt_cls),
                iou,
            )
            .cpu()
            .numpy()
        )

    tp = np.concatenate(tp_list, 0) if tp_list else np.zeros((0, len(_IOU_THRESHOLDS)), dtype=bool)
    conf = np.concatenate(conf_list, 0) if conf_list else np.zeros((0,), dtype=np.float32)
    pred_cls_all = np.concatenate(pred_cls_list, 0) if pred_cls_list else np.zeros((0,), dtype=np.float32)
    target_cls_all = np.concatenate(target_cls_list, 0) if target_cls_list else np.zeros((0,), dtype=np.float32)

    if target_cls_all.shape[0] == 0 and pred_cls_all.shape[0] == 0:
        return empty_df

    names = (
        {i: name for i, name in enumerate(class_names)}
        if class_names is not None
        else {0: "object"}
    )
    results = ap_per_class(
        tp,
        conf,
        pred_cls_all,
        target_cls_all,
        plot=False,
        names=names,
        prefix="Pose",
    )[2:]
    metric = Metric()
    metric.nc = len(names)
    metric.update(results)
    pose_p, pose_r, pose_map50, pose_map = metric.mean_results()

    df_keypoints_metrics = pd.DataFrame(
        {
            "images_support": len(true_images_data),
            "support": [support],
            "pose_P": [float(pose_p)],
            "pose_R": [float(pose_r)],
            "pose_mAP50": [float(pose_map50)],
            "pose_mAP50_95": [float(pose_map)],
        },
        dtype=object,
    ).T
    df_keypoints_metrics.columns = df_keypoints_metrics_columns
    return df_keypoints_metrics
