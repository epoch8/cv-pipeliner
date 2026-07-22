from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

Bbox = Tuple[int, int, int, int]
Score = float
Label = str
Keypoint = Tuple[int, int]
Keypoints = List[Keypoint]
Mask = List[List[Tuple[int, int]]]
KeypointsScores = List[float]


@dataclass
class RawDetectionImage:
    """Per-image backend-native detection tensors (pre-postprocess)."""

    bboxes: Any
    keypoints: Any
    detection_scores: Any
    class_ids: Any
    masks: Any = field(default_factory=lambda: [[]])
    keypoints_scores: Any = None


@dataclass
class RawDetectionPredictions:
    """Batch of backend-native detection tensors (pre-postprocess)."""

    bboxes: List[Any]
    keypoints: List[Any]
    detection_scores: List[Any]
    class_ids: List[Any]
    masks: Optional[List[Any]] = None
    keypoints_scores: Optional[List[Any]] = None

    def __len__(self) -> int:
        return len(self.bboxes)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> RawDetectionImage:
        masks = self.masks[index] if self.masks is not None else [[]]
        keypoints_scores = self.keypoints_scores[index] if self.keypoints_scores is not None else None
        return RawDetectionImage(
            bboxes=self.bboxes[index],
            keypoints=self.keypoints[index],
            detection_scores=self.detection_scores[index],
            class_ids=self.class_ids[index],
            masks=masks,
            keypoints_scores=keypoints_scores,
        )

    @classmethod
    def from_images(cls, images: Sequence[RawDetectionImage]) -> RawDetectionPredictions:
        has_keypoints_scores = any(image.keypoints_scores is not None for image in images)
        return cls(
            bboxes=[image.bboxes for image in images],
            keypoints=[image.keypoints for image in images],
            detection_scores=[image.detection_scores for image in images],
            class_ids=[image.class_ids for image in images],
            masks=[image.masks for image in images],
            keypoints_scores=[image.keypoints_scores for image in images] if has_keypoints_scores else None,
        )


@dataclass
class PostprocessedDetectionImage:
    """Per-image detection output after backend-specific postprocess."""

    bboxes: Any
    keypoints: Any
    detection_scores: Any
    labels_top_n: Any
    classification_scores_top_n: Any


@dataclass
class DetectionResult:
    bboxes: List[List[Bbox]]
    keypoints: List[List[Keypoints]]
    masks: List[List[Mask]]
    detection_scores: List[List[Score]]
    labels_top_n: Optional[List[List[List[Label]]]] = None
    classification_scores_top_n: Optional[List[List[List[Score]]]] = None
    keypoints_scores: Optional[List[List[Optional[KeypointsScores]]]] = None

    @classmethod
    def from_tuple(cls, output: tuple) -> DetectionResult:
        """Backward-compatible constructor from the legacy positional tuple."""
        bboxes = output[0]
        keypoints = output[1]
        masks = output[2]
        detection_scores = output[3]
        labels_top_n = output[4] if len(output) > 4 else None
        classification_scores_top_n = output[5] if len(output) > 5 else None
        keypoints_scores = output[6] if len(output) > 6 else None
        return cls(
            bboxes=bboxes,
            keypoints=keypoints,
            masks=masks,
            detection_scores=detection_scores,
            labels_top_n=labels_top_n,
            classification_scores_top_n=classification_scores_top_n,
            keypoints_scores=keypoints_scores,
        )

    def as_tuple(self) -> tuple:
        return (
            self.bboxes,
            self.keypoints,
            self.masks,
            self.detection_scores,
            self.labels_top_n,
            self.classification_scores_top_n,
            self.keypoints_scores,
        )


@dataclass
class ClassificationResult:
    labels_top_n: List[List[Label]]
    scores_top_n: List[List[Score]]

    @classmethod
    def from_tuple(cls, output: tuple) -> ClassificationResult:
        labels_top_n, scores_top_n = output
        return cls(labels_top_n=labels_top_n, scores_top_n=scores_top_n)

    def as_tuple(self) -> tuple:
        return self.labels_top_n, self.scores_top_n


@dataclass
class EmbeddingResult:
    embeddings: List[np.ndarray]


@dataclass
class KeypointsResult:
    keypoints: List[Keypoints]
