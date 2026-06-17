from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Bbox = Tuple[int, int, int, int]
Score = float
Label = str
Keypoint = Tuple[int, int]
Keypoints = List[Keypoint]
Mask = List[List[Tuple[int, int]]]


@dataclass
class DetectionResult:
    bboxes: List[List[Bbox]]
    keypoints: List[List[Keypoints]]
    masks: List[List[Mask]]
    detection_scores: List[List[Score]]
    labels_top_n: Optional[List[List[List[Label]]]] = None
    classification_scores_top_n: Optional[List[List[List[Score]]]] = None

    @classmethod
    def from_tuple(cls, output: tuple) -> "DetectionResult":
        (
            bboxes,
            keypoints,
            masks,
            detection_scores,
            labels_top_n,
            classification_scores_top_n,
        ) = output
        return cls(
            bboxes=bboxes,
            keypoints=keypoints,
            masks=masks,
            detection_scores=detection_scores,
            labels_top_n=labels_top_n,
            classification_scores_top_n=classification_scores_top_n,
        )

    def as_tuple(self) -> tuple:
        return (
            self.bboxes,
            self.keypoints,
            self.masks,
            self.detection_scores,
            self.labels_top_n,
            self.classification_scores_top_n,
        )


@dataclass
class ClassificationResult:
    labels_top_n: List[List[Label]]
    scores_top_n: List[List[Score]]

    @classmethod
    def from_tuple(cls, output: tuple) -> "ClassificationResult":
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
