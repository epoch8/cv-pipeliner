from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.inferencers.pipeline import PipelineInferencer

from cv_pipeliner.tracking.opencv_tracker import OpenCVTracker
from cv_pipeliner.tracking.sort_tracker import Sort


@dataclass
class FrameResult:
    label: str
    track_id: int
    ready_at_frame: int


class RealTimeInferencer:
    def __init__(
        self,
        pipeline_inferencer: PipelineInferencer,
        fps: float,
        detection_delay: int,
        classification_delay: int,
        batch_size: int = 16
    ):
        self.detection_inferencer = DetectionInferencer(pipeline_inferencer.model.detection_model)
        self.classification_inferencer = ClassificationInferencer(pipeline_inferencer.model.classification_model)

        self.fps = fps
        self.detection_delay = detection_delay
        self.classification_delay = classification_delay
        self.current_frame_idx = 0

        self.detection_delay_frames = int(round(self.detection_delay * fps / 1000))
        self.classification_delay_frames = int(round(self.classification_delay * fps / 1000))

        self.sort_tracker = Sort()
        self.opencv_tracker = None

        self.current_ready_frames_queue: List[FrameResult] = []

    def update_sort_tracker(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        scores: List[float] = None
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        if scores is None:
            # SORT requires detection scores, use 1 while tracking
            scores = np.ones(bboxes.shape[0])
        tracked_bboxes_sort = self.sort_tracker.update(
            np.column_stack([bboxes, scores])
        )
        tracked_bboxes_sort = tracked_bboxes_sort.round().astype(int)
        tracked_bboxes = tracked_bboxes_sort[:, :-1]
        tracked_bboxes = np.clip(tracked_bboxes, 0, tracked_bboxes.max(initial=0))
        tracked_ids = tracked_bboxes_sort[:, -1]
        return tracked_bboxes, tracked_ids

    def run_tracking_on_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        tracked_bboxes_optical = self.opencv_tracker.update(frame)
        tracked_bboxes, tracked_ids = self.update_sort_tracker(
            bboxes=tracked_bboxes_optical
        )
        return tracked_bboxes, tracked_ids

    def run_pipeline_on_frame(
        self,
        frame: np.ndarray,
        detection_score_threshold: float,
        batch_size: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        frame = frame.copy()
        image_data = ImageData(image=frame)
        image_data_gen = BatchGeneratorImageData([image_data], batch_size=batch_size,
                                                 use_not_caught_elements_as_last_batch=True)

        pred_image_data = self.detection_inferencer.predict(
            images_data_gen=image_data_gen,
            score_threshold=detection_score_threshold
        )[0]
        bboxes = np.array([
            (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)
            for bbox_data in pred_image_data.bboxes_data
        ])
        detection_scores = np.array([bbox_data.detection_score for bbox_data in pred_image_data.bboxes_data])

        self.opencv_tracker = OpenCVTracker(bboxes, frame)
        tracked_bboxes, tracked_ids = self.update_sort_tracker(
            bboxes=bboxes, scores=detection_scores
        )

        current_tracks_ids = [frame_result.track_id for frame_result in self.current_ready_frames_queue]
        current_not_tracked_items_idxs = [
            idx for idx, tracked_id in enumerate(tracked_ids)
            if tracked_id not in current_tracks_ids
        ]
        if current_not_tracked_items_idxs:
            current_not_tracked_bboxes = tracked_bboxes[current_not_tracked_items_idxs]
            current_not_tracked_ids = tracked_ids[current_not_tracked_items_idxs]
            bboxes_data = [
                BboxData(image=frame, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                for (xmin, ymin, xmax, ymax) in current_not_tracked_bboxes
            ]
            bboxes_data_gen = BatchGeneratorBboxData([bboxes_data], batch_size=batch_size,
                                                     use_not_caught_elements_as_last_batch=True)
            pred_bboxes_data = self.classification_inferencer.predict(bboxes_data_gen)[0]

            for bbox_data, tracked_id in zip(pred_bboxes_data, current_not_tracked_ids):
                frame_result = FrameResult(
                    label=bbox_data.label,
                    track_id=tracked_id,
                    ready_at_frame=self.current_frame_idx
                )
                self.current_ready_frames_queue.append(frame_result)

        return tracked_bboxes, tracked_ids

    def predict_on_frame(
        self,
        frame: np.ndarray,
        detection_score_threshold: float,
        batch_size: int
    ) -> List[BboxData]:
        if self.current_frame_idx % self.detection_delay_frames == 0:
            tracked_bboxes, tracked_ids = self.run_pipeline_on_frame(
                frame=frame,
                detection_score_threshold=detection_score_threshold,
                batch_size=batch_size
            )
        else:
            tracked_bboxes, tracked_ids = self.run_tracking_on_frame(frame)

        ready_frames_at_the_moment = [
            ready_frame
            for ready_frame in self.current_ready_frames_queue
            if ready_frame.ready_at_frame <= self.current_frame_idx
        ]
        ready_tracks_ids_at_the_moment = [
            ready_frame.track_id for ready_frame in ready_frames_at_the_moment
        ]

        bboxes_data = []
        for bbox, track_id in zip(tracked_bboxes, tracked_ids):
            if track_id not in ready_tracks_ids_at_the_moment:
                continue
            xmin, ymin, xmax, ymax = bbox
            ready_frame = ready_frames_at_the_moment[ready_tracks_ids_at_the_moment.index(track_id)]
            bboxes_data.append(BboxData(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                label=ready_frame.label
            ))
        self.current_frame_idx += 1
        return bboxes_data
