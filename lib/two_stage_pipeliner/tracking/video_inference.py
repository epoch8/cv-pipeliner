from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Literal

import numpy as np
import tempfile

import imutils
import imageio

from two_stage_pipeliner.core.data import ImageData, BboxData
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.inferencers.classification import ClassificationInferencer

from two_stage_pipeliner.tracking.opencv_tracker import OpenCVTracker
from two_stage_pipeliner.tracking.sort_tracker import Sort
from two_stage_pipeliner.tracking.visualization import draw_overlay


@dataclass
class FrameResult:
    bbox_data: BboxData
    track_id: int
    ready_at_frame: int


class VideoInferer:
    def __init__(
        self,
        pipeline_inferencer: ClassificationInferencer,
        label_images_path: List[Union[str, Path]],
        frame_width: int = 640,
        frame_height: int = 1152,
        batch_size: int = 16
    ):
        self.detection_inferencer = DetectionInferencer(
            pipeline_inferencer.model_spec.detection_model
        )
        self.classification_inferencer = ClassificationInferencer(
            pipeline_inferencer.model_spec.classification_model
        )
        self.label_images = {
            label: imageio.imread(f"{label_images_path}/{label}.png")
            for label in self.clf_model.classes
        }

        self.sort_tracker = None
        self.opencv_tracker = None

        self.frame_width = frame_width
        self.frame_height = frame_height

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
        frame_pixels: np.ndarray,
        frame_idx: int,
        fps: float,
        detection_delay: int,
        classification_delay: int,
        detection_score_threshold: float,
        classification_score_threshold: float,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        frame = frame_pixels.copy()
        image_data = ImageData(image=frame)
        image_data_gen = BatchGeneratorImageData([image_data], batch_size=1)

        pred_image_data = self.detection_inferencer.predict(
            images_data_gen=image_data_gen,
            detection_score_threshold=detection_score_threshold
        )[0]
        bboxes = [
            (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)
            for bbox_data in pred_image_data.bboxes_data
        ]
        detection_scores = [bbox_data.detection_score for bbox_data in pred_image_data.bboxes_data]

        self.opencv_tracker = OpenCVTracker(bboxes, frame)
        tracked_bboxes, tracked_ids = self.update_sort_tracker(
            bboxes=bboxes, scores=detection_scores
        )

        detection_delay_frames = int(round(detection_delay * fps / 1000))
        classification_delay_frames = int(round(classification_delay * fps / 1000))

        current_tracks_ids = [frame_result.track_id for frame_result in self.current_ready_frames_queue]
        current_not_tracked_items_idxs = [
            idx for idx, tracked_id in enumerate(tracked_ids)
            if tracked_id not in current_tracks_ids
        ]
        original_idxs = np.range(len(tracked_ids))[current_not_tracked_items_idxs]
        current_not_tracked_bboxes = tracked_bboxes[current_not_tracked_items_idxs]
        current_not_tracked_ids = tracked_ids[current_not_tracked_items_idxs]
        bboxes_data = [
            BboxData(image=frame, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            for (xmin, ymin, xmax, ymax) in current_not_tracked_bboxes
        ]
        n_bboxes_data_gen = BatchGeneratorBboxData([bboxes_data], batch_size=1)
        pred_bboxes_data = self.classification_inferencer.predict(n_bboxes_data_gen)[0]

        for i, bbox_data, tracked_id in zip(original_idxs, pred_bboxes_data, current_not_tracked_ids):
            if bbox_data.classification_score >= classification_score_threshold:
                ready_at_frame = (
                    frame_idx
                    + detection_delay_frames
                    + (classification_delay_frames * i)
                )

                frame_result = FrameResult(
                    bbox_data=bbox_data,
                    track_id=tracked_id,
                    ready_at_frame=ready_at_frame
                )
                self.current_ready_frames_queue.append(frame_result)

        return tracked_bboxes, tracked_ids

    def process_video(
        self,
        video_path: Union[str, Path],
        classification_delay: int,
        detection_delay: int,
        detection_score_threshold: float,
        classification_score_threshold: float,
        class_names_to_find: List[Literal[str, 'all classes']] = None
    ) -> tempfile.NamedTemporaryFile:
        result = []

        self.sort_tracker = Sort()
        with imageio.get_reader(video_path, ".mp4") as reader:
            fps = reader.get_meta_data()["fps"]

            for frame_idx, frame in enumerate(reader):
                frame = imutils.resize(frame, width=self.frame_height, height=self.frame_height)

                detection_delay_frames = detection_delay * fps / 1000
                if frame_idx % detection_delay_frames == 0:
                    tracked_bboxes, tracked_ids = self.run_pipeline_on_frame(
                        frame_pixels=frame,
                        frame_idx=frame_idx,
                        fps=fps,
                        detection_delay=detection_delay,
                        classification_delay=classification_delay,
                        detection_score_threshold=detection_score_threshold,
                        classification_score_threshold=classification_score_threshold,
                    )
                else:
                    tracked_bboxes, tracked_ids = self.run_tracking_on_frame(frame)

                ready_frames_at_the_moment = [
                    ready_frame
                    for ready_frame in self.current_ready_frames_queue
                    if ready_frame.ready_at_frame <= frame_idx
                ]

                result_frame = draw_overlay(
                    frame=frame,
                    tracked_bboxes=tracked_bboxes,
                    tracked_ids=tracked_ids,
                    ready_frames_at_the_moment=ready_frames_at_the_moment,
                    label_images=self.label_images,
                    class_names_to_find=class_names_to_find
                )
                result.append(result_frame)

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4').name
        imageio.mimwrite(temp_file, result, codec="h264", fps=fps)
        return temp_file
