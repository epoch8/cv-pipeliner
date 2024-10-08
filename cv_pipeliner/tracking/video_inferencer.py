import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Tuple, Union

import imageio
import imutils
import numpy as np
from tqdm import tqdm

from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.tracking.opencv_tracker import OpenCVTracker
from cv_pipeliner.tracking.sort_tracker import Sort
from cv_pipeliner.visualizers.core.image_data import visualize_image_data


@dataclass
class FrameResult:
    label: str
    track_id: int
    ready_at_frame: int


class VideoInferencer:
    def __init__(
        self,
        pipeline_inferencer: PipelineInferencer,
        draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray],
        write_labels: bool = True,
        frame_width: int = 640,
        frame_height: int = 1152,
        batch_size: int = 16,
    ):
        self.detection_inferencer = DetectionInferencer(pipeline_inferencer.model.detection_model)
        self.classification_inferencer = ClassificationInferencer(pipeline_inferencer.model.classification_model)
        self.draw_base_labels_with_given_label_to_base_label_image = (
            draw_base_labels_with_given_label_to_base_label_image
        )
        self.write_labels = write_labels

        self.sort_tracker = None
        self.opencv_tracker = None

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.current_ready_frames_queue: List[FrameResult] = []

    def update_sort_tracker(
        self, bboxes: List[Tuple[int, int, int, int]], scores: List[float] = None
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        if scores is None:
            # SORT requires detection scores, use 1 while tracking
            scores = np.ones(bboxes.shape[0])
        tracked_bboxes_sort = self.sort_tracker.update(np.column_stack([bboxes, scores]))
        tracked_bboxes_sort = tracked_bboxes_sort.round().astype(int)
        tracked_bboxes = tracked_bboxes_sort[:, :-1]
        tracked_bboxes = np.clip(tracked_bboxes, 0, tracked_bboxes.max(initial=0))
        tracked_ids = tracked_bboxes_sort[:, -1]
        return tracked_bboxes, tracked_ids

    def run_tracking_on_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        tracked_bboxes_optical = self.opencv_tracker.update(frame)
        tracked_bboxes, tracked_ids = self.update_sort_tracker(bboxes=tracked_bboxes_optical)
        return tracked_bboxes, tracked_ids

    def run_pipeline_on_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        detection_delay: int,
        classification_delay: int,
        detection_score_threshold: float,
        batch_size: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        frame = frame.copy()
        image_data = ImageData(image=frame)
        image_data_gen = BatchGeneratorImageData(
            [image_data], batch_size=batch_size, use_not_caught_elements_as_last_batch=True
        )

        pred_image_data = self.detection_inferencer.predict(
            images_data_gen=image_data_gen, score_threshold=detection_score_threshold
        )[0]
        bboxes = np.array(
            [
                (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)
                for bbox_data in pred_image_data.bboxes_data
            ]
        )
        detection_scores = np.array([bbox_data.detection_score for bbox_data in pred_image_data.bboxes_data])

        self.opencv_tracker = OpenCVTracker(bboxes, frame)
        tracked_bboxes, tracked_ids = self.update_sort_tracker(bboxes=bboxes, scores=detection_scores)

        # detection_delay_frames = int(round(detection_delay * fps / 1000))
        # classification_delay_frames = int(round(classification_delay * fps / 1000))

        current_tracks_ids = [frame_result.track_id for frame_result in self.current_ready_frames_queue]
        current_not_tracked_items_idxs = [
            idx for idx, tracked_id in enumerate(tracked_ids) if tracked_id not in current_tracks_ids
        ]
        if current_not_tracked_items_idxs:
            current_not_tracked_bboxes = tracked_bboxes[current_not_tracked_items_idxs]
            current_not_tracked_ids = tracked_ids[current_not_tracked_items_idxs]
            bboxes_data = [
                BboxData(image=frame, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                for (xmin, ymin, xmax, ymax) in current_not_tracked_bboxes
            ]
            bboxes_data_gen = BatchGeneratorBboxData(
                [bboxes_data], batch_size=batch_size, use_not_caught_elements_as_last_batch=True
            )
            pred_bboxes_data = self.classification_inferencer.predict(bboxes_data_gen)[0]

            for bbox_data, tracked_id in zip(pred_bboxes_data, current_not_tracked_ids):
                ready_at_frame = frame_idx
                frame_result = FrameResult(label=bbox_data.label, track_id=tracked_id, ready_at_frame=ready_at_frame)
                self.current_ready_frames_queue.append(frame_result)

        return tracked_bboxes, tracked_ids

    def draw_overlay(
        self,
        frame: np.ndarray,
        tracked_bboxes: List[Tuple[int, int, int, int]],
        tracked_ids: List[int],
        ready_frames_at_the_moment: List["FrameResult"],
        filter_by_labels: List[str] = None,
    ) -> np.ndarray:
        image = frame.copy()
        tracked_bboxes = tracked_bboxes.astype(int)
        ready_tracks_ids_at_the_moment = [ready_frame.track_id for ready_frame in ready_frames_at_the_moment]

        current_bboxes_data = []
        for bbox, track_id in zip(tracked_bboxes, tracked_ids):
            if track_id not in ready_tracks_ids_at_the_moment:
                continue

            xmin, ymin, xmax, ymax = bbox
            ready_frame = ready_frames_at_the_moment[ready_tracks_ids_at_the_moment.index(track_id)]
            label = ready_frame.label
            current_bbox_data = BboxData(image=image, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label)
            current_bboxes_data.append(current_bbox_data)

        image_data = ImageData(image=frame, bboxes_data=current_bboxes_data)

        image = visualize_image_data(
            image_data=image_data,
            include_labels=self.write_labels,
            filter_by_labels=filter_by_labels,
            draw_base_labels_with_given_label_to_base_label_image=(
                self.draw_base_labels_with_given_label_to_base_label_image
            ),
            known_labels=self.classification_inferencer.class_names,
        )

        return image

    def process_video(
        self,
        video_file: Union[str, Path, BytesIO],
        classification_delay: int,
        detection_delay: int,
        detection_score_threshold: float,
        filter_by_labels: List[str],
        disable_tqdm: bool = False,
        batch_size: int = 16,
    ) -> tempfile.NamedTemporaryFile:
        result = []

        self.sort_tracker = Sort()
        with imageio.get_reader(video_file, ".mp4") as reader:
            fps = reader.get_meta_data()["fps"]

            for frame_idx, frame in tqdm(list(enumerate(reader)), disable=disable_tqdm):
                frame = imutils.resize(frame, width=self.frame_height, height=self.frame_height)

                detection_delay_frames = detection_delay * fps / 1000
                if frame_idx % detection_delay_frames == 0:
                    tracked_bboxes, tracked_ids = self.run_pipeline_on_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        fps=fps,
                        detection_delay=detection_delay,
                        classification_delay=classification_delay,
                        detection_score_threshold=detection_score_threshold,
                        batch_size=batch_size,
                    )
                else:
                    tracked_bboxes, tracked_ids = self.run_tracking_on_frame(frame)

                ready_frames_at_the_moment = [
                    ready_frame
                    for ready_frame in self.current_ready_frames_queue
                    if ready_frame.ready_at_frame <= frame_idx
                ]

                result_frame = self.draw_overlay(
                    frame=frame,
                    tracked_bboxes=tracked_bboxes,
                    tracked_ids=tracked_ids,
                    ready_frames_at_the_moment=ready_frames_at_the_moment,
                    filter_by_labels=filter_by_labels,
                )
                result.append(result_frame)

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4").name
        imageio.mimwrite(temp_file, result, codec="h264", fps=fps)
        return temp_file
