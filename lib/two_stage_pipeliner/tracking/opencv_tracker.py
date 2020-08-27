import cv2

from app.utils.bbox_utils import coco_bboxes_to_voc, voc_bboxes_to_coco


class OpenCVTracker:
    def __init__(self, initial_bboxes, initial_frame):
        self.multiTracker = cv2.MultiTracker_create()
        formated_bboxes = voc_bboxes_to_coco(initial_bboxes.copy())
        for bbox in formated_bboxes:
            self.multiTracker.add(cv2.TrackerMedianFlow_create(), initial_frame, bbox)

    def update(self, frame_pixels):
        success, tracked_bboxes = self.multiTracker.update(frame_pixels)
        tracked_bboxes = coco_bboxes_to_voc(tracked_bboxes, frame_pixels.shape)
        return tracked_bboxes
