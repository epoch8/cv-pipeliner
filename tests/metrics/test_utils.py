import copy

import cv2

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_image_data


def visualize_images_data_with_overlay(image_data1: ImageData, image_data2: ImageData):
    image_data1 = copy.deepcopy(image_data1)
    image_data2 = copy.deepcopy(image_data2)
    for image_data, tag in [(image_data1, "left"), (image_data2, "right")]:
        for bbox_data in image_data.bboxes_data:
            bbox_data.label = f"{bbox_data.label} [{tag}]"
    image1 = visualize_image_data(image_data=image_data1, include_labels=True)
    image2 = visualize_image_data(image_data=image_data2, include_labels=True)
    image = cv2.addWeighted(src1=image1, alpha=1.0, src2=image2, beta=1.0, gamma=0.0)
    return image
