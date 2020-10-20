from pathlib import Path
from label_studio.ml import LabelStudioMLBase

from cv_pipeliner.core.data import BboxData
from cv_pipeliner.utils.label_studio.detection_project import MAIN_PROJECT_FILENAME, BACKEND_PROJECT_FILENAME

DIRECTORY = Path(__file__).absolute().parent.parent  # this script __file__ will be in backend folder
MAIN_PROJECT_DIRECTORY = DIRECTORY / MAIN_PROJECT_FILENAME
BACKEND_PROJECT_DIRECTORY = DIRECTORY / BACKEND_PROJECT_FILENAME


class ClassificationBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super().__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        self.from_name = 'bbox'
        self.info = self.parsed_label_config['bbox']
        self.to_name = 'image'
        if not self.train_output:
            # If there is no trainings, define cold-started model
            pass
        else:
            # otherwise load the model from the latest training results
            pass

    def predict(self, tasks, **kwargs):
        # collect input images
        predictions = []
        for task in tasks:
            bbox_data_as_cropped_image = BboxData(
                image_path=task['data']['cropped_image_path'],
                xmin=task['data']['bbox_data_as_cropped_image']['xmin'],
                ymin=task['data']['bbox_data_as_cropped_image']['ymin'],
                xmax=task['data']['bbox_data_as_cropped_image']['xmax'],
                ymax=task['data']['bbox_data_as_cropped_image']['ymax'],
                label=task['data']['bbox_data_as_cropped_image']['label'],
                top_n=task['data']['bbox_data_as_cropped_image']['top_n'],
                labels_top_n=task['data']['bbox_data_as_cropped_image']['labels_top_n'],
                additional_info=task['data']['bbox_data_as_cropped_image']['additional_info']
            )
            image = bbox_data_as_cropped_image.open_image()
            original_width, original_height = image.shape[1], image.shape[0]
            result = []
            ymin, xmin, ymax, xmax = (
                bbox_data_as_cropped_image.ymin, bbox_data_as_cropped_image.xmin,
                bbox_data_as_cropped_image.ymax, bbox_data_as_cropped_image.xmax
            )
            height = ymax - ymin
            width = xmax - xmin
            x = xmin / original_width * 100
            y = ymin / original_height * 100
            height = height / original_height * 100
            width = width / original_width * 100
            result.append({
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "original_width": original_width,
                    "original_height": original_height,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [
                        bbox_data_as_cropped_image.label
                    ],
                    "rotation": 0,
                }
            })
            predictions.append(
                {'result': result, 'score': 1.0}
            )
        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        return {}
