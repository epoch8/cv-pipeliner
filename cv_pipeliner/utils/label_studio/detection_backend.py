from pathlib import Path
from label_studio.ml import LabelStudioMLBase

from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.utils.label_studio.detection_project import MAIN_PROJECT_FILENAME, BACKEND_PROJECT_FILENAME


DIRECTORY = Path(__file__).absolute().parent.parent  # this script __file__ will be in backend folder
MAIN_PROJECT_DIRECTORY = DIRECTORY / MAIN_PROJECT_FILENAME
BACKEND_PROJECT_DIRECTORY = DIRECTORY / BACKEND_PROJECT_FILENAME
IMAGE_PATHS = (MAIN_PROJECT_DIRECTORY / 'upload').glob('*.*')
IMAGES_DATA = BrickitDataConverter().get_images_data_from_annots(
    image_paths=IMAGE_PATHS,
    annots=BACKEND_PROJECT_DIRECTORY / 'predictions.json'
) if (BACKEND_PROJECT_DIRECTORY / 'predictions.json').exists() else []
FILENAME_TO_PRED_IMAGE_DATA = {
    image_data.image_path.name: image_data
    for image_data in IMAGES_DATA
}


class DetectionBackend(LabelStudioMLBase):
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
        images = [task['data']['image'] for task in tasks]
        predictions = []
        for image_filepath in images:
            filename = Path(image_filepath).name
            pred_image_data = FILENAME_TO_PRED_IMAGE_DATA[filename]
            image = pred_image_data.open_image()
            original_width, original_height = image.shape[1], image.shape[0]
            result = []
            for pred_bbox_data in pred_image_data.bboxes_data:
                ymin, xmin, ymax, xmax = (
                    pred_bbox_data.ymin, pred_bbox_data.xmin, pred_bbox_data.ymax, pred_bbox_data.xmax
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
                            pred_bbox_data.label
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
