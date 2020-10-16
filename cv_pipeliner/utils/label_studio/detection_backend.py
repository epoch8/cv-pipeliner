from pathlib import Path
import json
import imageio
from label_studio.ml import LabelStudioMLBase

from cv_pipeliner.data_converters.brickit import BrickitDataConverter

DIRECTORY = '/mnt/c/Users/bobokvsky/YandexDisk-bobok100500@yandex.ru/Job/temp/label_studio_detection/'
DIRECTORY = Path(DIRECTORY)  # cv.pipeliner --> DIRECTORY_TO_BE_CHANGED
IMAGE_PATHS = (DIRECTORY / 'main_project' / 'upload').glob('*.*')
IMAGES_DATA = BrickitDataConverter().get_images_data_from_annots(
    image_paths=IMAGE_PATHS,
    annots=DIRECTORY / 'backend' / 'annotations.json'
)
FILENAME_TO_IMAGE_DATA = {
    image_path
}

class PipelineBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(PipelineBackend, self).__init__(**kwargs)

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
            image = imageio.imread(image_filepath, pilmode='RGB')
            original_width, original_height = image.shape[1], image.shape[0]

            with open(BACKEND_PREIDCTIONS_DIR/f"{filename}.json", 'r') as src:
                bbox_in_total_image_data = json.load(src)
            xmin, ymin, xmax, ymax = bbox_in_total_image_data['bbox_in_total_image']
            height = ymax-ymin
            width = xmax-xmin
            x = xmin / original_width * 100
            y = ymin / original_height * 100
            height = height / original_height * 100
            width = width / original_width * 100
            result = [{
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
                        "brick"
                    ],
                    "rotation": 0,
                }
            }]
            predictions.append(
                {'result': result, 'score': 1.0}
            )
        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        return {}
