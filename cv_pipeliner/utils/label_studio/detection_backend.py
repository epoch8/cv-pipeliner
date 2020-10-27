import json

from pathlib import Path
from label_studio.ml import LabelStudioMLBase

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.utils.label_studio.detection_project import (
    MAIN_PROJECT_FILENAME, BACKEND_PROJECT_FILENAME, load_tasks
)


DIRECTORY = Path(__file__).absolute().parent.parent  # this script __file__ will be in backend folder
MAIN_PROJECT_DIRECTORY = DIRECTORY / MAIN_PROJECT_FILENAME
BACKEND_PROJECT_DIRECTORY = DIRECTORY / BACKEND_PROJECT_FILENAME


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
        predictions = []
        tasks_data = load_tasks(MAIN_PROJECT_DIRECTORY)
        id_to_task_data = {
            task_data.id: task_data for task_data in tasks_data
        }
        for task in tasks:
            task_data = id_to_task_data[int(task['id'])]
            predictions.append(
                {
                    'result': task_data.convert_to_rectangle_labels(),
                    'score': 1.0
                }
            )
        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        return {}
