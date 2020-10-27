
import os
import time
import stat
import json
import subprocess
import shutil
import pickle

from pathlib import Path
from typing import List, Union, Dict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.logging import logger
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.inferencers.pipeline import PipelineInferencer

from cv_pipeliner.data_converters.brickit import BrickitDataConverter

from cv_pipeliner.utils.images import rotate_point


MAIN_PROJECT_FILENAME = 'main_project'
BACKEND_PROJECT_FILENAME = 'backend'
DETECTION_BACKEND_SCRIPT = Path(__file__).absolute().parent / 'detection_backend.py'
SPECIAL_CHARACTER = '\u2800'  # Label studio can't accept class_names when it's digit, so we add this invisible char.


@dataclass
class TaskData:
    task_json: Dict
    id: int
    image_data: ImageData
    is_done: bool = False
    is_skipped: bool = False
    is_trash: bool = False

    def convert_to_rectangle_labels(self) -> Dict:
        image = self.image_data.open_image()
        original_width, original_height = image.shape[1], image.shape[0]
        rectangle_labels = []
        for bbox_data in self.image_data.bboxes_data:
            xmin, ymin, xmax, ymax = (
                bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax
            )
            height = ymax - ymin
            width = xmax - xmin
            x = xmin / original_width * 100
            y = ymin / original_height * 100
            height = height / original_height * 100
            width = width / original_width * 100
            rectangle_labels.append({
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": original_width,
                "original_height": original_height,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [
                        bbox_data.label, json.dumps(bbox_data.additional_info)
                    ],
                    "rotation": 0,
                }
            })
        return rectangle_labels

    def parse_rectangle_labels(
        self,
        result: Dict,
        image_path: Union[str, Path]
    ) -> BboxData:
        original_height = result['original_height']
        original_width = result['original_width']
        height = result['value']['height']
        width = result['value']['width']
        xmin = result['value']['x']
        ymin = result['value']['y']
        angle = result['value']['rotation']
        label = result['value']['rectanglelabels'][0]
        if len(result['value']['rectanglelabels']) == 2:
            additional_info = json.loads(result['value']['rectanglelabels'][1])
        else:
            additional_info = {}
        xmax = xmin + width
        ymax = ymin + height
        xmin = xmin / 100 * original_width
        ymin = ymin / 100 * original_height
        xmax = xmax / 100 * original_width
        ymax = ymax / 100 * original_height
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        new_points = [rotate_point(x=x, y=y, cx=xmin, cy=ymin, angle=angle) for (x, y) in points]
        xmin = max(0, min([x for (x, y) in new_points]))
        ymin = max(0, min([y for (x, y) in new_points]))
        xmax = max([x for (x, y) in new_points])
        ymax = max([y for (x, y) in new_points])
        bbox = np.array([xmin, ymin, xmax, ymax])
        bbox = bbox.round().astype(int)
        xmin, ymin, xmax, ymax = bbox
        bbox_data = BboxData(
            image_path=image_path,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            label=label,
            additional_info=additional_info
        )
        return bbox_data

    def _parse_completion(
        self,
        completions_json: Dict
    ):
        self.is_done = True
        if len(completions_json['completions']) > 1:
            raise ValueError(
                f'Find a task with two or more completions. Task_id: {self.id}'
            )
        completion = completions_json['completions'][0]
        image_path = Path(completions_json['data']['src_image_path'])
        additional_info = completions_json['data']['src_image_data']['additional_info']
        if 'skipped' in completion and completion['skipped']:
            self.is_skipped = True
            image_data = ImageData()
            image_data.from_dict(completions_json['data']['src_image_data'])
        else:
            bboxes_data = []
            for result in completion['result']:
                from_name = result['from_name']
                if from_name == 'bbox':
                    bboxes_data.append(
                        self.parse_rectangle_labels(
                            result=result,
                            image_path=image_path
                        )
                    )
                elif from_name == 'trash':
                    if 'Trash' in result['value']['choices']:
                        self.is_trash = True
                        additional_info['is_trash'] = True
            image_data = ImageData(
                image_path=Path(completions_json['data']['src_image_path']),
                bboxes_data=bboxes_data,
                additional_info=additional_info
            )
        return image_data

    def __init__(
        self,
        main_project_directory: Union[str, Path],
        task_json: Dict
    ):
        self.task_json = task_json
        self.id = task_json['id']
        completions_filepath = Path(main_project_directory) / 'completions' / f'{self.id}.json'
        if completions_filepath.exists():
            with open(completions_filepath, 'r') as src:
                completions_json = json.load(src)
            self.image_data = self._parse_completion(
                completions_json=completions_json
            )
        else:
            self.image_data = ImageData()
            self.image_data.from_dict(task_json['data']['src_image_data'])


def load_tasks(main_project_directory) -> List[TaskData]:
    with open(main_project_directory/'tasks.json', 'r') as src:
        tasks_json = json.load(src)
    tasks_json = [tasks_json[key] for key in tasks_json]
    tasks_data = [
        TaskData(
            main_project_directory=main_project_directory,
            task_json=task_json
        )
        for task_json in tasks_json
    ]
    return tasks_data


class LabelStudioProject_Detection:
    def __init__(
        self,
        directory: Union[str, Path]
    ):
        self.directory = Path(directory).absolute()
        self.main_project_directory = self.directory / MAIN_PROJECT_FILENAME
        self.backend_project_directory = self.directory / BACKEND_PROJECT_FILENAME

        self.running_project_process = None
        self.load()

    def _class_name_with_special_character(self, class_name: str) -> str:
        if SPECIAL_CHARACTER in class_name:
            return class_name
        else:
            return f"{class_name}{SPECIAL_CHARACTER}" if class_name.isdigit() else class_name

    def _class_name_without_special_character(self, class_name: str) -> str:
        return class_name.replace(SPECIAL_CHARACTER, '')

    def generate_config_xml(
        self,
        class_names: List[str]
    ):
        labels = '\n'.join(
            [f'<Label value="{class_name}"/>' for class_name in class_names]
        )
        config_xml = f'''
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image" canRotate="false">
    {labels}
  </RectangleLabels>
  <Choices name="trash" choice="multiple" toName="image" showInLine="true">
    <Choice value="Trash"/>
    <Choice value="---"/>
  </Choices>
</View>
'''.strip('\n')

        with open(self.main_project_directory/'config.xml', 'w') as out:
            out.write(config_xml)

    def set_class_names(
        self,
        class_names: Union[str, Path, List[str]]
    ):
        if isinstance(class_names, str) or isinstance(class_names, Path):
            with open(class_names, 'r') as src:
                class_names = json.load(src)
        self.class_names = [
            self._class_name_with_special_character(class_name)
            for class_name in class_names
        ]
        with open(self.main_project_directory/'class_names.json', 'w') as out:
            json.dump(self.class_names, out, indent=4)
        self.generate_config_xml(self.class_names)

    def inference_and_make_predictions_for_backend(
        self,
        image_paths: List[Union[str, Path]],
        detection_score_threshold: float,
        default_class_name: str = None,
        batch_size: int = 16
    ) -> List[ImageData]:
        logger.info('Making inference for tasks...')
        images_data = [ImageData(image_path=image_path) for image_path in image_paths]
        images_data_gen = BatchGeneratorImageData(
            data=images_data,
            batch_size=16,
            use_not_caught_elements_as_last_batch=True
        )

        if default_class_name is None:
            default_class_name = self.class_names[0]
        else:
            default_class_name = self._class_name_with_special_character(default_class_name)
            assert default_class_name in self.class_names

        if self.classification_model_spec is None:
            detection_model = self.detection_model_spec.load()
            detection_inferencer = DetectionInferencer(detection_model)
            pred_images_data = detection_inferencer.predict(
                images_data_gen=images_data_gen,
                score_threshold=detection_score_threshold
            )
            pred_images_data = [
                ImageData(
                    image_path=image_data.image_path,
                    bboxes_data=[
                        BboxData(
                            image_path=bbox_data.image_path,
                            xmin=bbox_data.xmin,
                            ymin=bbox_data.ymin,
                            xmax=bbox_data.xmax,
                            ymax=bbox_data.ymax,
                            label=default_class_name
                        )
                        for bbox_data in image_data.bboxes_data
                    ]
                )
                for image_data in pred_images_data
            ]
        else:
            pipeline_model_spec = PipelineModelSpec(
                detection_model_spec=self.detection_model_spec,
                classification_model_spec=self.classification_model_spec
            )
            pipeline_model = pipeline_model_spec.load()
            pipeline_inferencer = PipelineInferencer(pipeline_model)
            pred_images_data = pipeline_inferencer.predict(
                images_data_gen=images_data_gen,
                detection_score_threshold=detection_score_threshold
            )

        return pred_images_data

    def initialize_backend(
        self,
        backend_port: int,
        detection_model_spec: DetectionModelSpec,
        classification_model_spec: ClassificationModelSpec = None
    ):
        self.detection_model_spec = detection_model_spec
        self.classification_model_spec = classification_model_spec
        logger.info('Initializing LabelStudioProject backend...')
        backend_project_process = subprocess.Popen(
            ["label-studio-ml", "init", str(self.backend_project_directory), '--script',
                str(DETECTION_BACKEND_SCRIPT)],
            stdout=subprocess.PIPE
        )
        backend_project_process.wait()
        if self.detection_model_spec is not None:
            with open(self.backend_project_directory / 'detection_model_spec.pkl', 'wb') as out:
                pickle.dump(detection_model_spec, out)
            if classification_model_spec is not None:
                with open(self.backend_project_directory / 'classification_model_spec.pkl', 'wb') as out:
                    pickle.dump(classification_model_spec, out)
        with open(self.main_project_directory/'config.json', 'r') as src:
            config_json = json.load(src)
        config_json['ml_backends'].append({
            'url': f'http://localhost:{backend_port}/',
            'name': 'main_model'
        })
        with open(self.main_project_directory/'config.json', 'w') as out:
            json.dump(config_json, out, indent=4)

    def initialize_project(
        self,
        class_names: Union[str, Path, List[str]],
        port: int = 8080,
        url: str = 'http://localhost:8080/',
        detection_model_spec: DetectionModelSpec = None,
        classification_model_spec: ClassificationModelSpec = None,
        backend_port: int = 9080,
    ):
        if self.directory.exists():
            raise ValueError(
                f'Directory {self.directory} is exists. Delete it before creating the project.'
            )
        label_studio_process = subprocess.Popen(
            ["label-studio", "init", str(self.main_project_directory)],
            stdout=subprocess.PIPE
        )
        output = '\n'.join([x.decode() for x in label_studio_process.communicate() if x])
        self.cv_pipeliner_settings = {
            'port': port,
            'backend_port': backend_port,
            'url': url,  # use hack for our cluster
            'additional_info': 'This project was created by cv_pipeliner.utils.label_studio.'
        }
        if 'Label Studio has been successfully initialized' in output:
            logger.info(f'Initializing LabelStudioProject "{self.directory.name}"...')
            self.set_class_names(class_names)
            (self.main_project_directory / 'upload').mkdir()
            with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'w') as out:
                json.dump(self.cv_pipeliner_settings, out, indent=4)
        else:
            raise ValueError(f'Label Studio has not been initialized. Output: {output}.')

        self.initialize_backend(
            backend_port=backend_port,
            detection_model_spec=detection_model_spec,
            classification_model_spec=classification_model_spec
        )
        run_command = (
            '#!/bin/sh\n'
            f'label-studio-ml start {self.backend_project_directory} -p {backend_port} & '
            f'label-studio start {self.main_project_directory} -p {port}'
        )
        with open(self.directory / 'run.sh', 'w') as out:
            out.write(run_command)
        st = os.stat(self.directory / 'run.sh')  # chmod +x
        os.chmod(self.directory / 'run.sh', st.st_mode | stat.S_IEXEC)

        self.running_project_process = None

    def run_project(self):
        if self.running_project_process is None:
            logger.info(
                f'Start project "{self.directory.name}" '
                f'(port {self.cv_pipeliner_settings["port"]} and '
                f'backend port port {self.cv_pipeliner_settings["backend_port"]})...'
            )
            self.running_project_process = subprocess.Popen(
                ["./run.sh"],
                stdout=subprocess.PIPE,
                cwd=str(self.directory)
            )
        else:
            raise ValueError('The project is already running.')

    def load(self):
        if (self.main_project_directory / 'cv_pipeliner_settings.json').exists():
            logger.info(f'Loading LabelStudioProject "{self.directory.name}"...')
            with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'r') as src:
                self.cv_pipeliner_settings = json.load(src)
            with open(self.main_project_directory/'class_names.json', 'r') as src:
                self.class_names = json.load(src)
            self.tasks_data = load_tasks(self.main_project_directory)

        if (self.backend_project_directory / 'detection_model_spec.pkl').exists():
            with open(self.backend_project_directory / 'detection_model_spec.pkl', 'rb') as src:
                self.detection_model_spec = pickle.load(src)
        else:
            self.detection_model_spec = None

        if (self.backend_project_directory / 'classification_model_spec.pkl').exists():
            with open(self.backend_project_directory / 'classification_model_spec.pkl', 'rb') as src:
                self.classification_model_spec = pickle.load(src)
        else:
            self.classification_model_spec = None

    def add_tasks(
        self,
        images: Union[List[Union[str, Path]], List[ImageData]],
        detection_score_threshold: float = None,
        default_class_name: str = None,
        batch_size: int = 16,
        reannotate_following_images_filenames: List[str] = None
    ):
        with open(self.main_project_directory/'tasks.json', 'r') as src:
            tasks_json = json.load(src)
        tasks_ids = [tasks_json[key]['id'] for key in tasks_json]
        start = max(tasks_ids) if len(tasks_ids) > 0 else 0
        logger.info('Adding tasks...')

        if all([isinstance(image, ImageData) for image in images]):
            images_data = images
            add_completions = True
        else:
            if not all([isinstance(image, str) or isinstance(image, Path) for image in images]):
                raise ValueError('Argument images can be only list of paths or list of ImageData.')
            else:
                image_paths = images
                if self.detection_model_spec is not None and detection_score_threshold is not None:
                    images_data = self.inference_and_make_predictions_for_backend(
                        image_paths=image_paths,
                        detection_score_threshold=detection_score_threshold,
                        default_class_name=default_class_name,
                        batch_size=batch_size
                    )
                else:
                    images_data = [ImageData(image_path=image_path, bboxes_data=[]) for image_path in image_paths]
            add_completions = False

        for image_data in images_data:
            image_data.apply_str_func_to_labels_inplace(self._class_name_with_special_character)

        for id, image_data in tqdm(list(enumerate(images_data, start=start))):
            image_path = image_data.image_path
            image = (
                f"{self.cv_pipeliner_settings['url']}/data/upload/{image_path.name}"  # noqa: E501
            )
            filepath = self.main_project_directory / 'upload' / f'{image_path.name}'
            if filepath.exists():
                raise ValueError(f'Error: task with image {image_path} is already exists.')
            shutil.copy(image_path, filepath)
            tasks_json[str(id)] = {
                'id': id,
                'data': {
                    'image': str(image),
                    'src_image_path': str(image_path),
                    'src_image_data': image_data.asdict()
                }
            }

        with open(self.main_project_directory/'tasks.json', 'w') as out:
            json.dump(tasks_json, out, indent=4)

        self.tasks_data = load_tasks(self.main_project_directory)
        if all([isinstance(image, ImageData) for image in images]):  # already annotated
            current_time = int(time.time())
            for task_data in self.tasks_data:
                filename = task_data.image_data.image_path.name
                if (
                    add_completions and reannotate_following_images_filenames is not None and
                    filename in reannotate_following_images_filenames
                ):
                    continue

                completions_json = tasks_json[str(task_data.id)].copy()
                completions_json['completions'] = [
                    {
                        'created_at': current_time,
                        'id': str(task_data.id),
                        'result': task_data.convert_to_rectangle_labels(),
                    }
                ]
                with open(self.main_project_directory / f"completions/{task_data.id}.json", 'w') as out:
                    json.dump(completions_json, out, indent=4)
            self.tasks_data = load_tasks(self.main_project_directory)

    def get_images_data(
        self,
        get_only_done: bool = True,
        get_only_not_skipped: bool = True
    ) -> List[ImageData]:
        images_data = [
            task_data.image_data
            for task_data in self.tasks_data
            if (
                (get_only_done and task_data.is_done) or (not get_only_done)
                and
                (get_only_not_skipped and not task_data.is_skipped) or (not get_only_not_skipped)
            )
        ]
        for image_data in images_data:
            image_data.apply_str_func_to_labels_inplace(self._class_name_without_special_character)

        return images_data
