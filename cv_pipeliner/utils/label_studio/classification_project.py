import os
import stat
import json
import subprocess
import pickle
from urllib.parse import urljoin
from multiprocessing import Process

import copy
from PIL import Image
from pathlib import Path
from typing import List, Union, Dict, Callable
from dataclasses import dataclass

import numpy as np
import imageio
from tqdm import tqdm

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.logging import logger
from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.utils.images import (
    get_label_to_base_label_image, put_text_on_image, concat_images,
    draw_n_base_labels_images
)
from cv_pipeliner.utils.data import get_label_to_description


MAIN_PROJECT_FILENAME = 'main_project'
BACKEND_PROJECT_FILENAME = 'backend'
CLASSIFICATION_BACKEND_SCRIPT = Path(__file__).absolute().parent / 'classification_backend.py'
SPECIAL_CHARACTER = '\u2800'  # Label studio can't accept class_names when it's digit, so we add this invisible char.


def class_name_with_special_character(
    class_name: str,
    label_to_description: Dict[str, str] = None
) -> str:
    if SPECIAL_CHARACTER in class_name:
        return class_name
    else:
        if label_to_description is not None:
            if class_name in label_to_description:
                description = label_to_description[class_name]
                return f"{class_name}{SPECIAL_CHARACTER}[{description}]"
        return f"{class_name}{SPECIAL_CHARACTER}" if class_name.isdigit() else class_name


@dataclass
class TaskData:
    task_json: Dict
    id: int
    bbox_data: BboxData
    is_done: bool = False
    is_skipped: bool = False
    is_trash: bool = False
    label_to_description: Dict[str, str] = None

    def convert_to_rectangle_label(
        self
    ) -> Dict:
        bbox_data_as_cropped_image = BboxData()
        bbox_data_as_cropped_image.from_dict(self.task_json['data']['bbox_data_as_cropped_image'])
        image = bbox_data_as_cropped_image.open_image()
        original_width, original_height = image.shape[1], image.shape[0]
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
        rectangle_label = {
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
                    "bbox"
                ],
                "rotation": bbox_data_as_cropped_image.angle,
            }
        }
        return rectangle_label

    def convert_to_choice(
        self
    ) -> Dict:
        bbox_data_as_cropped_image = BboxData()
        bbox_data_as_cropped_image.from_dict(self.task_json['data']['bbox_data_as_cropped_image'])
        choice = {
            "value": {
                "choices": [
                    class_name_with_special_character(
                        bbox_data_as_cropped_image.label, self.label_to_description
                    )
                ]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "choices"
        }
        return choice

    def parse_rectangle_labels(
        self,
        result: Dict,
        src_image_path: Union[str, Path],
        src_bbox_data: BboxData,
        bbox_data_as_cropped_image: BboxData
    ) -> BboxData:
        original_height = result['original_height']
        original_width = result['original_width']
        height = result['value']['height']
        width = result['value']['width']
        xmin = result['value']['x']
        ymin = result['value']['y']
        angle = int(result['value']['rotation'])
        xmax = xmin + width
        ymax = ymin + height
        xmin = xmin / 100 * original_width
        ymin = ymin / 100 * original_height
        xmax = xmax / 100 * original_width
        ymax = ymax / 100 * original_height
        src_xmin = int(bbox_data_as_cropped_image.additional_info['src_xmin'])
        src_ymin = int(bbox_data_as_cropped_image.additional_info['src_ymin'])
        bbox = np.array([xmin, ymin, xmax, ymax])
        bbox = bbox.round().astype(int)
        xmin, ymin, xmax, ymax = bbox
        bbox_data = BboxData(
            image_path=src_image_path,
            xmin=src_xmin + xmin,
            ymin=src_ymin + ymin,
            xmax=src_xmin + xmax,
            ymax=src_ymin + ymax,
            angle=angle,
            top_n=src_bbox_data.top_n,
            labels_top_n=src_bbox_data.labels_top_n,
            additional_info=src_bbox_data.additional_info
        )
        return bbox_data

    def _parse_completion(
        self,
        completions_json: Dict
    ):
        self.is_done = True
        src_labels_from_top_n = completions_json['data']['src_bbox_data']['labels_top_n']
        src_image_path = completions_json['data']['src_image_path']
        bbox_data = BboxData()
        bbox_data.from_dict(completions_json['data']['src_bbox_data'])
        bbox_data_as_cropped_image = BboxData()
        bbox_data_as_cropped_image.from_dict(completions_json['data']['bbox_data_as_cropped_image'])
        additional_info = bbox_data.additional_info

        if len(completions_json['completions']) > 1:
            logger.warning(
                f'Find a task with two or more completions, fix it. Task_id: {self.id}'
            )
        completion = completions_json['completions'][-1]
        if 'skipped' in completion and completion['skipped']:
            self.is_skipped = True
        else:
            from_names = [result['from_name'] for result in completion['result']]
            label = None
            if from_names.count('bbox') == 0:
                logger.warning(f"Task {self.id} doesn't have bbox. Fix it. It will be taken from default bbox...")
            if from_names.count('bbox') >= 2:
                logger.warning(f"Task {self.id} have more than 1 bboxes. Fix it. It will be taken from default bbox...")
            for result in completion['result']:
                from_name = result['from_name']
                if from_name == 'bbox':
                    bbox_data = self.parse_rectangle_labels(
                        result=result,
                        src_image_path=src_image_path,
                        src_bbox_data=bbox_data,
                        bbox_data_as_cropped_image=bbox_data_as_cropped_image
                    )
                    additional_info = bbox_data.additional_info
                elif from_name == 'label':
                    value = result['value']['choices'][0]
                    if value[0] == '[' and value[-1] == ']' and value[1:-1].isdigit():
                        top_n = int(value[1:-1])
                        label = src_labels_from_top_n[top_n-1]
                    else:
                        label = value.split(SPECIAL_CHARACTER)[0]
                elif from_name == 'trash':
                    if 'Trash' in result['value']['choices']:
                        self.is_trash = True
                        additional_info['is_trash'] = True
                elif from_name == 'additional_label':
                    additional_info['additional_label'] = result['value']['text'][0]
            if label is None:
                logger.warning(f"Task {self.id} have no annotation for label. Fix it...")
            bbox_data.label = label
        return bbox_data

    def __init__(
        self,
        main_project_directory: Union[str, Path],
        task_json: Dict,
        label_to_description: Dict[str, str] = None
    ):
        self.task_json = task_json
        self.id = task_json['id']
        self.label_to_description = label_to_description
        completions_filepath = Path(main_project_directory) / 'completions' / f'{self.id}.json'
        if completions_filepath.exists():
            with open(completions_filepath, 'r') as src:
                completions_json = json.load(src)
            self.bbox_data = self._parse_completion(
                completions_json=completions_json
            )
        else:
            self.bbox_data = BboxData()
            self.bbox_data.from_dict(task_json['data']['src_bbox_data'])


def load_tasks(main_project_directory) -> List[TaskData]:
    with open(main_project_directory / 'cv_pipeliner_settings.json', 'r') as src:
        cv_pipeliner_settings = json.load(src)
    label_to_description = get_label_to_description(
        label_to_description_dict=cv_pipeliner_settings['label_to_description']
    ) if cv_pipeliner_settings['label_to_description'] is not None else None
    with open(main_project_directory / 'tasks.json', 'r') as src:
        tasks_json = json.load(src)
    tasks_json = [tasks_json[key] for key in tasks_json]
    tasks_data = [
        TaskData(
            main_project_directory=main_project_directory,
            task_json=task_json,
            label_to_description=label_to_description
        )
        for task_json in tasks_json
    ]
    return tasks_data


class LabelStudioProject_Classification:
    def __init__(
        self,
        directory: Union[str, Path]
    ):
        self.directory = Path(directory).absolute()
        self.main_project_directory = self.directory / MAIN_PROJECT_FILENAME
        self.backend_project_directory = self.directory / BACKEND_PROJECT_FILENAME

        self.class_names = None
        self.cv_pipeliner_settings = None
        self.classification_model_spec = None
        self.label_to_description = None
        self.load()

    def _class_name_with_special_character(self, class_name: str) -> str:
        return class_name_with_special_character(class_name, self.label_to_description)

    def _class_name_without_special_character(self, class_name: str) -> str:
        return class_name.split(SPECIAL_CHARACTER)[0]

    def _save_current_settings(self):
        with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'w') as out:
            json.dump(self.cv_pipeliner_settings, out, indent=4)

    def _generate_base_labels_images(
        self,
        class_names: List[str]
    ) -> str:
        label_to_base_label_image = get_label_to_base_label_image(
            base_labels_images=self.cv_pipeliner_settings['base_labels_images'],
            label_to_description=self.cv_pipeliner_settings['label_to_description'],
            add_label_to_image=True,
            make_labels_for_these_class_names_too=list(map(self._class_name_without_special_character, class_names))
        ) if self.cv_pipeliner_settings['base_labels_images'] is not None else None
        if label_to_base_label_image is None:
            return ''
        logger.info(
            'Base labels images detected. Appending changes to config...'
        )
        (self.main_project_directory / 'upload' / 'base_label_images').mkdir(exist_ok=True)
        labels_views = []
        for class_name in tqdm(class_names):
            label = self._class_name_without_special_character(class_name)
            image_path = self.main_project_directory / 'upload' / 'base_label_images' / f'{label}.png'
            Image.fromarray(label_to_base_label_image(label)).save(image_path)
            base_label_image_url = (
                urljoin(self.cv_pipeliner_settings['url'], f"data/upload/base_label_images/{label}.png")
            )
            labels_views.append(f'''
  <View visibleWhen="choice-selected"
        whenTagName="label" whenChoiceValue="{self._class_name_with_special_character(class_name)}">
    <Image name="image_{label}" value="{base_label_image_url}"></Image>
  </View>
'''.strip('\n'))

        base_labels_config_xml = '\n'.join(labels_views)
        return base_labels_config_xml

    def _generate_config_xml(
        self,
        class_names: List[str],
        can_rotate: bool
    ):
        labels = '\n'.join(
            [f'    <Label value="{class_name}"/>' for class_name in class_names]
        )
        can_rotate = 'true' if can_rotate else 'false'

        base_labels_config_xml = self._generate_base_labels_images(
            class_names=class_names
        )
        base_labels_config_xml = base_labels_config_xml if self.cv_pipeliner_settings['show_base_labels_images'] else ''
        labels = [
            f'    <Choice value="[{i+1}]"/>' for i in range(self.cv_pipeliner_settings['top_n'])
            if self.cv_pipeliner_settings['show_top_n_labels_images']
        ] + [
            f'    <Choice value="{self._class_name_with_special_character(class_name)}"/>'
            for class_name in class_names
        ]
        labels = '\n'.join(labels)

        config_xml = f'''
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image" canRotate="{can_rotate}">
    <Label value="bbox" hotkey="f"/>
  </RectangleLabels>
  <Filter name="filter_base_labels" toName="label"
    hotkey="shift+f" minlength="0"
    placeholder="(HINT) Type object name here..." />
  <Choices name="label" toName="image" showInLine="true">
{labels}
  </Choices>
{base_labels_config_xml}
  <Choices name="trash" choice="multiple" toName="image" showInLine="true">
    <Choice value="Trash"/>
    <Choice value="---"/>
  </Choices>
  <TextArea name="additional_label"
            toName="image"
            showSubmitButton="true"
            maxSubmissions="1"
            editable="true"
            placeholder="Additional label if needed"/>
</View>
'''.strip('\n')

        with open(self.main_project_directory / 'config.xml', 'w') as out:
            out.write(config_xml)

    def set_class_names(
        self,
        class_names: Union[str, Path, List[str]],
        can_rotate: bool = False,
    ):
        if isinstance(class_names, str) or isinstance(class_names, Path):
            with open(class_names, 'r') as src:
                class_names = json.load(src)
        assert len(set(class_names)) == len(class_names)
        self.class_names = [
            self._class_name_with_special_character(class_name)
            for class_name in class_names
        ]
        with open(self.main_project_directory / 'class_names.json', 'w') as out:
            json.dump(self.class_names, out, indent=4)
        self._generate_config_xml(
            class_names=self.class_names, can_rotate=can_rotate
        )

    def inference_and_make_predictions_for_backend(
        self,
        image_paths: List[Union[str, Path]],
        n_bboxes_data: List[List[BboxData]],
        default_class_name: str = None,
        batch_size: int = 16,
    ):
        logger.info('Making inference for tasks...')
        n_bboxes_data_gen = BatchGeneratorBboxData(
            data=n_bboxes_data,
            batch_size=16,
            use_not_caught_elements_as_last_batch=True
        )

        if default_class_name is None:
            default_class_name = self.class_names[0]
        else:
            default_class_name = self._class_name_with_special_character(default_class_name)
            assert default_class_name in self.class_names

        classification_model = self.classification_model_spec.load()
        classification_inferencer = ClassificationInferencer(classification_model)
        pred_n_bboxes_data = classification_inferencer.predict(
            n_bboxes_data_gen=n_bboxes_data_gen,
            top_n=self.cv_pipeliner_settings['top_n']
        )

        def func_label(label: str) -> str:
            label_with_special_character = self._class_name_with_special_character(label)
            if label_with_special_character in self.class_names:
                return label_with_special_character
            else:
                return default_class_name

        for pred_bboxes_data in pred_n_bboxes_data:
            for pred_bbox_data in pred_bboxes_data:
                pred_bbox_data.label = func_label(pred_bbox_data.label)
        return pred_n_bboxes_data

    def initialize_backend(
        self,
        backend_port: int,
        classification_model_spec: ClassificationModelSpec = None
    ):
        self.classification_model_spec = classification_model_spec
        logger.info('Initializing LabelStudioProject backend...')
        backend_project_process = subprocess.Popen(
            ["label-studio-ml", "init", str(self.backend_project_directory), '--script',
                str(CLASSIFICATION_BACKEND_SCRIPT)],
            stdout=subprocess.PIPE
        )
        output = '\n'.join([x.decode() for x in backend_project_process.communicate() if x])
        logger.info(output)
        if 'ML Backend has been successfully initialized' not in output:
            raise ValueError(f'Label Studio ML has not been initialized.')

        if self.classification_model_spec is not None:
            with open(self.backend_project_directory / 'classification_model_spec.pkl', 'wb') as out:
                pickle.dump(classification_model_spec, out)
        with open(self.main_project_directory / 'config.json', 'r') as src:
            config_json = json.load(src)
        config_json['ml_backends'].append({
            'url': f'http://localhost:{backend_port}/',
            'name': 'main_model'
        })
        with open(self.main_project_directory / 'config.json', 'w') as out:
            json.dump(config_json, out, indent=4)

    def initialize_project(
        self,
        class_names: Union[str, Path, List[str]],
        can_rotate: bool = False,
        port: int = 8080,
        url: str = 'http://localhost:8080/',
        classification_model_spec: ClassificationModelSpec = None,
        backend_port: int = 9080,
        top_n: int = 1,
        label_to_description: Union[str, Path] = None,
        base_labels_images: Union[str, Path] = None,
        show_description: bool = False,
        show_base_labels_images: bool = False,
        show_top_n_labels_images: bool = False
    ):
        if self.directory.exists():
            raise ValueError(
                f'Directory {self.directory} is exists. Delete it before creating the project.'
            )

        if show_description:
            assert label_to_description is not None

        if show_base_labels_images or show_top_n_labels_images:
            assert base_labels_images is not None

        self.label_to_description = get_label_to_description(
            label_to_description_dict=label_to_description
        ) if show_description and label_to_description is not None else None

        logger.info(f'Initializing LabelStudioProject "{self.directory.name}"...')
        label_studio_process = subprocess.Popen(
            ["label-studio", "init", str(self.main_project_directory)],
            stdout=subprocess.PIPE
        )
        output = '\n'.join([x.decode() for x in label_studio_process.communicate() if x])

        self.cv_pipeliner_settings = {
            'port': port,
            'backend_port': backend_port,
            'url': url,  # use hack for our cluster
            'additional_info': 'This project was created by cv_pipeliner.utils.label_studio.',
            'top_n': top_n,
            'label_to_description': label_to_description,
            'base_labels_images': base_labels_images,
            'show_description': show_description,
            'show_base_labels_images': show_base_labels_images,
            'show_top_n_labels_images': show_top_n_labels_images
        }
        logger.info(output)
        if 'Label Studio has been successfully initialized' not in output:
            raise ValueError(f'Label Studio has not been initialized.')

        (self.main_project_directory / 'upload').mkdir()
        self.set_class_names(
            class_names=class_names, can_rotate=can_rotate
        )
        self._save_current_settings()
        self.initialize_backend(
            backend_port=backend_port,
            classification_model_spec=classification_model_spec
        )

    def run_backend(self):
        os.system(f"label-studio-ml start {self.backend_project_directory} -p {self.cv_pipeliner_settings['backend_port']}")

    def run_app(self):
        os.system(f"label-studio start {self.main_project_directory} -p {self.cv_pipeliner_settings['port']}")

    def run_project(self):
        try:
            logger.info(
                f'Start project "{self.directory.name}": {self.cv_pipeliner_settings["url"]}'
            )
            processes = [
                Process(target=self.run_backend),
                Process(target=self.run_app)
            ]
            for p in processes:
                p.start()
        finally:
            for p in processes:
                p.join()

    def load(self):
        if (self.main_project_directory / 'cv_pipeliner_settings.json').exists():
            logger.info(f'Loading LabelStudioProject "{self.directory.name}"...')
            with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'r') as src:
                self.cv_pipeliner_settings = json.load(src)
            with open(self.main_project_directory / 'class_names.json', 'r') as src:
                self.class_names = json.load(src)
            self.tasks_data = load_tasks(self.main_project_directory)
            self.label_to_description = get_label_to_description(
                label_to_description_dict=self.cv_pipeliner_settings['label_to_description']
            ) if (
                self.cv_pipeliner_settings['show_description']
                and
                self.cv_pipeliner_settings['label_to_description'] is not None
            ) else None

        if (self.backend_project_directory / 'classification_model_spec.pkl').exists():
            with open(self.backend_project_directory / 'classification_model_spec.pkl', 'rb') as src:
                self.classification_model_spec = pickle.load(src)
        else:
            self.classification_model_spec = None

    def _append_GT_to_bbox_data_as_cropped_image(
        self,
        bbox_data_as_cropped_image: np.ndarray,
        label_to_base_label_image: Callable[[str], np.ndarray]
    ) -> BboxData:
        bbox_data_as_cropped_image = copy.deepcopy(bbox_data_as_cropped_image)
        image_GT = label_to_base_label_image(bbox_data_as_cropped_image.label)
        image_GT = np.pad(
            image_GT,
            pad_width=((60, 0), (0, 0), (0, 0)),
            constant_values=((0, 0), (0, 0), (0, 0)),
            mode='constant'
        )
        image_GT = put_text_on_image(
            image=image_GT,
            text='GT',
            fontsize=60,
            ymax=5,
            maximum_width=14
        )
        cropped_image = bbox_data_as_cropped_image.open_image()
        ha, wa = cropped_image.shape[:2]
        hb, wb = image_GT.shape[:2]
        max_height = np.max([ha, hb])
        min_ha = max_height // 2 - ha // 2
        total_image_with_cropped_image = concat_images(
            image_a=cropped_image,
            image_b=image_GT,
            how='horizontally'
        )
        bbox_data_as_cropped_image.image = total_image_with_cropped_image
        bbox_data_as_cropped_image.ymin += min_ha
        bbox_data_as_cropped_image.ymax += min_ha
        bbox_data_as_cropped_image.additional_info['src_ymin'] -= int(min_ha)
        return bbox_data_as_cropped_image

    def _append_top_n_labels_to_bbox_data_as_cropped_image(
        self,
        bbox_data_as_cropped_image: np.ndarray,
        label_to_base_label_image: Callable[[str], np.ndarray]
    ) -> BboxData:
        bbox_data_as_cropped_image = copy.deepcopy(bbox_data_as_cropped_image)
        total_image = draw_n_base_labels_images(
            labels=bbox_data_as_cropped_image.labels_top_n,
            label_to_base_label_image=label_to_base_label_image
        )
        cropped_image = bbox_data_as_cropped_image.open_image()
        ha, wa = cropped_image.shape[:2]
        hb, wb = total_image.shape[:2]
        max_width = np.max([wa, wb])
        min_wa = max_width // 2 - wa // 2
        total_image_with_cropped_image = concat_images(
            image_a=cropped_image,
            image_b=total_image,
            how='vertically'
        )
        bbox_data_as_cropped_image.image = total_image_with_cropped_image
        bbox_data_as_cropped_image.xmin += min_wa
        bbox_data_as_cropped_image.xmax += min_wa
        bbox_data_as_cropped_image.additional_info['src_xmin'] -= int(min_wa)
        return bbox_data_as_cropped_image

    def add_tasks(
        self,
        image_paths: List[Union[str, Path]],
        n_bboxes_data: List[List[BboxData]],
        default_class_name: str = None,
        batch_size: int = 16,
        bbox_offset: int = 150,
        show_ground_truth: bool = False
    ):
        with open(self.main_project_directory / 'tasks.json', 'r') as src:
            tasks_json = json.load(src)
        tasks_ids = [tasks_json[task_id]['id'] for task_id in tasks_json]
        id = max(tasks_ids) + 1 if len(tasks_ids) > 0 else 0
        logger.info('Adding tasks...')
        if self.cv_pipeliner_settings['show_top_n_labels_images']:
            assert self.classification_model_spec is not None
            label_to_base_label_image = get_label_to_base_label_image(
                base_labels_images=self.cv_pipeliner_settings['base_labels_images'],
                label_to_description=self.cv_pipeliner_settings['label_to_description'],
                add_label_to_image=True,
                make_labels_for_these_class_names_too=list(
                    map(self._class_name_without_special_character, self.class_names)
                )
            ) if self.cv_pipeliner_settings['base_labels_images'] is not None else None
        if self.classification_model_spec is not None:
            n_pred_bboxes_data = self.inference_and_make_predictions_for_backend(
                image_paths=image_paths,
                n_bboxes_data=n_bboxes_data,
                default_class_name=default_class_name,
                batch_size=batch_size
            )
            for true_bboxes_data, pred_bboxes_data in zip(n_bboxes_data, n_pred_bboxes_data):
                for true_bbox_data, pred_bbox_data in zip(true_bboxes_data, pred_bboxes_data):
                    true_bbox_data.top_n = pred_bbox_data.top_n
                    true_bbox_data.labels_top_n = pred_bbox_data.labels_top_n
        tasks_images = set(tasks_json[task_id]['data']['image'] for task_id in tasks_json)
        for image_path, bboxes_data in tqdm(list(zip(image_paths, n_bboxes_data))):
            source_image = imageio.imread(image_path, pilmode='RGB')
            for bbox_data in bboxes_data:
                bbox = (bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)
                filename = f"{image_path.stem}_{bbox}.png"
                image = (
                    urljoin(self.cv_pipeliner_settings['url'], f"data/upload/{filename}")
                )
                if str(image) in tasks_images:
                    logger.info(f"Task with filename {filename} is already exists. Skipping...")
                    continue
                tasks_images.add(str(image))
                bbox_data_as_cropped_image = bbox_data.open_cropped_image(
                    source_image=source_image,
                    xmin_offset=bbox_offset,
                    ymin_offset=bbox_offset,
                    xmax_offset=bbox_offset,
                    ymax_offset=bbox_offset,
                    draw_rectangle_with_color=[0, 255, 0],
                    return_as_bbox_data_in_cropped_image=True
                )
                if show_ground_truth:
                    bbox_data_as_cropped_image = self._append_GT_to_bbox_data_as_cropped_image(
                        bbox_data_as_cropped_image=bbox_data_as_cropped_image,
                        label_to_base_label_image=label_to_base_label_image
                    )
                if self.cv_pipeliner_settings['show_top_n_labels_images']:
                    bbox_data_as_cropped_image = self._append_top_n_labels_to_bbox_data_as_cropped_image(
                        bbox_data_as_cropped_image=bbox_data_as_cropped_image,
                        label_to_base_label_image=label_to_base_label_image
                    )
                cropped_image = bbox_data_as_cropped_image.open_image()
                cropped_image_path = self.main_project_directory / 'upload' / filename
                Image.fromarray(cropped_image).save(cropped_image_path)
                bbox_data.label = self._class_name_with_special_character(bbox_data.label)
                bbox_data_as_cropped_image.label = self._class_name_with_special_character(
                    bbox_data_as_cropped_image.label
                )
                bbox_data_as_cropped_image.image_path = cropped_image_path
                tasks_json[str(id)] = {
                    'id': id,
                    'data': {
                        'image': str(image),
                        'src_image_path': str(image_path),
                        'src_bbox_data': bbox_data.asdict(),
                        'cropped_image_path': str(cropped_image_path),
                        'bbox_data_as_cropped_image': bbox_data_as_cropped_image.asdict()
                    }
                }
                id += 1
                with open(self.main_project_directory / 'tasks.json', 'w') as out:
                    json.dump(tasks_json, out, indent=4)
        load_tasks(self.main_project_directory)

    def get_ready_images_data(
        self,
        get_only_done: bool = True,
        get_only_not_skipped: bool = True
    ) -> List[ImageData]:
        with open(self.main_project_directory / 'tasks.json', 'r') as src:
            tasks_json = json.load(src)
        image_paths = list(set([str(tasks_json[key]['data']['src_image_path']) for key in tasks_json]))
        bboxes_data = np.array([
            task_data.bbox_data
            for task_data in self.tasks_data
            if (
                (get_only_done and task_data.is_done) or (not get_only_done)
                and
                (get_only_not_skipped and not task_data.is_skipped) or (not get_only_not_skipped)
            )
        ])
        image_paths_in_bboxes_data = np.array([
            str(bbox_data.image_path)
            for bbox_data in bboxes_data
        ])
        images_data = [
            ImageData(
                image_path=image_path,
                bboxes_data=[
                    BboxData(
                        image_path=bbox_data.image_path,
                        xmin=bbox_data.xmin,
                        ymin=bbox_data.ymin,
                        xmax=bbox_data.xmax,
                        ymax=bbox_data.ymax,
                        angle=bbox_data.angle,
                        label=(
                            self._class_name_without_special_character(bbox_data.label)
                        ),
                        top_n=bbox_data.top_n,
                        labels_top_n=bbox_data.labels_top_n,
                        additional_info=bbox_data.additional_info
                    )
                    for bbox_data in bboxes_data[image_paths_in_bboxes_data == image_path]
                ]
            )
            for image_path in image_paths
        ]
        return images_data
