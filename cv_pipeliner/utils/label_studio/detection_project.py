
import os
import stat
import json
import subprocess
import shutil

from pathlib import Path
from typing import List, Union, Dict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.logging import logger

from cv_pipeliner.utils.images import rotate_point


@dataclass
class TaskData:
    task_json: Dict
    id: int
    image_data: ImageData
    is_done: bool = False
    is_skipped: bool = False
    is_trash: bool = False

    def _parse_rectangle_labels(
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
            label=label
        )
        return bbox_data

    def _parse_completion(
        self,
        completions_json: Dict
    ):
        self.is_done = True
        bboxes_data = []
        additional_info = {}

        if len(completions_json['completions']) > 1:
            raise ValueError(
                f'Find a task with two or more completions. Task_id: {self.id}'
            )
        completion = completions_json['completions'][0]
        image_path = Path(completions_json['data']['src_image_path'])
        if 'skipped' in completion and completion['skipped']:
            self.is_skipped = True
        else:
            for result in completion['result']:
                from_name = result['from_name']
                if from_name == 'bbox':
                    bboxes_data.append(
                        self._parse_rectangle_labels(
                            result=result,
                            image_path=image_path
                        )
                    )
                elif from_name == 'trash':
                    if 'Trash' in result['value']['choices']:
                        self.is_trash = True
                        additional_info['is_trash'] = True

        return ImageData(
            image_path=image_path,
            bboxes_data=bboxes_data,
            additional_info=additional_info
        )

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
            image_path = Path(task_json['data']['src_image_path'])
            self.image_data = ImageData(
                image_path=image_path
            )


SPECIAL_CHARACTER = '⠀'  # Label studio can't accept class_names when it's number, so we add this invisible character.


class LabelStudioProject:
    def __init__(
        self,
        directory: Union[str, Path]
    ):
        self.directory = Path(directory)
        self.main_project_directory = self.directory / 'main_project'
        self.load()

    def generate_config_xml(
        self,
        class_names: List[str]
    ):
        labels = '\n'.join(
            [f'    <Label value="{class_name}"/>' for class_name in class_names]
        )
        config_xml = f'''
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image">
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
        class_names: List[str]
    ):
        self.class_names = [
            f"{class_name}{SPECIAL_CHARACTER}" if class_name.isdigit() else class_name
            for class_name in class_names
        ]
        with open(self.main_project_directory/'class_names.json', 'w') as out:
            json.dump(self.class_names, out, indent=4)
        self.generate_config_xml(self.class_names)

    def initialize_project(
        self,
        class_names: List[str],
        port: int = 8080,
        url: str = 'http://localhost:8080/'
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
            'url': url,  # use hack for our cluster
            'additional_info': 'This project was created by cv_pipeliner.utils.label_studio.'
        }
        if 'Label Studio has been successfully initialized' in output:
            logger.info(f'Initializing LabelStudioProject "{self.directory.name}"...')
            self.set_class_names(class_names)
            (self.main_project_directory / 'upload').mkdir()
            with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'w') as out:
                json.dump(self.cv_pipeliner_settings, out, indent=4)

            with open(self.directory / 'run.sh', 'w') as out:
                out.write(f'label-studio start {self.main_project_directory} -p {port}')
            st = os.stat(self.directory / 'run.sh')  # chmod +x
            os.chmod(self.directory / 'run.sh', st.st_mode | stat.S_IEXEC)

    def _load_tasks(self):
        with open(self.main_project_directory/'tasks.json', 'r') as src:
            tasks_json = json.load(src)
        tasks_json = [tasks_json[key] for key in tasks_json]
        self.tasks_data = [
            TaskData(
                main_project_directory=self.main_project_directory,
                task_json=task_json
            )
            for task_json in tasks_json
        ]

    def load(self):
        if (self.main_project_directory / 'cv_pipeliner_settings.json').exists():
            logger.info(f'Loading LabelStudioProject "{self.directory.name}"...')
            with open(self.main_project_directory / 'cv_pipeliner_settings.json', 'r') as src:
                self.cv_pipeliner_settings = json.load(src)
            with open(self.main_project_directory/'class_names.json', 'r') as src:
                self.class_names = json.load(src)
            self._load_tasks()

    def add_tasks(
        self,
        image_paths: List[Union[str, Path]]
    ):
        with open(self.main_project_directory/'tasks.json', 'r') as src:
            tasks_json = json.load(src)
        tasks_ids = [tasks_json[key]['id'] for key in tasks_json]
        start = max(tasks_ids) if len(tasks_ids) > 0 else 0
        logger.info('Adding tasks...')
        for id, image_path in tqdm(list(enumerate(image_paths, start=start))):
            image = (
                f"{self.cv_pipeliner_settings['url']}/data/upload/{image_path.name}"  # noqa: E501
            )
            shutil.copy(image_path, self.main_project_directory / 'upload' / f'{image_path.name}')
            tasks_json[str(id)] = {
                'id': id,
                'data': {
                    'image': str(image),
                    'src_image_path': str(image_path)
                },
            }
        with open(self.main_project_directory/'tasks.json', 'w') as out:
            json.dump(tasks_json, out, indent=4)
        self._load_tasks()

    def get_ready_images_data(self) -> List[ImageData]:
        images_data = [
            task_data.image_data
            for task_data in self.tasks_data
            if task_data.is_done and not task_data.is_skipped
        ]
        return images_data
