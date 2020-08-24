import copy
import tarfile
import shutil

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, Tuple, ClassVar

import requests

from tqdm import tqdm
from two_stage_pipeliner.logging import logger

from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec


@dataclass
class DetectionModelSpecTF(DetectionModelSpec):
    name: str
    input_size: Tuple[int, int]
    standard_model_url: str
    config_path: Union[str, Path] = None
    checkpoint_path: Union[str, Path] = None

    @property
    def inference_model(self) -> ClassVar['DetectionModelTF']:
        from two_stage_pipeliner.inference_models.detection.tf.detector import DetectionModelTF
        return DetectionModelTF


ZOO_MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / 'object_detection_api_zoo_models'
ZOO_MODELS_DIR.mkdir(exist_ok=True)


def download(url: str, filepath: Union[str, Path]):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(filepath, 'wb') as file, tqdm(
            desc=str(filepath),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as tbar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            tbar.update(size)


def download_model(model_name: str, standard_model_url: str):
    logger.info(
        f"Download model '{model_name}' from url '{standard_model_url}'"
    )
    filepath = ZOO_MODELS_DIR / Path(standard_model_url).name
    folder = Path(str(filepath).split('.tar.gz')[0])
    download(standard_model_url, filepath)

    tar = tarfile.open(filepath)
    tar.extractall(path=ZOO_MODELS_DIR)
    tar.close()

    filepath.unlink()
    shutil.move(folder, folder.parent / model_name)
    logger.info(
        f"Saved to '{folder.parent / model_name}'"
    )


spec_name_to_detection_model_spec_tf: Dict[str, DetectionModelSpecTF] = {
    spec.name: spec for spec in [
        DetectionModelSpecTF(
            name='ssd_mobilenet_v2_320x320_coco17_tpu-8',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
            input_size=(320, 320),
        ),
        DetectionModelSpecTF(
            name='centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz',
            input_size=(512, 512),
        ),
        DetectionModelSpecTF(
            name='centernet_hg104_512x512_coco17_tpu-8',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
            input_size=(512, 512),
        ),
        DetectionModelSpecTF(
            name='ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            input_size=(640, 640),
        ),
        DetectionModelSpecTF(
            name='efficientdet_d0_coco17_tpu-32',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
            input_size=(512, 512),
        ),
        DetectionModelSpecTF(
            name='efficientdet_d1_coco17_tpu-32',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
            input_size=(640, 640),
        ),
        DetectionModelSpecTF(
            name='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8',
            standard_model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
            input_size=(512, 512),
        ),
    ]
}


def load_detection_model_spec_tf_from_standard_list_of_models_specs(
    spec_name: str,
    config_path: Union[str, Path] = None,
    checkpoint_path: Union[str, Path] = None,
) -> DetectionModelSpecTF:

    model_spec = copy.deepcopy(spec_name_to_detection_model_spec_tf[spec_name])

    if config_path is None and checkpoint_path is None:
        model_dir = ZOO_MODELS_DIR / spec_name
        if not model_dir.exists():
            download_model(spec_name, model_spec.standard_model_url)
        model_spec.config_path = model_dir / 'pipeline.config'
        model_spec.checkpoint_path = model_dir / 'checkpoint/ckpt-0.index'
    else:
        model_spec.config_path = config_path
        model_spec.checkpoint_path = checkpoint_path

    return model_spec
