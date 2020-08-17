import tarfile
import shutil

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union

import requests

from tqdm import tqdm
from two_stage_pipeliner.logging import logger


@dataclass
class DetectorModelSpecTF:
    name: str
    input_size: int
    model_url: str
    fine_tune_checkpoint_type: str = None
    config_path: Union[str, Path] = None
    checkpoint_filename: Union[str, Path] = None


ZOO_MODELS_DIR = Path(__file__).parent / 'tf_zoo_models'
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


def download_model(model_name: str, model_url: str):
    logger.info(
        f"Download model '{model_name}' from url '{model_url}'"
    )
    filepath = ZOO_MODELS_DIR / Path(model_url).name
    folder = Path(str(filepath).split('.tar.gz')[0])
    download(model_url, filepath)

    tar = tarfile.open(filepath)
    tar.extractall(path=ZOO_MODELS_DIR)
    tar.close()

    filepath.unlink()
    shutil.move(folder, folder.parent / model_name)
    logger.info(
        f"Saved to '{folder.parent / model_name}'"
    )


name_to_model_spec: Dict[str, DetectorModelSpecTF] = {
    spec.name: spec for spec in [
        DetectorModelSpecTF(
            name='ssd_mobilenet_v2_320x320_coco17_tpu-8',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
            input_size=320,
            fine_tune_checkpoint_type='detection'
        ),
        DetectorModelSpecTF(
            name='centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz',
            input_size=512,
            fine_tune_checkpoint_type='fine_tune'
        ),
        DetectorModelSpecTF(
            name='centernet_hg104_512x512_coco17_tpu-8',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
            input_size=512,
            fine_tune_checkpoint_type='detection'
        ),
        DetectorModelSpecTF(
            name='ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
            input_size=640,
            fine_tune_checkpoint_type='detection'
        ),
        DetectorModelSpecTF(
            name='efficientdet_d0_coco17_tpu-32',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
            input_size=512,
            fine_tune_checkpoint_type='detection'
        ),
        DetectorModelSpecTF(
            name='efficientdet_d1_coco17_tpu-32',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
            input_size=640,
            fine_tune_checkpoint_type='detection'
        ),
        DetectorModelSpecTF(
            name='centernet_resnet50_v1_fpn_512x512_coco17_tpu-8',
            model_url='http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
            input_size=512,
            fine_tune_checkpoint_type='detection'
        ),
    ]
}


def load_detector_model_spec_tf(
    model_name: str,
    model_dir: Union[str, Path] = None,
    checkpoint_filename: str = None
) -> DetectorModelSpecTF:
    model_spec = name_to_model_spec[model_name]

    if model_dir is None:
        model_dir = ZOO_MODELS_DIR / model_name
        if not model_dir.exists():
            download_model(model_name, model_spec.model_url)

    model_dir = Path(model_dir)
    model_spec.model_dir = model_dir
    model_spec.config_path = model_dir / 'pipeline.config'
    if checkpoint_filename is None:
        if (model_spec.model_dir / 'checkpoint/ckpt-0.index').exists():
            # default checkpoint
            model_spec.checkpoint_filename = 'ckpt-0'
        else:
            # first saved checkpoint in training
            model_spec.checkpoint_filename = 'ckpt-1'
    else:
        model_spec.checkpoint_filename = checkpoint_filename

    return model_spec
