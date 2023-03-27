import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union, Tuple

import contextlib2
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util as creation_util

from google.protobuf import text_format

from object_detection.utils.config_util import (
    create_pipeline_proto_from_configs,
    get_configs_from_pipeline_file,
    save_pipeline_config,
)
from object_detection.protos import pipeline_pb2
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem

from cv_pipeliner.core.data import ImageData

logger = logging.getLogger(__name__)


def create_tf_record(
    height: int,
    width: int,
    encoded_filename: bytes,
    encoded_jpg: bytes,
    image_format: bytes,
    xmins: List[float],
    ymins: List[float],
    xmaxs: List[float],
    ymaxs: List[float],
    encoded_class_names: List[bytes],
    classes: int,
) -> tf.train.Example:
    tf_record = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(encoded_filename),
                "image/source_id": dataset_util.bytes_feature(encoded_filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(encoded_class_names),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_record


def tf_record_from_image_data(image_data: ImageData, label_map: Dict[str, int], use_thumbnail: Tuple[int, int] = None):
    filename = image_data.image_path
    encoded_filename = str(filename).encode("utf8")

    normalized_true_bboxes = np.array([bbox_data.coords_n for bbox_data in image_data.bboxes_data], dtype=float)
    image = image_data.open_image()
    width, height = image_data.get_image_size()
    if len(normalized_true_bboxes) > 0:
        xmins = normalized_true_bboxes[:, 0]
        ymins = normalized_true_bboxes[:, 1]
        xmaxs = normalized_true_bboxes[:, 2]
        ymaxs = normalized_true_bboxes[:, 3]
    else:
        ymins, xmins, ymaxs, xmaxs = [], [], [], []

    encoded_jpg = BytesIO()
    image = Image.fromarray(image)
    if use_thumbnail:
        image.thumbnail(use_thumbnail)
    image.save(encoded_jpg, format="JPEG")
    encoded_jpg = encoded_jpg.getvalue()
    image_format = b"jpg"

    class_names = [bbox_data.label for bbox_data in image_data.bboxes_data]
    encoded_class_names = [class_name.encode("utf-8") for class_name in class_names]
    classes = [label_map[class_name] for class_name in class_names]

    tf_record = create_tf_record(
        height=height,
        width=width,
        encoded_filename=encoded_filename,
        encoded_jpg=encoded_jpg,
        image_format=image_format,
        xmins=xmins,
        ymins=ymins,
        xmaxs=xmaxs,
        ymaxs=ymaxs,
        encoded_class_names=encoded_class_names,
        classes=classes,
    )
    return tf_record


def convert_to_tf_records(
    images_data: List[ImageData],
    label_map: Dict[str, int],
    filepath: Union[str, Path],
    num_workers: int = 1,
    num_shards: int = 2,
    max_pictures_per_worker: int = 1000,
    use_thumbnail: Tuple[int, int] = None,
):
    logger.info("Create tf_records from data.")

    data_chunks = np.array_split(images_data, max_pictures_per_worker)
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = creation_util.open_sharded_output_tfrecords(
            exit_stack=tf_record_close_stack, base_path=filepath, num_shards=num_shards
        )
        for data_chunk in tqdm(data_chunks):
            tf_records = Parallel(n_jobs=num_workers, prefer="threads")(
                delayed(tf_record_from_image_data)(
                    image_data=image_data, label_map=label_map, use_thumbnail=use_thumbnail
                )
                for image_data in data_chunk
            )

            for index, tf_record in enumerate(tf_records):
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_record.SerializeToString())

    logger.info(f"tf_records saved to {filepath}-?????-of-{str(num_shards).zfill(5)}.")


def label_map_to_file(label_map: Dict[str, int], filepath: Union[str, Path]):
    msg = StringIntLabelMap()
    label_map = {label: i for label, i in sorted(label_map.items(), key=lambda item: item[1])}
    for label, i in label_map.items():
        # pylint: disable=no-member
        msg.item.append(StringIntLabelMapItem(id=i, name=label))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), "utf-8")
    with open(filepath, "w") as out:
        out.write(text)
    logger.info(f"label_map saved to {filepath}")


def count_tfrecord_examples(filepath: Union[str, Path]):
    filepath = Path(filepath)
    if filepath.exists():
        count = sum(1 for _ in tf.data.TFRecordDataset(str(filepath)))
    else:
        filepaths = sorted(filepath.parent.glob(f"{filepath.name}*-of-*"))
        count = sum(1 for filepath in filepaths for _ in tf.data.TFRecordDataset(str(filepath)))
    return count


def set_config(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    tf_records_train_path: Union[str, Path],
    label_map: Dict[str, int],
    label_map_filepath: Union[str, Path],
    batch_size: int,
    max_box_predictions: int,
    max_number_of_boxes: int,
    fine_tune_checkpoint_type: str = "detection",
    augment_path: str = None,
    min_dimension: int = None,
    max_dimension: int = None,
    total_steps: int = None,
    warmup_steps: int = None,
    num_steps: int = None,
):
    logger.info(f"Set configs {config_path}...")

    configs = get_configs_from_pipeline_file(str(config_path))

    train_len = count_tfrecord_examples(str(tf_records_train_path))
    logger.info(f"Train has {train_len} tf_records.")
    num_classes = len(set(label_map.values()))
    _, config_model = configs["model"].ListFields()[0]
    config_model.num_classes = num_classes

    configs["model"].center_net.object_center_params.max_box_predictions = max_box_predictions
    if min_dimension is not None:
        configs["model"].center_net.image_resizer.keep_aspect_ratio_resizer.min_dimension = min_dimension
    if max_dimension is not None:
        configs["model"].center_net.image_resizer.keep_aspect_ratio_resizer.max_dimension = max_dimension

    configs["train_config"].fine_tune_checkpoint_type = fine_tune_checkpoint_type
    configs["train_config"].fine_tune_checkpoint = str(checkpoint_path)
    configs["train_config"].batch_size = batch_size

    configs["train_config"].max_number_of_boxes = max_number_of_boxes
    if total_steps is not None:
        configs[
            "train_config"
        ].optimizer.adam_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = total_steps
    if warmup_steps is not None:
        configs[
            "train_config"
        ].optimizer.adam_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = warmup_steps
    if num_steps is not None:
        configs["train_config"].num_steps = num_steps

    if augment_path is not None:
        augment_config = configs["train_config"].data_augmentation_options
        for _ in augment_config:
            augment_config.pop()
        augment = text_format.Merge(str(augment_path), pipeline_pb2.TrainEvalPipelineConfig())
        augment_config.extend(augment.train_config.data_augmentation_options)

    label_map_to_file(label_map=label_map, filepath=label_map_filepath)

    def clear_repeated_proto(proto):
        for _ in proto:
            proto.pop()

    configs["train_input_config"].label_map_path = str(label_map_filepath)
    clear_repeated_proto(configs["train_input_config"].tf_record_input_reader.input_path)
    configs["train_input_config"].tf_record_input_reader.input_path.append(str(tf_records_train_path))

    pipeline_proto = create_pipeline_proto_from_configs(configs)
    save_pipeline_config(pipeline_proto, str(config_path.parent))
    logger.info(f"Config {config_path} changed")
