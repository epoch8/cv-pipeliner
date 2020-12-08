import copy
import tensorflow as tf

from google.protobuf import text_format

from object_detection.utils.config_util import \
    create_pipeline_proto_from_configs, \
    get_configs_from_pipeline_file, save_pipeline_config
from object_detection.protos import pipeline_pb2
from object_detection.protos.string_int_label_map_pb2 import \
    StringIntLabelMap, StringIntLabelMapItem

from crpt_ml.logging_utils import logger


def label_map_to_text(label_map, filepath):
    msg = StringIntLabelMap()
    label_map = {
        label: i
        for label, i in sorted(label_map.items(), key=lambda item: item[1])
    }
    for label, i in label_map.items():
        # pylint: disable=no-member
        msg.item.append(
            StringIntLabelMapItem(id=i, name=label)
        )

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    with open(filepath, 'w') as out:
        out.write(text)
    logger.info(f'label_map saved to {filepath}')


def count_tfrecord_examples(filepath):
    if filepath.exists():
        count = sum(1 for _ in tf.data.TFRecordDataset(str(filepath)))
    else:
        filepaths = sorted(filepath.parent.glob(f"{filepath.name}*-of-*"))
        count = sum(
            1 for filepath in filepaths
            for _ in tf.data.TFRecordDataset(str(filepath))
        )
    return count


def clear_repeated_proto(proto):
    for _ in proto:
        proto.pop()


def set_config(model_spec,
               tf_records_train_path,
               tf_records_test_path,
               tf_records_val_path,
               label_map,
               batch_size,
               directory,
               num_shards,
               num_readers=8,
               num_visualizations=50,
               eval_interval_secs=300,
               augment=None):
    logger.info(f"Set configs for model {model_spec.name}")

    configs = get_configs_from_pipeline_file(
        str(model_spec.config_path)
    )

    assert len(configs['model'].ListFields()) == 1
    assert len(configs['eval_input_configs']) == 1

    train_len = count_tfrecord_examples(tf_records_train_path)
    logger.info(f"Train has {train_len} tf_records.")
    test_len = count_tfrecord_examples(tf_records_test_path)
    logger.info(f"Train has {test_len} tf_records.")
    val_len = count_tfrecord_examples(tf_records_val_path)
    logger.info(f"Train has {val_len} tf_records.")

    num_classes = len(set(label_map.values()))
    _, config_model = configs['model'].ListFields()[0]
    config_model.num_classes = num_classes

    configs['train_config'].fine_tune_checkpoint = str(
        model_spec.model_dir / 'checkpoint' / model_spec.checkpoint_filename
    )
    configs['train_config'].batch_size = batch_size
    configs['train_config'].fine_tune_checkpoint_type = model_spec.fine_tune_checkpoint_type
    if augment:
        augment_config = configs['train_config'].data_augmentation_options
        for _ in augment_config:
            augment_config.pop()
        augment = text_format.Merge(
            augment, pipeline_pb2.TrainEvalPipelineConfig()
        )
        augment_config.extend(
            augment.train_config.data_augmentation_options
        )

    label_map_filepath = directory / 'label_map.txt'
    label_map_to_text(label_map, label_map_filepath)

    configs['train_input_config'].label_map_path = str(label_map_filepath)
    clear_repeated_proto(configs[
        'train_input_config'].tf_record_input_reader.input_path
    )
    configs[
        'train_input_config'].tf_record_input_reader.input_path.append(
            f"{tf_records_train_path}-?????-of-{str(num_shards).zfill(5)}"
            )

    eval_config = configs['eval_config']
    eval_config.num_examples = max(test_len, val_len)
    eval_config.num_visualizations = num_visualizations
    eval_config.eval_interval_secs = eval_interval_secs
    clear_repeated_proto(eval_config.metrics_set)
    eval_config.metrics_set.extend([
        "coco_detection_metrics", "precision_at_recall_detection_metrics"
    ])

    test_eval_config = configs['eval_input_config']
    val_eval_config = copy.deepcopy(configs['eval_input_config'])

    test_eval_config.label_map_path = str(label_map_filepath)
    clear_repeated_proto(test_eval_config.tf_record_input_reader.input_path)
    test_eval_config.tf_record_input_reader.input_path.append(
        f"{tf_records_test_path}-?????-of-{str(num_shards).zfill(5)}"
    )
    test_eval_config.num_readers = num_readers

    val_eval_config.label_map_path = str(label_map_filepath)
    clear_repeated_proto(val_eval_config.tf_record_input_reader.input_path)
    val_eval_config.tf_record_input_reader.input_path.append(
        f"{tf_records_val_path}-?????-of-{str(num_shards).zfill(5)}"
    )
    val_eval_config.num_readers = num_readers

    configs['eval_input_configs'] = [test_eval_config, val_eval_config]

    pipeline_proto = create_pipeline_proto_from_configs(configs)

    save_pipeline_config(pipeline_proto, str(directory))
    logger.info(f"Detector train configs saved to '{directory}'.")
