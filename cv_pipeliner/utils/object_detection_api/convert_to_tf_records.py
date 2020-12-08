import contextlib2
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

from object_detection.utils import dataset_util
from object_detection.dataset_tools import \
    tf_record_creation_util as creation_util

from crpt_ml.image_utils import encode_image, normalize_bbox, \
                                open_and_normalize_image, \
                                resize_keep_ratio

from crpt_ml.logging_utils import logger


def create_tf_record(height,
                     width,
                     encoded_filename,
                     encoded_jpg,
                     image_format,
                     xmins,
                     xmaxs,
                     ymins,
                     ymaxs,
                     classes_text,
                     classes):
    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_filename),
        'image/source_id': dataset_util.bytes_feature(encoded_filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text
        ),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record


def tf_record_from_image_data(image_data, model_spec,
                              label_map, class_name_to_label,
                              do_resize_keep_ratio):
    filename = image_data.image_path
    encoded_filename = str(filename).encode('utf8')

    image = open_and_normalize_image(filename)

    true_bboxes = image_data.coords
    normalized_true_bboxes = normalize_bbox(image, true_bboxes)

    if len(normalized_true_bboxes) > 0:
        ymins = normalized_true_bboxes[:, 0]
        xmins = normalized_true_bboxes[:, 1]
        ymaxs = normalized_true_bboxes[:, 2]
        xmaxs = normalized_true_bboxes[:, 3]
    else:
        ymins, xmins, ymaxs, xmaxs = [], [], [], []

    if do_resize_keep_ratio:
        image = resize_keep_ratio(image, model_spec.size)
    height, width = image.height, image.width
    encoded_jpg = encode_image(image)
    image_format = b'jpg'

    class_names = image_data.labels
    default_label_map = 'Label'  # TODO: Delete this
    classes_text = [
        class_name_to_label.get(class_name, default_label_map).encode('utf-8')
        for class_name in class_names
    ]
    classes = [
        label_map[label.decode()]
        for label in classes_text
    ]

    tf_record = create_tf_record(
        height, width, encoded_filename, encoded_jpg,
        image_format, xmins, xmaxs, ymins, ymaxs,
        classes_text, classes
    )
    return tf_record


def convert_to_tf_records(data,
                          model_spec,
                          label_map,
                          class_name_to_label,
                          filepath,
                          num_workers=1,
                          num_shards=2,
                          max_pictures_per_worker=1000,
                          do_resize_keep_ratio=False):
    logger.info(f'Create tf_records from data.')

    data_chunks = np.array_split(data, max_pictures_per_worker)
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, filepath, num_shards
        )
        for data_chunk in tqdm(data_chunks):
            tf_records = Parallel(n_jobs=num_workers)(
                delayed(tf_record_from_image_data)(
                    image_data, model_spec, label_map, class_name_to_label,
                    do_resize_keep_ratio
                )
                for image_data in data_chunk
            )

            for index, tf_record in enumerate(tf_records):
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(
                    tf_record.SerializeToString()
                )

    logger.info(
        f"tf_records saved to {filepath}-?????-of-{str(num_shards).zfill(5)}."
    )
