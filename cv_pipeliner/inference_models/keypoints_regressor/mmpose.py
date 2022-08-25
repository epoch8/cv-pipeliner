import tempfile
from dataclasses import dataclass
from typing import Tuple, Union, Type

import numpy as np
import fsspec
from pathy import Pathy

from cv_pipeliner.inference_models.keypoints_regressor.core import (
    KeypointsRegressorModelSpec, KeypointsRegressorModel, KeypointsRegressorInput, KeypointsRegressorOutput
)
from cv_pipeliner.inference_models.keypoints_regressor.utils.mmpose_utils import (
    bbox_xywh2cs, top_down_affine, preprocess, keypoints_from_heatmaps
)


class MMPose_KeypointsRegressorModelSpec_TFLite(KeypointsRegressorModelSpec):
    model_path: Union[str, Pathy]  # can be also tf.keras.Model

    @property
    def inference_model_cls(self) -> Type['MMPose_KeypointsRegressorModel']:
        from cv_pipeliner.inference_models.keypoints_regressor.mmpose import MMPose_KeypointsRegressorModel
        return MMPose_KeypointsRegressorModel


INPUT_TYPE_TO_DTYPE = {
    "image_tensor": np.uint8,
    "float_image_tensor": np.float32,
    "encoded_image_string_tensor": np.uint8
}


class MMPose_KeypointsRegressorModel(KeypointsRegressorModel):
    def _load_tensorflow_KeypointsRegressor_model_spec(
        self,
        model_spec: MMPose_KeypointsRegressorModelSpec_TFLite
    ):
        import tensorflow as tf
        model_openfile = fsspec.open(model_spec.model_path, 'rb')
        temp_file = tempfile.NamedTemporaryFile()
        with model_openfile as src:
            temp_file.write(src.read())
        model_path = Pathy(temp_file.name)
        temp_files_cleanup = temp_file.close

        self.model = tf.lite.Interpreter(str(model_path))
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()[0]
        self.input_index = self.input_details['index']
        self.input_dtype = self.input_details['dtype']
        self.output_index = self.model.get_output_details()[0]['index']

        temp_files_cleanup()

    def __init__(
        self,
        model_spec: MMPose_KeypointsRegressorModelSpec_TFLite
    ):
        super().__init__(model_spec)

        if isinstance(model_spec, MMPose_KeypointsRegressorModelSpec_TFLite):
            self._load_tensorflow_KeypointsRegressor_model_spec(model_spec)
            self._raw_predict = self._raw_predict_tensorflow
        else:
            raise ValueError(
                f"MMPose_KeypointsRegressorModel got unknown MMPose_KeypointsRegressorModelSpec: {type(model_spec)}"
            )

    def _raw_predict_tensorflow(
        self,
        image: np.ndarray
    ):
        import tensorflow as tf
        image = tf.convert_to_tensor(image, dtype=self.input_dtype)
        self.model.set_tensor(self.input_index, image)
        self.model.invoke()
        heatmaps = self.model.get_tensor(self.output_index)
        return heatmaps

    def predict(
        self,
        input: KeypointsRegressorInput
    ) -> KeypointsRegressorOutput:
        n_keypoints = []
        for image in input:
            height, width, _ = image.shape
            center, scale = bbox_xywh2cs(
                [0, 0, width, height],
                aspect_ratio=192/256,
                padding=1.25,
                pixel_std=200
            )
            image = top_down_affine(image, 0, center, scale, [192, 256], False)
            image = preprocess(image)
            heatmaps = self._raw_predict(image)
            keypoints = keypoints_from_heatmaps(
                heatmaps,
                [center],
                [scale],
                unbiased=False,
                post_process='default',
                kernel=11,
                valid_radius_factor=0.0546875,
                use_udp=False,
                target_type='GaussianHeatmap'
            )[0][0]
            n_keypoints.append(keypoints)

        n_keypoints = np.array(n_keypoints)
        return n_keypoints

    def preprocess_input(self, input: KeypointsRegressorInput):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
