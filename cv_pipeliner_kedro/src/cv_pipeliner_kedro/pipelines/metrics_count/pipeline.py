# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node
from cv_pipeliner.metrics.detection import get_df_detection_metrics
from cv_pipeliner.metrics.pipeline import get_df_pipeline_metrics

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.metrics.detection import get_df_detection_metrics, df_detection_metrics_columns
from cv_pipeliner.metrics.pipeline import get_df_pipeline_metrics, df_pipeline_metrics_columns
from cv_pipeliner.visualizers.pipeline import PipelineVisualizer
from cv_pipeliner.logging import logger
from cv_pipeliner.utils.dataframes import transpose_columns_and_write_diffs_to_df_with_tags
from cv_pipeliner.utils.images_datas import cut_images_data_by_bboxes

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_detection_model_definition_from_config,
                inputs=[
                    'detection_model_definition_yaml'
                ],
                outputs='detection_model_definition'
            ),
            node(
                func=get_classification_model_definition_from_config,
                inputs=[
                    'classification_model_definition_yaml'
                ],
                outputs='classification_model_definition'
            ),
            node(
                func=BatchGeneratorImageData,
                inputs=[
                    'images_data_train', 'params:evaluate_batch_size', 'params:use_not_caught_elements_as_last_batch'
                ],
                outputs='images_data_train_gen'
            ),
            node(
                func=BatchGeneratorImageData,
                inputs=[
                    'images_data_val', 'params:evaluate_batch_size', 'params:use_not_caught_elements_as_last_batch'
                ],
                outputs='images_data_val_gen'
            ),
            node(
                func=get_model_spec,
                inputs=[
                    'detection_model_definition'
                ],
                outputs='detection_model_spec'
            ),
            node(
                func=get_model_spec,
                inputs=[
                    'classification_model_definition'
                ],
                outputs='classification_model_spec'
            ),
            node(
                func=get_detection_score_threshold,
                inputs=[
                    'detection_model_definition'
                ],
                outputs='detection_score_threshold'
            ),
            node(
                func=PipelineModelSpec,
                inputs=[
                    'detection_model_spec', 'classification_model_spec'
                ],
                outputs='pipeline_model_spec'
            ),
            node(
                func=get_model,
                inputs=[
                    'detection_model_spec'
                ],
                outputs='detection_model'
            ),
            node(
                func=get_model,
                inputs=[
                    'classification_model_spec'
                ],
                outputs='classification_model'
            ),
            node(
                func=get_model_class_names,
                inputs=[
                    'classification_model_spec'
                ],
                outputs='model_class_names'
            ),
            node(
                func=get_pipeline_model,
                inputs=[
                    'detection_model', 'classification_model'
                ],
                outputs='pipeline_model'
            ),
            node(
                func=DetectionInferencer,
                inputs=[
                    'detection_model'
                ],
                outputs='detection_inferencer'
            ),
            node(
                func=ClassificationInferencer,
                inputs=[
                    'classification_model'
                ],
                outputs='classification_inferencer'
            ),
            node(
                func=PipelineInferencer,
                inputs=[
                    'pipeline_model'
                ],
                outputs='pipeline_inferencer'
            ),
            node(
                func=make_detection_inference,
                inputs=[
                    'detection_inferencer', 'images_data_val_gen', 'detection_score_threshold'
                ],
                outputs='pred_detection_images_data_val'
            ),
            node(
                func=make_raw_detection_inference,
                inputs=[
                    'detection_inferencer', 'images_data_val_gen'
                ],
                outputs='raw_pred_detection_images_data_val'
            ),
            node(
                func=make_pipeline_inference,
                inputs=[
                    'pipeline_inferencer', 'images_data_val_gen', 'detection_score_threshold'
                ],
                outputs='pred_pipeline_images_data_val'
            ),
            node(
                func=get_df_detection_metrics,
                inputs=[
                    'images_data_val',
                    'pred_detection_images_data_val',
                    'params:minimum_iou',
                    'raw_pred_detection_images_data_val',
                ],
                outputs='df_detection_metrics_on_val'
            ),
            node(
                func=get_df_pipeline_metrics,
                inputs=[
                    'images_data_val',
                    'pred_pipeline_images_data_val',
                    'params:minimum_iou',
                    'params:extra_bbox_label',
                    'params:pseudo_class_names',
                    'model_class_names'
                ],
                outputs='df_pipeline_metrics_on_val'
            )
        ]
    )
