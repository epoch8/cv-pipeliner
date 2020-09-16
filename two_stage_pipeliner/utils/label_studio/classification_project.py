import json
import math
import numpy as np

from typing import Dict, Union, Tuple, List
from pathlib import Path

from two_stage_pipeliner.core.data import ImageData, BboxData
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec


def prepare_annotations_by_using_top_n_predictions(
    images_data: List[ImageData],
    classification_model_spec: ClassificationModelSpec
):
    