import json
import math
import numpy as np

from typing import Dict, Union, Tuple, List
from pathlib import Path

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec


def prepare_annotations_by_using_top_n_predictions(
    images_data: List[ImageData],
    classification_model_spec: ClassificationModelSpec
):
    pass
