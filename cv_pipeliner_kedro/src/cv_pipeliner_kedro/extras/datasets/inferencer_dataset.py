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

"""``TensorflowModelDataset`` is a data set implementation which can save and load
TensorFlow models.
"""
import importlib
import copy
import tempfile
import json
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict

import fsspec
import tensorflow as tf

from kedro.io.core import AbstractDataSet

from cv_pipeliner.core.inference_model import ModelSpec
from cv_pipeliner.core.inferencer import (
    Inferencer
)

from .dataclass_dataset import DataClassDataSet


class Inferencer_Dataset(AbstractDataSet):
    def __init__(
        self,
        model_spec_data_class: ModelSpec,
        model_spec_filepath: str,
        infrencer: str,
    ):
        self.model_spec_dataset = DataClassDataSet(
            data_class=model_spec_data_class,
            filepath=model_spec_filepath
        )
        split = infrencer.split('.')
        module, inferencer = '.'.join(split[:-1]), split[-1]
        module = importlib.import_module(module)
        self.inferencer = eval(f"module.{inferencer}")

    def _load(self) -> Inferencer:
        model_spec = self.model_spec_dataset.load()
        inference_model = model_spec.load()
        return self.inferencer(inference_model)

    def _save(self, inferencer: Inferencer) -> None:
        self.model_spec_dataset._save(inferencer.model.model_spec)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            model_spec_dataset=self.model_spec_dataset._describe(),
            inferencer=self.inferencer
        )
