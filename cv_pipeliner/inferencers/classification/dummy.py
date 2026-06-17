from pathlib import Path
from typing import List, Tuple, Type, Union

from cv_pipeliner.inferencers.classification.core import (
    ClassificationInput,
    ClassificationRuntime,
    ClassificationModelSpec,
    ClassificationOutput,
)


class Dummy_ClassificationModelSpec(ClassificationModelSpec):
    default_class_name: Union[List[str], str, Path]

    def __post_init__(self):
        self.class_names = [self.default_class_name]

    @property
    def runtime_cls(self) -> Type["DummyClassificationRuntime"]:
        from cv_pipeliner.inferencers.classification.dummy import (
            DummyClassificationRuntime,
        )

        return DummyClassificationRuntime


class DummyClassificationRuntime(ClassificationRuntime):
    def __init__(self, model_spec: Dummy_ClassificationModelSpec, **kwargs):
        assert isinstance(model_spec, Dummy_ClassificationModelSpec)
        super().__init__(model_spec)
        self.default_class_name = model_spec.default_class_name

    def predict(self, input: ClassificationInput, top_n: int = 1) -> ClassificationOutput:
        pred_labels_top_n = [[self.default_class_name for j in range(top_n)] for i in range(len(input))]
        pred_scores_top_n = [[1.0 for j in range(top_n)] for i in range(len(input))]
        return pred_labels_top_n, pred_scores_top_n

    def preprocess_input(self, input: ClassificationInput):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return (None, None)

    @property
    def class_names(self) -> List[str]:
        return [self.default_class_name]
