import numpy as np

from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.embedder import EmbedderInferencer
from cv_pipeliner.inferencers.embedder.core import EmbedderRuntime, EmbedderModelSpec


class FakeEmbedderModelSpec(EmbedderModelSpec):
    @property
    def runtime_cls(self):
        return FakeEmbedderRuntime


class FakeEmbedderRuntime(EmbedderRuntime):
    def predict(self, input):
        return [np.array([idx, image.shape[0], image.shape[1]], dtype=np.float32) for idx, image in enumerate(input)]

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (4, 4)


def test_embedder_inferencer_predicts_list_in_batches():
    inferencer = EmbedderInferencer(FakeEmbedderRuntime(FakeEmbedderModelSpec()))
    images_data = [
        ImageData(image=np.zeros((4, 4, 3), dtype=np.uint8)),
        ImageData(image=np.zeros((5, 6, 3), dtype=np.uint8)),
        ImageData(image=np.zeros((7, 8, 3), dtype=np.uint8)),
    ]

    embeddings = inferencer.predict(images_data, batch_size_default=2, disable_tqdm=True)

    assert len(embeddings) == 3
    np.testing.assert_array_equal(embeddings[0], np.array([0, 4, 4], dtype=np.float32))
    np.testing.assert_array_equal(embeddings[1], np.array([1, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(embeddings[2], np.array([0, 7, 8], dtype=np.float32))


def test_embedder_inferencer_accepts_image_data_generator():
    inferencer = EmbedderInferencer(FakeEmbedderRuntime(FakeEmbedderModelSpec()))
    images_data_gen = BatchGeneratorImageData(
        [ImageData(image=np.zeros((4, 4, 3), dtype=np.uint8))],
        batch_size=1,
        max_workers=1,
    )

    embeddings = inferencer.predict(images_data_gen, disable_tqdm=True)

    assert len(embeddings) == 1
    assert embeddings[0].shape == (3,)
