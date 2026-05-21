import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.pipeline import PipelineModelSpec


@pytest.fixture(scope="session")
def model_artifact_cache(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("cv_pipeliner_model_artifacts")
    yield artifact_dir
    shutil.rmtree(artifact_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def tensorflow_module():
    return pytest.importorskip("tensorflow")


def _constant_model(tf, input_shape, output_values):
    inputs = tf.keras.Input(shape=input_shape)
    output = tf.keras.layers.Lambda(
        lambda x: tf.tile(
            tf.constant([output_values], dtype=tf.float32),
            [tf.shape(x)[0]] + [1] * np.array(output_values).ndim,
        )
    )(inputs)
    return tf.keras.Model(inputs=inputs, outputs=output)


def _save_saved_model(tf, model, path):
    export = getattr(model, "export", None)
    if export is not None:
        export(path)
    else:
        model.save(path, save_format="tf")


def _download_file(url: str, output_path: Path):
    if output_path.exists():
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as response, output_path.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        pytest.skip(f"Could not download {url}: {exc}")


def test_real_tensorflow_classification_smoke(tensorflow_module, model_artifact_cache):
    from cv_pipeliner.inferencers.classification.tensorflow import TensorFlow_ClassificationModelSpec

    model_path = model_artifact_cache / "classification_saved_model"
    model = _constant_model(tensorflow_module, input_shape=(4, 4, 3), output_values=[0.1, 0.9])
    _save_saved_model(tensorflow_module, model, model_path)

    spec = TensorFlow_ClassificationModelSpec(
        input_size=(4, 4),
        class_names=["low", "high"],
        model_path=str(model_path),
        saved_model_type="tf.saved_model",
        preprocess_input=lambda images: np.array(images, dtype=np.float32),
    )
    inferencer = spec.load_classification_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((4, 4, 3), dtype=np.uint8))],
        top_n=2,
        disable_tqdm=True,
    )

    assert model_path.exists()
    assert result[0].label == "high"
    np.testing.assert_allclose(result[0].classification_score, 0.9, rtol=1e-6)


def test_real_tensorflow_embedder_smoke(tensorflow_module, model_artifact_cache):
    from cv_pipeliner.inferencers.embedder.tensorflow import TensorFlow_EmbedderModelSpec

    model_path = model_artifact_cache / "embedder_saved_model"
    model = _constant_model(tensorflow_module, input_shape=(4, 4, 3), output_values=[1.0, 2.0, 3.0])
    _save_saved_model(tensorflow_module, model, model_path)

    spec = TensorFlow_EmbedderModelSpec(
        input_size=(4, 4),
        model_path=str(model_path),
        saved_model_type="tf.saved_model",
        preprocess_input=lambda images: np.array(images, dtype=np.float32),
    )
    inferencer = spec.load_embedder_inferencer()

    embeddings = inferencer.predict(
        [ImageData(image=np.zeros((4, 4, 3), dtype=np.uint8)) for _ in range(2)],
        batch_size_default=1,
        disable_tqdm=True,
    )

    assert model_path.exists()
    assert len(embeddings) == 2
    np.testing.assert_allclose(embeddings[0], np.array([1.0, 2.0, 3.0]), rtol=1e-6)


def test_real_tensorflow_keypoints_regressor_smoke(tensorflow_module, model_artifact_cache):
    from cv_pipeliner.inferencers.keypoints_regressor.tensorflow import TensorFlow_KeypointsRegressorModelSpec

    model_path = model_artifact_cache / "keypoints_saved_model"
    model = _constant_model(tensorflow_module, input_shape=(10, 10, 3), output_values=[[0.25, 0.5], [0.75, 0.5]])
    _save_saved_model(tensorflow_module, model, model_path)

    spec = TensorFlow_KeypointsRegressorModelSpec(
        input_size=(10, 10),
        model_path=str(model_path),
        saved_model_type="tf.saved_model",
        preprocess_input=lambda images: np.array(images, dtype=np.float32),
    )
    inferencer = spec.load_keypoints_regressor_inferencer()

    result = inferencer.predict(
        [[BboxData(image=np.zeros((10, 10, 3), dtype=np.uint8), xmin=0, ymin=0, xmax=10, ymax=10)]],
        open_images_in_data=True,
        disable_tqdm=True,
    )

    assert model_path.exists()
    np.testing.assert_array_equal(result[0][0].keypoints, np.array([[2, 4], [6, 4]]))


def test_real_yolov8_detection_smoke(model_artifact_cache):
    ultralytics = pytest.importorskip("ultralytics")
    from cv_pipeliner.inferencers.detection.yolov8 import YOLOv8_ModelSpec

    weight_path = model_artifact_cache / "yolov8n.pt"
    if not weight_path.exists():
        try:
            downloaded_model = ultralytics.YOLO("yolov8n.pt")
        except Exception as exc:
            pytest.skip(f"Could not download/load yolov8n.pt: {exc}")
        source = Path(downloaded_model.ckpt_path)
        if not source.exists():
            pytest.skip("Ultralytics did not expose a cached yolov8n.pt artifact")
        shutil.copy2(source, weight_path)

    spec = YOLOv8_ModelSpec(model_path=weight_path, device="cpu")
    inferencer = spec.load_detection_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((64, 64, 3), dtype=np.uint8))],
        score_threshold=0.99,
        disable_tqdm=True,
    )

    assert len(result) == 1
    assert result[0].bboxes_data == []


def test_real_yolov5_detection_smoke(model_artifact_cache):
    pytest.importorskip("torch")
    from cv_pipeliner.inferencers.detection.yolov5 import YOLOv5_ModelSpec

    weight_path = model_artifact_cache / "yolov5n.pt"
    _download_file("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt", weight_path)

    spec = YOLOv5_ModelSpec(model_path=weight_path, device="cpu", skip_validation=True)
    inferencer = spec.load_detection_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((64, 64, 3), dtype=np.uint8))],
        score_threshold=0.99,
        disable_tqdm=True,
    )

    assert len(result) == 1
    assert isinstance(result[0], ImageData)


def test_real_pipeline_smoke_with_downloaded_detector_and_tiny_classifier(tensorflow_module, model_artifact_cache):
    ultralytics = pytest.importorskip("ultralytics")
    from cv_pipeliner.inferencers.classification.tensorflow import TensorFlow_ClassificationModelSpec
    from cv_pipeliner.inferencers.detection.yolov8 import YOLOv8_ModelSpec

    weight_path = model_artifact_cache / "pipeline-yolov8n.pt"
    if not weight_path.exists():
        try:
            downloaded_model = ultralytics.YOLO("yolov8n.pt")
        except Exception as exc:
            pytest.skip(f"Could not download/load yolov8n.pt: {exc}")
        source = Path(downloaded_model.ckpt_path)
        if not source.exists():
            pytest.skip("Ultralytics did not expose a cached yolov8n.pt artifact")
        shutil.copy2(source, weight_path)

    classifier_path = model_artifact_cache / "pipeline_classifier_saved_model"
    classifier_model = _constant_model(tensorflow_module, input_shape=(4, 4, 3), output_values=[1.0])
    _save_saved_model(tensorflow_module, classifier_model, classifier_path)
    classifier_spec = TensorFlow_ClassificationModelSpec(
        input_size=(4, 4),
        class_names=["tiny"],
        model_path=str(classifier_path),
        saved_model_type="tf.saved_model",
        preprocess_input=lambda images: np.array(images, dtype=np.float32),
    )

    inferencer = PipelineModelSpec(
        detection_model_spec=YOLOv8_ModelSpec(model_path=weight_path, device="cpu"),
        classification_model_spec=classifier_spec,
    ).load_pipeline_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((64, 64, 3), dtype=np.uint8))],
        detection_score_threshold=0.99,
        disable_tqdm=True,
        disable_tqdm_classification=True,
    )

    assert classifier_path.exists()
    assert len(result) == 1
    assert isinstance(result[0], ImageData)
