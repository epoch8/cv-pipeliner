from pathlib import Path


def test_no_inference_models_package_or_inference_model_concept():
    package_root = Path(__file__).parents[1] / "cv_pipeliner"

    assert not (package_root / "inference_models").exists()
    assert not (package_root / "core" / "inference_model.py").exists()
    assert not (package_root / "core" / "inferencer.py").exists()


def test_runtime_implementation_does_not_import_old_inference_models():
    package_root = Path(__file__).parents[1] / "cv_pipeliner"
    forbidden = ("cv_pipeliner.inference_models", "InferenceModel", "inference_model_cls")

    offenders = []
    for path in package_root.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append((path.relative_to(package_root), token))

    assert offenders == []
