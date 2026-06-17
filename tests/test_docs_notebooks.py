from pathlib import Path

import pytest


def test_getting_started_notebook_executes(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("tensorflow")
    pytest.importorskip("ultralytics")
    pytest.importorskip("torch")

    notebook_path = Path(__file__).parents[1] / "docs" / "getting_started.ipynb"
    output_path = tmp_path / "getting_started.executed.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent.resolve())}},
        allow_errors=False,
    )

    client.execute()
    nbformat.write(notebook, output_path)

    assert output_path.exists()
