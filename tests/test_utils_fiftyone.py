import os

import pytest

from cv_pipeliner.utils import fiftyone as fiftyone_utils
from cv_pipeliner.utils.fiftyone import FiftyOneSession, FifyOneSession


def _mock_fiftyone_import(monkeypatch):
    monkeypatch.setattr(FiftyOneSession, "_active_sessions", 0)
    monkeypatch.setattr(fiftyone_utils.importlib, "import_module", lambda name: object())


def test_fify_one_session_alias_is_preserved():
    assert FifyOneSession is FiftyOneSession


def test_fiftyone_session_restores_previous_environment(monkeypatch):
    _mock_fiftyone_import(monkeypatch)
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", "previous-dir")
    monkeypatch.delenv("FIFTYONE_DATABASE_URI", raising=False)

    with FiftyOneSession(database_dir="new-dir", database_uri="mongodb://localhost:27017"):
        assert os.environ["FIFTYONE_DATABASE_DIR"] == "new-dir"
        assert os.environ["FIFTYONE_DATABASE_URI"] == "mongodb://localhost:27017"

    assert os.environ["FIFTYONE_DATABASE_DIR"] == "previous-dir"
    assert "FIFTYONE_DATABASE_URI" not in os.environ


def test_fiftyone_session_close_is_idempotent(monkeypatch):
    _mock_fiftyone_import(monkeypatch)

    session = FiftyOneSession(database_name="test")
    session.close()
    session.close()

    assert FiftyOneSession._active_sessions == 0
    assert "FIFTYONE_DATABASE_NAME" not in os.environ


def test_fiftyone_session_rejects_two_active_sessions(monkeypatch):
    _mock_fiftyone_import(monkeypatch)
    session = FiftyOneSession()

    try:
        with pytest.raises(RuntimeError):
            FiftyOneSession()
    finally:
        session.close()
