import numpy as np
import pytest
from xgboost import XGBClassifier

from models.persistence import ARTIFACTS_DIR, load_xgb_model, save_xgb_model
from models.predict import predict


@pytest.fixture
def trained_model():
    rng = np.random.default_rng(0)
    X = rng.random((60, 2))
    y = (X[:, 0] * 3).astype(int)  # three classes, matching real multi:softprob usage

    model = XGBClassifier(n_estimators=5, objective="multi:softprob")
    model.fit(X, y)
    return model, X


def test_save_and_load_preserves_predictions(trained_model, tmp_path, monkeypatch):
    model, X = trained_model
    monkeypatch.setattr("models.persistence.ARTIFACTS_DIR", tmp_path)

    save_xgb_model(model, "test", ["a", "b"], ["low", "mid", "high"])
    saved = load_xgb_model("test")

    assert np.allclose(model.predict_proba(X), saved.model.predict_proba(X))
    assert saved.feature_columns == ["a", "b"]
    assert saved.classes == ["low", "mid", "high"]


def test_predict_returns_one_probability_per_class(trained_model, tmp_path, monkeypatch):
    model, _ = trained_model
    monkeypatch.setattr("models.persistence.ARTIFACTS_DIR", tmp_path)

    save_xgb_model(model, "test", ["a", "b"], ["low", "mid", "high"])
    saved = load_xgb_model("test")

    result = predict(saved, {"a": 0.9, "b": 0.1})

    assert set(result.keys()) == {"low", "mid", "high"}
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_artifacts_dir_is_repo_root_artifacts():
    assert ARTIFACTS_DIR.name == "artifacts"
