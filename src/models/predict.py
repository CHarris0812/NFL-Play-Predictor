"""Predict with a previously saved model - no training involved."""

import numpy as np

from models.persistence import SavedModel


def predict(saved: SavedModel, features: dict[str, float]) -> dict[str, float]:
    """Predict class probabilities for one play.

    `features` must have a value for every key in saved.feature_columns
    (extra keys are ignored, order doesn't matter).
    """
    x = np.array([[features[col] for col in saved.feature_columns]])
    proba = saved.model.predict_proba(x)[0]
    return dict(zip(saved.classes, proba, strict=True))
