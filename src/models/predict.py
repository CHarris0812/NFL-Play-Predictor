"""Predict with a previously saved model - no training involved."""

import numpy as np

from models.persistence import SavedModel


def predict(saved: SavedModel, features: dict[str, float]) -> dict[str, float]:
    """Predict class probabilities for one play.

    `features` must have a value for every key in saved.feature_columns
    (extra keys are ignored, order doesn't matter).
    """
    return predict_batch(saved, [features])[0]


def predict_batch(
    saved: SavedModel, features_list: list[dict[str, float]]
) -> list[dict[str, float]]:
    """Predict class probabilities for many plays in one vectorized call.

    One model inference over the whole batch instead of many small ones -
    XGBoost's predict_proba is built for this and it's far faster than
    calling predict() in a loop, especially combined with loading the
    model once instead of once per play (see models.persistence).
    """
    x = np.array([[f[col] for col in saved.feature_columns] for f in features_list])
    proba = saved.model.predict_proba(x)
    return [
        {label: float(p) for label, p in zip(saved.classes, row, strict=True)} for row in proba
    ]
