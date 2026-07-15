"""Save/load trained models to/from disk, so we can predict without retraining.

Only XGBoost gets persisted - it's outperformed logistic regression on
both baselines so far, and logistic regression stays in training/eval
purely as a comparison point. XGBoost's own native format doesn't
carry the feature order or class labels, so those are saved alongside
it in a small metadata file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from xgboost import XGBClassifier

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class SavedModel:
    model: XGBClassifier
    feature_columns: list[str]
    classes: list[str]


def save_xgb_model(
    model: XGBClassifier, name: str, feature_columns: list[str], classes: list[str]
) -> None:
    """Save a fitted XGBClassifier plus the metadata needed to use it later.

    classes must be the sorted class order used to encode the model's
    integer labels (see models.baseline.train_and_evaluate) - that's
    what predict_proba's output columns correspond to.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(ARTIFACTS_DIR / f"{name}.xgb.json")
    meta = {"feature_columns": feature_columns, "classes": classes}
    (ARTIFACTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


def load_xgb_model(name: str) -> SavedModel:
    """Load a model saved by save_xgb_model - no training involved."""
    meta = json.loads((ARTIFACTS_DIR / f"{name}.meta.json").read_text())
    model = XGBClassifier()
    model.load_model(ARTIFACTS_DIR / f"{name}.xgb.json")
    return SavedModel(model=model, feature_columns=meta["feature_columns"], classes=meta["classes"])
