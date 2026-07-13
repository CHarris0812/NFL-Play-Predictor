"""Baseline outcome classifiers: logistic regression and XGBoost.

Trains on situational features only (see features.outcome).
"""

import polars as pl

from features.outcome import FEATURE_COLUMNS, LABEL_COLUMN, OUTCOMES
from models.baseline import EvalResult
from models.baseline import train_and_evaluate as _train_and_evaluate

__all__ = ["EvalResult", "train_and_evaluate"]


def train_and_evaluate(dataset: pl.DataFrame, test_season: int) -> list[EvalResult]:
    """Train on all seasons before test_season, evaluate on test_season."""
    return _train_and_evaluate(dataset, test_season, FEATURE_COLUMNS, LABEL_COLUMN, OUTCOMES)
