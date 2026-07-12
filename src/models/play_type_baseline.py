"""Baseline play-type classifiers: logistic regression and XGBoost.

Trains on situational features only (see features.situational) with a
season-based train/test split - plays from the same game must never
straddle train and test.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features.situational import FEATURE_COLUMNS, LABEL_COLUMN, PLAY_TYPES


@dataclass
class EvalResult:
    name: str
    accuracy: float
    log_loss: float
    report: str
    confusion: np.ndarray


def _to_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df.select(FEATURE_COLUMNS).to_numpy(), df[LABEL_COLUMN].to_numpy()


def _evaluate(name: str, y_test, y_pred, y_proba, proba_labels) -> EvalResult:
    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        log_loss=log_loss(y_test, y_proba, labels=proba_labels),
        report=classification_report(y_test, y_pred, labels=PLAY_TYPES, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred, labels=PLAY_TYPES),
    )


def train_and_evaluate(dataset: pl.DataFrame, test_season: int) -> list[EvalResult]:
    """Train on all seasons before test_season, evaluate on test_season."""
    train_df = dataset.filter(pl.col("season") < test_season)
    test_df = dataset.filter(pl.col("season") == test_season)

    X_train, y_train = _to_arrays(train_df)
    X_test, y_test = _to_arrays(test_df)

    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    results.append(
        _evaluate(
            "Logistic Regression",
            y_test,
            logreg.predict(X_test_scaled),
            logreg.predict_proba(X_test_scaled),
            logreg.classes_,
        )
    )

    # xgboost's sklearn API requires integer class labels. sklearn's log_loss
    # separately requires the labels it's given to be in lexicographic order
    # and to match the probability columns exactly - so encode with the
    # sorted label order, not PLAY_TYPES' display order, to satisfy both.
    sorted_play_types = sorted(PLAY_TYPES)
    label_to_idx = {label: i for i, label in enumerate(sorted_play_types)}
    y_train_idx = np.array([label_to_idx[label] for label in y_train])

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train_idx)
    y_pred_idx = xgb.predict(X_test)
    y_pred = np.array([sorted_play_types[i] for i in y_pred_idx])
    results.append(
        _evaluate(
            "XGBoost",
            y_test,
            y_pred,
            xgb.predict_proba(X_test),
            sorted_play_types,
        )
    )

    return results
