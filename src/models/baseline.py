"""Shared training/eval logic for the situational baselines.

Both the play-type and outcome models are multinomial classifiers
(logistic regression + XGBoost) trained on situational features with a
season-based train/test split - plays from the same game must never
straddle train and test. This is the one place that logic lives.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class EvalResult:
    name: str
    accuracy: float
    log_loss: float
    report: str
    confusion: np.ndarray
    model: object


def _to_arrays(
    df: pl.DataFrame, feature_columns: list[str], label_column: str
) -> tuple[np.ndarray, np.ndarray]:
    return df.select(feature_columns).to_numpy(), df[label_column].to_numpy()


def _evaluate(
    name: str, y_test, y_pred, y_proba, proba_labels, display_labels: list[str], model: object
) -> EvalResult:
    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        log_loss=log_loss(y_test, y_proba, labels=proba_labels),
        report=classification_report(y_test, y_pred, labels=display_labels, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred, labels=display_labels),
        model=model,
    )


def train_and_evaluate(
    dataset: pl.DataFrame,
    test_season: int,
    feature_columns: list[str],
    label_column: str,
    classes: list[str],
) -> list[EvalResult]:
    """Train on all seasons before test_season, evaluate on test_season."""
    train_df = dataset.filter(pl.col("season") < test_season)
    test_df = dataset.filter(pl.col("season") == test_season)

    X_train, y_train = _to_arrays(train_df, feature_columns, label_column)
    X_test, y_test = _to_arrays(test_df, feature_columns, label_column)

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
            classes,
            logreg,
        )
    )

    # xgboost's sklearn API requires integer class labels. sklearn's log_loss
    # separately requires the labels it's given to be in lexicographic order
    # and to match the probability columns exactly - so encode with the
    # sorted label order, not the display order, to satisfy both.
    sorted_classes = sorted(classes)
    label_to_idx = {label: i for i, label in enumerate(sorted_classes)}
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
    y_pred = np.array([sorted_classes[i] for i in y_pred_idx])
    results.append(
        _evaluate(
            "XGBoost",
            y_test,
            y_pred,
            xgb.predict_proba(X_test),
            sorted_classes,
            classes,
            xgb,
        )
    )

    return results
