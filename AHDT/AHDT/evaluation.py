"""Cross-validation and reporting helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


def cross_validate_ahdt(
    X: Any,
    y: Any,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[Dict[str, Any]]]:
    """Run stratified cross-validation without coupling the model to a dataset."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[Dict[str, float]] = []
    fold_details: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_array, y_array), start=1):
        model = model_factory()
        model.fit(X_array[train_idx], y_array[train_idx])
        y_pred = model.predict(X_array[test_idx])
        scores = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_array[test_idx], y_pred),
            "Precision": precision_score(y_array[test_idx], y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_array[test_idx], y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_array[test_idx], y_pred, average="weighted", zero_division=0),
        }
        fold_scores.append(scores)
        fold_details.append(
            {
                "Fold": fold,
                "Model": model,
                "y_true": y_array[test_idx],
                "y_pred": y_pred,
                "test_indices": test_idx,
            }
        )
        if verbose:
            print(
                f"Fold {fold}/{n_splits}: "
                f"ACC={scores['Accuracy']:.4f}, "
                f"Precision={scores['Precision']:.4f}, "
                f"Recall={scores['Recall']:.4f}, "
                f"F1={scores['F1-Score']:.4f}"
            )

    results = pd.DataFrame(fold_scores)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    return results, results[metric_columns].mean(), results[metric_columns].std(), fold_details


def print_classification_report(
    fold_details: List[Dict[str, Any]], target_names: Any = None
) -> None:
    """Print the classification report for the final validation fold."""
    if not fold_details:
        raise ValueError("fold_details is empty")
    last_fold = fold_details[-1]
    print(
        classification_report(
            last_fold["y_true"],
            last_fold["y_pred"],
            target_names=target_names,
            zero_division=0,
        )
    )
