"""Dataset loading utilities.

The model itself is dataset-agnostic. This module only provides a reusable
loader for delimited tabular classification data; replace it with a project-
specific loader when a dataset needs special preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_tabular_classification_data(
    path: str,
    target_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    """Load a delimited classification dataset and ordinal-encode categories.

    Returns ``X, y, feature_names, metadata``. Encoders are included in
    ``metadata`` in case the caller needs to decode values later.
    """
    data = pd.read_csv(path, sep=delimiter, encoding=encoding)
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found in {path!r}")

    drop = [column for column in (drop_columns or []) if column in data.columns]
    data = data.drop(columns=drop)
    y_raw = data.pop(target_column)
    X_frame = data.copy()

    encoders: Dict[str, LabelEncoder] = {}
    for column in X_frame.select_dtypes(include=["object", "category", "bool"]).columns:
        encoder = LabelEncoder()
        X_frame[column] = encoder.fit_transform(X_frame[column].astype(str))
        encoders[column] = encoder

    if y_raw.dtype == object or str(y_raw.dtype).startswith("category") or y_raw.dtype == bool:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
        encoders["__target__"] = target_encoder
    else:
        y = y_raw.to_numpy()

    try:
        X = X_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature columns must be numeric or categorical") from exc
    if not np.isfinite(X).all():
        raise ValueError("The dataset contains missing or infinite feature values")

    metadata: Dict[str, Any] = {
        "path": path,
        "target_column": target_column,
        "encoders": encoders,
        "dataframe": data,
    }
    return X, np.asarray(y), list(X_frame.columns), metadata


def load_bank_marketing_data(path: str) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    """Example loader for the UCI Bank Marketing files."""
    return load_tabular_classification_data(
        path=path,
        target_column="y",
        delimiter=";",
    )
