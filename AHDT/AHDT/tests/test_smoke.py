"""Small smoke tests for the AHDT reference implementation."""

import numpy as np

try:
    from AHDT.model import AHDT
except ImportError:
    from model import AHDT


def test_fit_predict() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = AHDT(max_depth=4, min_samples_split=5, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X[:8])
    assert predictions.shape == (8,)
    assert model.tree_ is not None
    assert model.feature_importances_ is not None
