"""Adaptive Hybrid-feature Decision Tree (AHDT).

The implementation keeps the paper's main pipeline:

1. discretize high-cardinality features with adaptive K-means;
2. pre-filter features with the dispersion ratio (DR);
3. combine DR and ReliefF-style feature weight into WDR;
4. recursively build a binary decision tree.

The model accepts a numeric matrix. Categorical columns should be encoded by
``data.py`` or by the caller before ``fit``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


class AHDT:
    """Adaptive Hybrid-feature Decision Tree.

    Parameters
    ----------
    alpha_strategy:
        ``"auto"`` calculates alpha at every node. A numeric value in [0, 1]
        can be supplied to use a fixed value.
    k_range:
        Inclusive lower and upper bounds for adaptive K-means clusters.
    quantile_threshold:
        DR quantile used by the feature pre-filter.
    max_depth:
        Maximum tree depth.
    min_samples_split:
        A node with fewer samples becomes a leaf.
    random_state:
        Seed used by K-means and ReliefF sampling.
    split_strategy:
        ``"information_gain"`` chooses the best candidate split after WDR
        chooses a feature. ``"first_valid"`` reproduces the simple equality
        split used by the original script.
    """

    def __init__(
        self,
        alpha_strategy: Any = "auto",
        k_range: Tuple[int, int] = (2, 6),
        quantile_threshold: float = 0.5,
        max_depth: int = 8,
        min_samples_split: int = 10,
        random_state: int = 42,
        split_strategy: str = "information_gain",
    ) -> None:
        if not (0.0 <= quantile_threshold <= 1.0):
            raise ValueError("quantile_threshold must be between 0 and 1")
        if len(k_range) != 2 or k_range[0] < 2 or k_range[1] < k_range[0]:
            raise ValueError("k_range must be an inclusive range such as (2, 6)")
        if split_strategy not in {"information_gain", "first_valid"}:
            raise ValueError("split_strategy must be 'information_gain' or 'first_valid'")

        self.alpha_strategy = alpha_strategy
        self.k_range = k_range
        self.quantile_threshold = quantile_threshold
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.split_strategy = split_strategy

        self.tree_: Optional[Dict[str, Any]] = None
        self.feature_importance_: Optional[Dict[str, Any]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.discretization_info_: Dict[int, Dict[str, Any]] = {}
        self.selected_features_: List[int] = []
        self.dr_scores_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.n_features_in_: Optional[int] = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Numeric preprocessing
    # ------------------------------------------------------------------
    def _check_X(self, X: Any) -> np.ndarray:
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2-dimensional numeric array")
        try:
            X_array = X_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("AHDT requires numeric features; encode categorical columns first") from exc
        if not np.isfinite(X_array).all():
            raise ValueError("X contains NaN or infinite values")
        return X_array

    def _kmeans_discretization_fit(self, feature_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        unique_values = np.unique(feature_data)
        if len(unique_values) <= 3:
            return feature_data.copy(), {"discretized": False, "optimal_k": len(unique_values)}

        n_samples = len(feature_data)
        max_possible_k = max(2, n_samples // 2)
        k_min = self.k_range[0]
        k_max = min(self.k_range[1], max_possible_k, len(unique_values))
        if k_max < k_min:
            return feature_data.copy(), {"discretized": False, "optimal_k": len(unique_values)}

        best_k = k_min
        best_score = -np.inf
        for k in range(k_min, k_max + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(feature_data.reshape(-1, 1))
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(feature_data.reshape(-1, 1), labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except (ValueError, TypeError):
                continue

        final_kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        labels = final_kmeans.fit_predict(feature_data.reshape(-1, 1))
        centers = final_kmeans.cluster_centers_.reshape(-1)
        return labels.astype(float), {
            "discretized": True,
            "optimal_k": int(best_k),
            "centers": centers,
            "silhouette_score": None if not np.isfinite(best_score) else float(best_score),
        }

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        X_processed = X.copy()
        self.discretization_info_ = {}
        for feature_idx in range(X.shape[1]):
            if len(np.unique(X[:, feature_idx])) > 5:
                discretized, info = self._kmeans_discretization_fit(X[:, feature_idx])
                X_processed[:, feature_idx] = discretized
                self.discretization_info_[feature_idx] = info
        return X_processed

    def _preprocess_new_data(self, X: Any) -> np.ndarray:
        X_array = self._check_X(X)
        if self.n_features_in_ is not None and X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, but AHDT was fitted with {self.n_features_in_}"
            )

        X_processed = X_array.copy()
        for feature_idx, info in self.discretization_info_.items():
            if not info.get("discretized", False):
                continue
            centers = np.asarray(info["centers"])
            distances = np.abs(X_array[:, feature_idx, None] - centers[None, :])
            X_processed[:, feature_idx] = np.argmin(distances, axis=1).astype(float)
        return X_processed

    # ------------------------------------------------------------------
    # Paper-inspired scoring functions
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_entropy(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))

    def _calculate_adaptive_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.alpha_strategy != "auto":
            alpha = float(self.alpha_strategy)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("alpha must be between 0 and 1")
            return alpha

        num_classes = len(np.unique(y))
        num_features = X.shape[1]
        num_classes_factor = min(1.0, num_classes / 5.0)
        num_features_factor = 0.4 + 0.1 * min(1.0, num_features / 20.0)
        class_counts = np.bincount(y)
        max_class_ratio = np.max(class_counts) / len(y)
        balance_factor = 0.8 if max_class_ratio > 0.7 else 0.5
        alpha = (
            num_classes_factor * 0.3
            + num_features_factor * 0.4
            + balance_factor * 0.3
        )
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _calculate_dispersion_ratio(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """Calculate DR from class-wise modal-value concentration."""
        feature_data = X[:, feature_idx]
        classes = np.unique(y)
        max_freq_per_class: List[int] = []
        class_sizes: List[int] = []
        class_modes: List[float] = []

        for cls in classes:
            values = feature_data[y == cls]
            class_sizes.append(len(values))
            if len(values) == 0:
                max_freq_per_class.append(0)
                class_modes.append(0.0)
                continue
            unique_values, counts = np.unique(values, return_counts=True)
            mode_idx = int(np.argmax(counts))
            max_freq_per_class.append(int(counts[mode_idx]))
            class_modes.append(float(unique_values[mode_idx]))

        total_instances = len(y)
        if total_instances == 0:
            return 0.0
        overall_importance = sum(max_freq_per_class) / total_instances
        relative_importance = [
            freq / size if size else 0.0
            for freq, size in zip(max_freq_per_class, class_sizes)
        ]
        numerator = sum(
            size * (rel - overall_importance) ** 2
            for size, rel in zip(class_sizes, relative_importance)
        )

        denominator = 0.0
        for cls, mode in zip(classes, class_modes):
            values = feature_data[y == cls]
            indicator = (values == mode).astype(float)
            denominator += float(np.sum((indicator - overall_importance) ** 2))
        if denominator <= 0.0:
            return 0.0
        return float(np.sqrt(max(0.0, numerator / denominator)))

    def _calculate_feature_weight_relieff(
        self, X: np.ndarray, y: np.ndarray, feature_idx: int, k: int = 5
    ) -> float:
        """Calculate a non-negative ReliefF-style feature weight."""
        n_samples = len(X)
        if n_samples < 2:
            return 0.0
        feature_data = X[:, feature_idx]
        feature_range = float(np.max(feature_data) - np.min(feature_data))
        if feature_range == 0.0:
            return 0.0

        k = min(max(1, k), n_samples - 1)
        sample_size = min(100, n_samples)
        sampled_indices = self._rng.choice(n_samples, sample_size, replace=False)
        weight = 0.0

        for sample_idx in sampled_indices:
            distances = np.sqrt(np.sum((X - X[sample_idx]) ** 2, axis=1))
            distances[sample_idx] = np.inf
            nearest_indices = np.argsort(distances)[:k]
            same = nearest_indices[y[nearest_indices] == y[sample_idx]]
            different = nearest_indices[y[nearest_indices] != y[sample_idx]]

            if len(same):
                diff_same = np.mean(np.abs(feature_data[same] - feature_data[sample_idx]) / feature_range)
            else:
                diff_same = 0.0
            if len(different):
                diff_different = np.mean(
                    np.abs(feature_data[different] - feature_data[sample_idx]) / feature_range
                )
            else:
                diff_different = 0.0
            weight += max(0.0, float(diff_different - diff_same))

        return float(np.clip(weight / sample_size, 0.0, 1.0))

    def _calculate_weighted_dispersion_ratio(
        self, X: np.ndarray, y: np.ndarray, feature_idx: int, alpha: float
    ) -> Tuple[float, float, float]:
        dr = self._calculate_dispersion_ratio(X, y, feature_idx)
        feature_weight = self._calculate_feature_weight_relieff(X, y, feature_idx)
        wdr = alpha * dr + (1.0 - alpha) * feature_weight
        return float(wdr), float(dr), float(feature_weight)

    def _dynamic_feature_prefilter(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[int], np.ndarray]:
        dr_scores = np.asarray(
            [self._calculate_dispersion_ratio(X, y, i) for i in range(X.shape[1])],
            dtype=float,
        )
        threshold = float(np.quantile(dr_scores, self.quantile_threshold))
        selected = [i for i, score in enumerate(dr_scores) if score >= threshold]
        if not selected:
            selected = [int(np.argmax(dr_scores))]
        return selected, dr_scores

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------
    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        return int(Counter(y.tolist()).most_common(1)[0][0])

    def _information_gain(
        self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> float:
        if not left_mask.any() or not right_mask.any():
            return -np.inf
        parent_entropy = self._calculate_entropy(y)
        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)
        child_entropy = (
            left_weight * self._calculate_entropy(y[left_mask])
            + right_weight * self._calculate_entropy(y[right_mask])
        )
        return float(parent_entropy - child_entropy)

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, selected_features: Sequence[int], alpha: float
    ) -> Tuple[Optional[int], Optional[float], float, Dict[int, Dict[str, float]]]:
        best_feature: Optional[int] = None
        best_split_value: Optional[float] = None
        best_wdr = -np.inf
        best_gain = -np.inf
        wdr_scores: Dict[int, Dict[str, float]] = {}

        for feature_idx in selected_features:
            wdr, dr, feature_weight = self._calculate_weighted_dispersion_ratio(
                X, y, feature_idx, alpha
            )
            wdr_scores[int(feature_idx)] = {"wdr": wdr, "dr": dr, "fw": feature_weight}
            feature_data = X[:, feature_idx]
            candidate_values = np.unique(feature_data)

            local_split: Optional[float] = None
            local_gain = -np.inf
            for split_value in candidate_values:
                left_mask = feature_data == split_value
                right_mask = ~left_mask
                gain = self._information_gain(y, left_mask, right_mask)
                if gain == -np.inf:
                    continue
                if self.split_strategy == "first_valid":
                    local_split = float(split_value)
                    local_gain = gain
                    break
                if gain > local_gain:
                    local_split = float(split_value)
                    local_gain = gain

            if local_split is None:
                continue
            if wdr > best_wdr or (np.isclose(wdr, best_wdr) and local_gain > best_gain):
                best_feature = int(feature_idx)
                best_split_value = local_split
                best_wdr = wdr
                best_gain = local_gain

        return best_feature, best_split_value, float(best_wdr), wdr_scores

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, selected_features: Sequence[int], depth: int = 0
    ) -> Dict[str, Any]:
        node: Dict[str, Any] = {}
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or len(selected_features) == 0
        ):
            node.update(type="leaf", class_=self._majority_class(y), n_samples=len(y))
            return node

        alpha = self._calculate_adaptive_alpha(X, y)
        best_feature, best_split_value, best_wdr, wdr_scores = self._find_best_split(
            X, y, selected_features, alpha
        )
        if best_feature is None or best_split_value is None:
            node.update(type="leaf", class_=self._majority_class(y), n_samples=len(y))
            return node

        left_mask = X[:, best_feature] == best_split_value
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            node.update(type="leaf", class_=self._majority_class(y), n_samples=len(y))
            return node

        node.update(
            type="internal",
            feature=int(best_feature),
            split_value=float(best_split_value),
            wdr=float(best_wdr),
            alpha=float(alpha),
            n_samples=len(y),
            wdr_scores=wdr_scores,
        )
        node["left"] = self._build_tree(X[left_mask], y[left_mask], selected_features, depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], selected_features, depth + 1)
        return node

    # ------------------------------------------------------------------
    # Public estimator API
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "AHDT":
        X_array = self._check_X(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must be a one-dimensional array with the same number of rows as X")
        if len(y_array) == 0:
            raise ValueError("Cannot fit AHDT on an empty dataset")

        self.n_features_in_ = X_array.shape[1]
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_array)
        self.classes_ = self.label_encoder_.classes_
        self._rng = np.random.default_rng(self.random_state)

        X_processed = self._preprocess_fit(X_array)
        selected_features, dr_scores = self._dynamic_feature_prefilter(X_processed, y_encoded)
        self.selected_features_ = selected_features
        self.dr_scores_ = dr_scores
        self.tree_ = self._build_tree(X_processed, y_encoded, selected_features)

        importance = np.zeros(self.n_features_in_, dtype=float)
        importance[:] = dr_scores
        total = importance.sum()
        if total > 0:
            importance /= total
        self.feature_importances_ = importance
        self.feature_importance_ = {
            "selected_features": selected_features,
            "dr_scores": dr_scores,
            "discretization_info": self.discretization_info_,
        }
        return self

    def _predict_encoded_sample(self, x: np.ndarray, node: Dict[str, Any]) -> int:
        if node["type"] == "leaf":
            return int(node["class_"])
        if x[node["feature"]] == node["split_value"]:
            return self._predict_encoded_sample(x, node["left"])
        return self._predict_encoded_sample(x, node["right"])

    def predict(self, X: Any) -> np.ndarray:
        if self.tree_ is None or self.label_encoder_ is None:
            raise ValueError("The model has not been fitted")
        X_processed = self._preprocess_new_data(X)
        encoded = np.asarray(
            [self._predict_encoded_sample(row, self.tree_) for row in X_processed],
            dtype=int,
        )
        return self.label_encoder_.inverse_transform(encoded)

    def get_depth(self) -> int:
        """Return the fitted tree depth."""
        if self.tree_ is None:
            raise ValueError("The model has not been fitted")

        def depth(node: Dict[str, Any]) -> int:
            if node["type"] == "leaf":
                return 0
            return 1 + max(depth(node["left"]), depth(node["right"]))

        return depth(self.tree_)

    def export_tree(self) -> Dict[str, Any]:
        """Return the fitted tree dictionary for inspection or serialization."""
        if self.tree_ is None:
            raise ValueError("The model has not been fitted")
        return self.tree_
