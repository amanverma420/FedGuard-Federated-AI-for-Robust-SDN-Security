"""
data/preprocessor.py
Feature normalization, outlier capping, and PCA-based dimensionality insight.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from utils.logger import get_logger
import config

logger = get_logger("Preprocessor")


class SDNPreprocessor:
    """
    Prepares network traffic features for model training.

    Steps:
      1. Clip extreme outliers (> 3 std) to reduce noise
      2. StandardScaler normalization (zero mean, unit variance)
      3. Optional: log-transform heavily skewed features (bytes, counts)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        # Indices of features that benefit from log transform
        # (bytes, packet counts - tend to be very skewed)
        self.log_feature_indices = [1, 2, 4, 5, 19, 20, 28, 29, 30]

    def fit(self, X: np.ndarray) -> "SDNPreprocessor":
        X_transformed = self._log_transform(X)
        self.scaler.fit(X_transformed)
        self.fitted = True
        logger.info(f"Preprocessor fitted on {X.shape[0]} samples, {X.shape[1]} features.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted, "Call fit() before transform()"
        X = self._clip_outliers(X)
        X = self._log_transform(X)
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx in self.log_feature_indices:
            if idx < X.shape[1]:
                X[:, idx] = np.log1p(np.abs(X[:, idx]))
        return X

    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip values beyond 3 std from mean (fitted mean/std from scaler)."""
        if not self.fitted:
            return X
        X = X.copy()
        means = self.scaler.mean_
        stds  = np.sqrt(self.scaler.var_)
        lower = means - 3 * stds
        upper = means + 3 * stds
        return np.clip(X, lower, upper)


def compute_feature_importance(X: np.ndarray, y: np.ndarray, top_k: int = 10):
    """
    Quick feature importance via variance ratio across classes.
    Returns top-k feature indices ranked by discriminability.
    """
    scores = []
    for feat_idx in range(X.shape[1]):
        class_means = []
        overall_mean = X[:, feat_idx].mean()
        for cls in np.unique(y):
            cls_data = X[y == cls, feat_idx]
            class_means.append(cls_data.mean())
        # Between-class variance / overall variance
        between_var = np.var(class_means)
        overall_var = X[:, feat_idx].var() + 1e-8
        scores.append(between_var / overall_var)

    ranked = np.argsort(scores)[::-1]
    logger.info(f"Top-{top_k} discriminative feature indices: {ranked[:top_k].tolist()}")
    return ranked[:top_k]
