import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger("Preprocessor")


class SDNPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
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
        if not self.fitted:
            return X
        X = X.copy()
        means = self.scaler.mean_
        stds = np.sqrt(self.scaler.var_)
        return np.clip(X, means - 3 * stds, means + 3 * stds)