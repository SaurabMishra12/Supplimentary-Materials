"""Detector implementations and operating point calibration."""
import numpy as np

class BaseDetector:
    def score(self, query: str, docs: list[str]) -> np.ndarray:
        raise NotImplementedError

class ThreeFeatureFilter(BaseDetector):
    def __init__(self, mu: np.ndarray, mean_len: float, clf):
        self.mu = mu
        self.mean_len = mean_len
        self.clf = clf

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        # Returns probability scalar
        return np.zeros(len(docs))

def calibrate_thresholds(scores: np.ndarray, target_fprs: list[float]) -> dict[float, float]:
    """Empirical quantile calibration on clean documents."""
    return {fpr: float(np.quantile(scores, 1.0 - fpr)) for fpr in target_fprs}
