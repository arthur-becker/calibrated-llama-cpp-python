from typing import Any
import joblib
from sklearn.isotonic import IsotonicRegression
import numpy as np

class Calibrator:
    def __init__(self, path: str):
        self._path = path
        self._calibrator = joblib.load(path)

    def __call__(self, X) -> Any:
        if self.is_isotonic_regressor():
            calibrated = self._calibrator.transform(X)
            return self._normalize(calibrated)
        else:
            raise NotImplementedError("Only IsotonicRegression is supported")
        
    def _normalize(self, X: np.ndarray):
        return X / np.sum(X)

    def is_isotonic_regressor(self):
        return type(self._calibrator) == IsotonicRegression