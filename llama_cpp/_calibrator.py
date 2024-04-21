from typing import Any
import joblib
from sklearn.isotonic import IsotonicRegression, LogisticRegression
import numpy as np

class Calibrator:
    def __init__(self, path: str):
        self._path = path
        self._calibrator = joblib.load(path)

        print(f"Calibrator loaded from {path}")
        print(f"Calibrator type: {type(self._calibrator)}")

    def __call__(self, X_proba) -> Any:
        calibrated_proba = None
        if self.is_isotonic_regressor():
            calibrated_proba = self._calibrator.transform(X_proba)
        elif self.is_logistic_regressor():
            X_logits = self.inverse_sigmoid(X_proba)
            calibrated_proba = self._calibrator.predict_proba(X_logits)[:, 1]
        else:
            raise NotImplementedError("Only IsotonicRegression is supported")
        
        return self._normalize(calibrated_proba)
        
    def _normalize(self, X: np.ndarray):
        return X / np.sum(X)
    
    def inverse_sigmoid(proba: np.ndarray):
        epsilon = 1e-10
        temp = proba / (1 - proba + epsilon)
        return np.log(temp)

    def is_isotonic_regressor(self):
        return type(self._calibrator) == IsotonicRegression
    
    def is_logistic_regressor(self):
        return type(self._calibrator) == LogisticRegression