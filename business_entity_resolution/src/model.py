"""
Classifier Training, Probability Calibration, and Threshold Optimization
Amazon ML Challenge 2026

Implements:
1. LightGBM / XGBoost Gradient Boosted Classifier with class weighting
2. Probability Calibration (Platt scaling / Sigmoid)
3. Optimal Macro F0.5 Threshold Sweeper (precision-weighted)
4. Group-Aware Multi-Match Decision Rule
"""

from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from .config import config
from .evaluation import evaluate_macro_f05


class EntityMatchingModel:
    def __init__(
        self,
        model_type: str = "lightgbm",
        lgb_params: Optional[dict] = None,
        calibration_method: str = "sigmoid",
        default_threshold: float = 0.58,
        relative_margin: float = 0.30,
    ):
        self.model_type = model_type.lower()
        self.lgb_params = lgb_params or config.lgb_params
        self.calibration_method = calibration_method
        self.threshold = default_threshold
        self.relative_margin = relative_margin

        self.base_model = None
        self.calibrated_model = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Fit classifier and calibrate output probabilities."""
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

        if self.model_type == "lightgbm":
            params = dict(self.lgb_params)
            params["scale_pos_weight"] = float(scale_pos)
            self.base_model = lgb.LGBMClassifier(**params)
        elif self.model_type == "xgboost":
            xgb_kwargs = {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 6,
                "scale_pos_weight": scale_pos,
                "eval_metric": "logloss",
                "random_state": config.random_seed,
            }
            # Attempt native CUDA GPU acceleration on NVIDIA RTX 5060
            try:
                self.base_model = xgb.XGBClassifier(**xgb_kwargs, tree_method="hist", device="cuda")
            except Exception:
                self.base_model = xgb.XGBClassifier(**xgb_kwargs, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if X_val is not None and y_val is not None:
            self.base_model.fit(X_train, y_train)
            # Calibrate on validation split
            try:
                from sklearn.frozen import FrozenEstimator
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(self.base_model),
                    method=self.calibration_method,
                )
            except Exception:
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=self.base_model,
                    method=self.calibration_method,
                    cv="prefit",
                )
            self.calibrated_model.fit(X_val, y_val)
        else:
            self.base_model.fit(X_train, y_train)
            # Calibrate using internal 3-fold CV on train set
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.base_model,
                method=self.calibration_method,
                cv=3,
            )
            self.calibrated_model.fit(X_train, y_train)

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated match probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")
        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict_matches(
        self,
        candidate_pairs: Dict[str, List[str]],
        pair_probabilities: Dict[Tuple[str, str], float],
        threshold: Optional[float] = None,
        relative_margin: Optional[float] = None,
    ) -> Dict[str, Set[str]]:
        """Apply group-aware decision rule to output matching entities.

        Rule:
        For each S1 entity:
        1. Find max probability among its candidates.
        2. If max_p < threshold: output empty set (singleton).
        3. Else, include every candidate where:
           p >= threshold AND (max_p - p) <= relative_margin.
        """
        th = threshold if threshold is not None else self.threshold
        margin = relative_margin if relative_margin is not None else self.relative_margin

        predictions: Dict[str, Set[str]] = {}

        for s1_id, cids in candidate_pairs.items():
            if not cids:
                predictions[s1_id] = set()
                continue

            cand_probs = []
            for cid in cids:
                prob = pair_probabilities.get((s1_id, cid), 0.0)
                cand_probs.append((cid, prob))

            max_p = max(p for _, p in cand_probs)

            # If top candidate does not cross threshold, declare singleton
            if max_p < th:
                predictions[s1_id] = set()
                continue

            matched = set()
            for cid, prob in cand_probs:
                if prob >= th and (max_p - prob) <= margin:
                    matched.add(cid)

            predictions[s1_id] = matched

        return predictions

    def save(self, filepath: str):
        """Save fitted model and parameters to disk."""
        import joblib
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EntityMatchingModel":
        """Load fitted model from disk."""
        import joblib
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def optimize_threshold(
    model: EntityMatchingModel,
    candidate_pairs: Dict[str, List[str]],
    pair_probabilities: Dict[Tuple[str, str], float],
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float, float] = (0.40, 0.92, 0.02),
    relative_margin: float = 0.30,
) -> Tuple[float, float, Dict[float, dict]]:
    """Sweep decision thresholds on validation data to maximize Macro F0.5.

    Returns:
        (best_threshold, best_f05, log_dict)
    """
    start_th, end_th, step = threshold_range
    thresholds = np.arange(start_th, end_th + 1e-5, step)

    best_f05 = -1.0
    best_th = 0.58
    sweep_log = {}

    for th in thresholds:
        th = round(float(th), 4)
        preds = model.predict_matches(
            candidate_pairs,
            pair_probabilities,
            threshold=th,
            relative_margin=relative_margin,
        )
        metrics = evaluate_macro_f05(
            ground_truth=ground_truth,
            predictions=preds,
            required_s1_ids=list(candidate_pairs.keys()),
        )
        f05 = metrics["macro_f05"]
        sweep_log[th] = metrics

        if f05 > best_f05:
            best_f05 = f05
            best_th = th

    model.threshold = best_th
    return best_th, best_f05, sweep_log
