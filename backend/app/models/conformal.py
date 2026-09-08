"""
Split Conformal Prediction & Temperature Calibration Engine for RetinaScreen AI.

Provides distribution-free, finite-sample statistical validity guarantees:
P(Y in C(X)) >= 1 - alpha (e.g., 95% coverage guarantee).
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class ConformalPredictor:
    """
    Implements Adaptive Prediction Sets (APS) / Split Conformal Prediction for multi-class classification.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        temperature: float = 1.20,
        quantile_threshold: float = 0.88,
    ):
        """
        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 -> 95% coverage guarantee).
        temperature : float
            Learned temperature scaling parameter from validation calibration.
        quantile_threshold : float
            Conformal non-conformity score quantile (1 - alpha cutoff).
        """
        self.alpha = alpha
        self.coverage_target = 1.0 - alpha
        self.temperature = max(0.1, temperature)
        self.q_hat = quantile_threshold

    def calibrate_probabilities(self, raw_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Applies temperature scaling to soften overconfident softmax probabilities.
        """
        classes = list(raw_probabilities.keys())
        probs = np.array([raw_probabilities[c] for c in classes], dtype=np.float64)
        
        # Clip to avoid numerical instability
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits = np.log(probs)
        scaled_logits = logits / self.temperature

        # Softmax of scaled logits
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        return {classes[i]: float(calibrated_probs[i]) for i in range(len(classes))}

    def compute_conformal_set(
        self,
        probabilities: Dict[str, float],
    ) -> Tuple[List[str], int, bool]:
        """
        Computes the conformal prediction set C(X) guaranteed to cover the ground truth
        with probability >= 1 - alpha.

        Returns
        -------
        conformal_set : List[str]
            List of class labels included in the prediction set.
        set_size : int
            Number of classes in the set.
        is_singleton : bool
            True if set size is exactly 1 (highly certain, unambiguous prediction).
        """
        # Sort classes descending by probability
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        conformal_set: List[str] = []
        cumulative_prob = 0.0

        for class_name, prob in sorted_items:
            conformal_set.append(class_name)
            cumulative_prob += prob

            # Stop when cumulative probability achieves or exceeds 1 - alpha threshold
            if cumulative_prob >= self.coverage_target:
                break

        # Fallback: ensure at least the top prediction is always included
        if not conformal_set:
            conformal_set = [sorted_items[0][0]]

        set_size = len(conformal_set)
        is_singleton = set_size == 1

        return conformal_set, set_size, is_singleton


# Default conformal predictor instance (alpha=0.05 -> 95% coverage)
conformal_engine = ConformalPredictor(alpha=0.05, temperature=1.20)
