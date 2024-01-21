"""
This module contains functions for computing subgroup variance and error metrics.
"""
from virny.configs.constants import *
from .accuracy_metrics import (
    mean_prediction,
    statistical_bias_from_predict_proba,
    statistical_bias,
    confusion_matrix_metrics
)
from .stability_metrics import (
    std,
    iqr,
    churn,
    jitter,
    per_sample_label_stability,
    label_stability
)
from .uncertainty_metrics import (
    entropy_from_predicted_probability,
    conf_interval,
    aleatoric_uncertainty,
    overall_uncertainty,
)

METRIC_TO_FUNCTION = {
    # Accuracy metrics
    MEAN_PREDICTION: mean_prediction,
    STATISTICAL_BIAS: statistical_bias,
    # Stability metrics
    STD: std,
    IQR: iqr,
    JITTER: jitter,
    LABEL_STABILITY: label_stability,
    # Uncertainty metrics
    ALEATORIC_UNCERTAINTY: aleatoric_uncertainty,
    OVERALL_UNCERTAINTY: overall_uncertainty,
}

METRICS_FOR_PREDICT_PROBA = {MEAN_PREDICTION, STATISTICAL_BIAS, STD, IQR,
                             ALEATORIC_UNCERTAINTY, OVERALL_UNCERTAINTY, EPISTEMIC_UNCERTAINTY}
METRICS_FOR_LABELS = set([metric for metric in METRIC_TO_FUNCTION.keys() if metric not in METRICS_FOR_PREDICT_PROBA])

__all__ = [
    "mean_prediction",
    "statistical_bias",
    "confusion_matrix_metrics",
    "std",
    "iqr",
    "jitter",
    "label_stability",
    "aleatoric_uncertainty",
    "overall_uncertainty",
]
