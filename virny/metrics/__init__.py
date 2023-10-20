"""
This module contains functions for computing subgroup variance and error metrics.
"""
from .accuracy_metrics import (
    mean_prediction,
    statistical_bias_from_predict_proba,
    statistical_bias
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


# Accuracy metrics
MEAN_PREDICTION = 'Mean_Prediction'
STATISTICAL_BIAS = 'Statistical_Bias'

# Stability metrics
STD = 'Std'
IQR = 'IQR'
JITTER = 'Jitter'
LABEL_STABILITY = 'Label_Stability'

# Uncertainty metrics
ALEATORIC_UNCERTAINTY = 'Aleatoric_Uncertainty'
OVERALL_UNCERTAINTY = 'Overall_Uncertainty'

# Error disparity metrics
EQUALIZED_ODDS_TPR = 'Equalized_Odds_TPR'
EQUALIZED_ODDS_TNR = 'Equalized_Odds_TNR'
EQUALIZED_ODDS_FPR = 'Equalized_Odds_FPR'
EQUALIZED_ODDS_FNR = 'Equalized_Odds_FNR'
DISPARATE_IMPACT = 'Disparate_Impact'
STATISTICAL_PARITY_DIFFERENCE = 'Statistical_Parity_Difference'
ACCURACY_PARITY = 'Accuracy_Parity'

# Stability disparity metrics
LABEL_STABILITY_RATIO = 'Label_Stability_Ratio'
IQR_PARITY = 'IQR_Parity'
STD_PARITY = 'Std_Parity'
STD_RATIO = 'Std_Ratio'
JITTER_PARITY = 'Jitter_Parity'

# Uncertainty disparity metrics
OVERALL_UNCERTAINTY_PARITY = 'Overall_Uncertainty_Parity'
OVERALL_UNCERTAINTY_RATIO = 'Overall_Uncertainty_Ratio'
ALEATORIC_UNCERTAINTY_PARITY = 'Aleatoric_Uncertainty_Parity'
ALEATORIC_UNCERTAINTY_RATIO = 'Aleatoric_Uncertainty_Ratio'

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

METRICS_FOR_PREDICT_PROBA = {MEAN_PREDICTION, STATISTICAL_BIAS,
                             STD, IQR,
                             ALEATORIC_UNCERTAINTY, OVERALL_UNCERTAINTY}
METRICS_FOR_LABELS = set([metric for metric in METRIC_TO_FUNCTION.keys() if metric not in METRICS_FOR_PREDICT_PROBA])

__all__ = [
    "mean_prediction",
    "statistical_bias_from_predict_proba",
    "statistical_bias",
    "std",
    "iqr",
    "churn",
    "jitter",
    "per_sample_label_stability",
    "label_stability",
    "entropy_from_predicted_probability",
    "conf_interval",
    "aleatoric_uncertainty",
    "overall_uncertainty",
    "METRIC_TO_FUNCTION",
    "METRICS_FOR_PREDICT_PROBA",
    "METRICS_FOR_LABELS"
]
