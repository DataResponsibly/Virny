"""
This module contains functions for computing subgroup variance and error metrics.
"""
from .stability_metrics import (
    compute_std_mean_iqr_metrics,
    compute_churn,
    compute_jitter,
    compute_entropy_from_predicted_probability,
    compute_conf_interval,
    compute_std_mean_iqr_metrics,
    compute_per_sample_accuracy,
)


__all__ = [
    "compute_std_mean_iqr_metrics",
    "compute_churn",
    "compute_jitter",
    "compute_entropy_from_predicted_probability",
    "compute_conf_interval",
    "compute_std_mean_iqr_metrics",
    "compute_per_sample_accuracy",
]
