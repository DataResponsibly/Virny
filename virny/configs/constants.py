import numpy as np

from enum import Enum
from dataclasses import dataclass


@dataclass
class CountPredictionStatsResponse:
    jitter: float
    means_lst: list
    stds_lst: list
    iqr_lst: list
    mean_ensemble_entropy_lst: list
    overall_entropy_lst: np.ndarray
    statistical_bias_lst: np.ndarray
    per_sample_accuracy_lst: list
    label_stability_lst: list


class ModelSetting(Enum):
    INCREMENTAL = "incremental"
    BATCH = "batch"


class ComputationMode(Enum):
    ERROR_ANALYSIS = "error_analysis"


class ReportType(Enum):
    MULTIPLE_RUNS_MULTIPLE_MODELS = "multiple_runs_multiple_models"
    ONE_RUN_MULTIPLE_MODELS = "one_run_multiple_models"
    ONE_RUN_ONE_MODEL = "one_run_one_model"


INTERSECTION_SIGN = '&'
MODELS_TUNING_SEED = 42
MODELS_TUNING_TEST_SET_FRACTION = 0.2
