from enum import Enum
from dataclasses import dataclass


@dataclass
class CountPredictionStatsResponse:
    jitter: float
    means_lst: list
    stds_lst: list
    iqr_lst: list
    entropy_lst: list
    per_sample_accuracy_lst: list
    label_stability_lst: list


class ModelSetting(Enum):
    INCREMENTAL = "incremental"
    BATCH = "batch"


INTERSECTION_SIGN = '&'
MODELS_TUNING_SEED = 42
MODELS_TUNING_TEST_SET_FRACTION = 0.2
