import pandas as pd

from virny.analyzers.abstract_subgroup_analyzer import AbstractSubgroupAnalyzer
from virny.utils.common_helpers import confusion_matrix_metrics


class SubgroupStatisticalBiasAnalyzer(AbstractSubgroupAnalyzer):
    """
    Analyzer to compute statistical bias metrics for subgroups.

    Parameters
    ----------
    X_test
        Processed features test set
    y_test
        Targets test set
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these subgroups
    test_protected_groups
        A dictionary where keys are sensitive attributes, and values input dataset rows
         that are correspondent to these sensitive attributes

    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 sensitive_attributes_dct: dict, test_protected_groups: dict=None):
        super().__init__(X_test, y_test, sensitive_attributes_dct, test_protected_groups)

    def _compute_metrics(self, y_test: pd.DataFrame, y_preds: list):
        """
        Compute metrics for subgroups using a confusion matrix
        """
        return confusion_matrix_metrics(y_test, y_preds)
