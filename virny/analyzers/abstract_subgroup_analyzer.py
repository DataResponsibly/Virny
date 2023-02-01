import os
import pandas as pd

from datetime import datetime, timezone
from abc import ABCMeta, abstractmethod


class AbstractSubgroupAnalyzer(metaclass=ABCMeta):
    """
    Abstract class for a subgroup analyzer to compute metrics for subgroups.

    Parameters
    ----------
    X_test
        Processed features test set
    y_test
        Targets test set
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these attributes
    test_protected_groups
        A dictionary where keys are sensitive attributes, and values input dataset rows
         that are correspondent to these sensitive attributes

    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, sensitive_attributes_dct: dict, test_protected_groups: dict):
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.X_test = X_test
        self.y_test = y_test
        self.test_protected_groups = test_protected_groups
        self.subgroup_metrics_dict = {}

    @abstractmethod
    def _compute_metrics(self, y_test, y_preds):
        pass

    def compute_subgroup_metrics(self, y_preds, save_results: bool,
                                 result_filename: str = None, save_dir_path: str = None):
        """
        Compute metrics for each subgroup in self.test_protected_groups using _compute_metrics method.

        Return a dictionary where keys are subgroup names, and values are subgroup metrics.
        
        Parameters
        ----------
        y_preds
            Models predictions
        save_results
            If to save results in a file
        result_filename
            [Optional] Filename for results to save
        save_dir_path
            [Optional] Location where to save the results file

        """
        y_pred_all = pd.Series(y_preds, index=self.y_test.index)

        # Compute metrics for each subgroup
        results = dict()
        results['overall'] = self._compute_metrics(self.y_test, y_pred_all)
        for group_name in self.test_protected_groups.keys():
            X_test_group = self.test_protected_groups[group_name]
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index], y_pred_all[X_test_group.index])

        self.subgroup_metrics_dict = results
        if save_results:
            self.save_metrics_to_file(result_filename, save_dir_path)

        return self.subgroup_metrics_dict

    def save_metrics_to_file(self, result_filename: str, save_dir_path: str):
        """
        Parameters
        ----------
        result_filename

        save_dir_path
        """
        metrics_df = pd.DataFrame(self.subgroup_metrics_dict)
        os.makedirs(save_dir_path, exist_ok=True)

        now = datetime.now(timezone.utc)
        date_time_str = now.strftime("%Y%m%d__%H%M%S")
        filename = f"{result_filename}_{date_time_str}.csv"
        metrics_df = metrics_df.reset_index()
        metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)
