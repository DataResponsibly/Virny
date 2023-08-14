import os
import pandas as pd

from datetime import datetime, timezone
from abc import ABCMeta, abstractmethod

from virny.configs.constants import ComputationMode


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
    computation_mode
        A mode to compute metrics. It can have two values 'error_analysis' and default (None).

    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, sensitive_attributes_dct: dict,
                 test_protected_groups: dict, computation_mode: str = None):
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.X_test = X_test
        self.y_test = y_test
        self.test_protected_groups = test_protected_groups
        self.computation_mode = computation_mode
        self.subgroup_metrics_dict = {}

    @abstractmethod
    def _compute_metrics(self, y_test, y_preds):
        pass

    def _partition_and_compute_metrics(self, y_pred_all, results: dict):
        for group_name in self.test_protected_groups.keys():
            X_test_group = self.test_protected_groups[group_name]
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index], y_pred_all[X_test_group.index])

        return results

    def _partition_and_compute_metrics_for_error_analysis(self, y_preds, results: dict):
        """
        Partition predictions on correct and incorrect and compute subgroup metrics for each of the partitions.
        Used for the 'error_analysis' mode.

        :param y_preds: a list of predictions
        :param results: a dict to add subgroup metrics for each partition
        """
        for group_name in self.test_protected_groups.keys():
            X_test_group = self.test_protected_groups[group_name]
            group_y_true = self.y_test[X_test_group.index]
            group_y_preds = y_preds[X_test_group.index]

            # Define indexes of each partition of the group: overall group indexes,
            # correct preds group indexes, incorrect preds group indexes
            correct_preds_indexes = group_y_true.index[group_y_true == group_y_preds]
            incorrect_preds_indexes = group_y_true.index[group_y_true != group_y_preds]
            partition_indexes_dct = {
                group_name: X_test_group.index,
                f'{group_name}_correct': correct_preds_indexes,
                f'{group_name}_incorrect': incorrect_preds_indexes,
            }

            # Compute metrics for each group partition
            for group_partition_name, partition_indexes in partition_indexes_dct.items():
                metrics_dct = self._compute_metrics(self.y_test[partition_indexes], y_preds[partition_indexes])
                metrics_dct['Sample_Size'] = len(partition_indexes)
                results[group_partition_name] = metrics_dct

        return results

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

        # Compute overall metrics
        results = dict()
        metrics_dct = self._compute_metrics(self.y_test, y_pred_all)
        metrics_dct['Sample_Size'] = self.y_test.shape[0]
        results['overall'] = metrics_dct

        # Compute metrics for subgroups
        if self.computation_mode == ComputationMode.ERROR_ANALYSIS.value:
            results = self._partition_and_compute_metrics_for_error_analysis(y_pred_all, results)
        else:
            results = self._partition_and_compute_metrics(y_pred_all, results)

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
