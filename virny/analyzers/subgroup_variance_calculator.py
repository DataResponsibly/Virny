import pandas as pd
from colorama import Fore

from virny.metrics import METRIC_TO_FUNCTION, METRICS_FOR_LABELS
from virny.configs.constants import ComputationMode
from virny.utils.stability_utils import count_prediction_metrics, combine_bootstrap_predictions
from virny.analyzers.abstract_subgroup_analyzer import AbstractSubgroupAnalyzer


class SubgroupVarianceCalculator(AbstractSubgroupAnalyzer):
    """
    Calculator that calculates variance metrics for subgroups.

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
         that are correspondent to these sensitive attributes.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    with_predict_proba
        [Optional] A flag if model can return probabilities for its predictions.
         If no, only metrics based on labels (not labels and probabilities) will be computed.

    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, sensitive_attributes_dct: dict,
                 test_protected_groups=None, computation_mode: str = None, with_predict_proba: bool = True):
        super().__init__(X_test, y_test, sensitive_attributes_dct, test_protected_groups, computation_mode)
        self.with_predict_proba = with_predict_proba
        self.overall_variance_metrics = None
        self.subgroup_variance_metrics_dict = None

    def set_overall_variance_metrics(self, overall_variance_metrics):
        self.overall_variance_metrics = overall_variance_metrics

    def _partition_and_compute_metrics(self, models_predictions, results: dict):
        for group_name in self.test_protected_groups.keys():
            X_test_group = self.test_protected_groups[group_name]
            group_models_predictions = {
                model_idx: models_predictions[model_idx][X_test_group.index].reset_index(drop=True)
                for model_idx in models_predictions.keys()
            }
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index].reset_index(drop=True),
                                                        group_models_predictions)

        return results

    def _partition_and_compute_metrics_for_error_analysis(self, y_preds, models_predictions: dict, results: dict):
        """
        Partition predictions on correct and incorrect and compute subgroup metrics for each of the partitions.
        Used for the 'error_analysis' mode.

        Parameters
        ----------
        y_preds
            Is not used in this function, but needed for function argument consistency.
        models_predictions
            A dictionary of models predictions.
        results
            A dict to add subgroup metrics for each partition
        """

        # Partition and compute subgroup metrics
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
                group_models_predictions = {
                    model_idx: models_predictions[model_idx][partition_indexes].reset_index(drop=True)
                    for model_idx in models_predictions.keys()
                }
                if partition_indexes.shape[0] == 0:
                    print(Fore.YELLOW + f'WARNING: "{group_partition_name}" group is empty. Stability metrics are set to None.' + Fore.RESET, flush=True)
                    metrics_dct = dict()
                    metric_names = list(METRIC_TO_FUNCTION.keys()) if self.with_predict_proba else METRICS_FOR_LABELS
                    for metric in metric_names:
                        metrics_dct[metric] = None
                else:
                    metrics_dct = self._compute_metrics(self.y_test[partition_indexes].reset_index(drop=True),
                                                        group_models_predictions)
                results[group_partition_name] = metrics_dct

        return results

    def _compute_metrics(self, y_test: pd.DataFrame, group_models_predictions):
        _, prediction_metrics = count_prediction_metrics(y_test, group_models_predictions,
                                                         with_predict_proba=self.with_predict_proba)
        return prediction_metrics

    def compute_subgroup_metrics(self, y_preds, models_predictions: dict, save_results: bool,
                                 result_filename: str = None, save_dir_path: str = None):
        """
        Compute variance metrics for subgroups.

        Return a dict of dicts where key is 'overall' or a subgroup name, and value is a dict of metrics for this subgroup.

        Parameters
        ----------
        models_predictions
            Dict of lists where key is a model index, and value is a list of model predictions based on X_test set
        save_results
            If we need to save result metrics in a file
        result_filename
            [Optional] Filename for results to save
        save_dir_path
            [Optional] Location where to save the results file

        """
        models_predictions = {
            model_idx: pd.Series(models_predictions[model_idx], index=self.y_test.index)
            for model_idx in models_predictions.keys()
        }

        # Compute overall stability metrics
        results = dict()
        results['overall'] = self.overall_variance_metrics

        # Compute stability metrics for subgroups
        if self.computation_mode == ComputationMode.ERROR_ANALYSIS.value:
            results = self._partition_and_compute_metrics_for_error_analysis(y_preds, models_predictions, results)
        else:
            results = self._partition_and_compute_metrics(models_predictions, results)

        self.subgroup_variance_metrics_dict = results
        if save_results:
            self.save_metrics_to_file(result_filename, save_dir_path)

        return self.subgroup_variance_metrics_dict
