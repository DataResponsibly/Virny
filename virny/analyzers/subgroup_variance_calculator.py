import numpy as np
import pandas as pd

from virny.utils.stability_utils import count_prediction_stats
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
         that are correspondent to these sensitive attributes

    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, sensitive_attributes_dct: dict, test_protected_groups=None):
        super().__init__(X_test, y_test, sensitive_attributes_dct, test_protected_groups)
        self.overall_variance_metrics = None
        self.subgroup_variance_metrics_dict = None

    def set_overall_variance_metrics(self, overall_variance_metrics):
        self.overall_variance_metrics = overall_variance_metrics

    def _compute_metrics(self, y_test: pd.DataFrame, group_models_predictions):
        _, _, prediction_stats = count_prediction_stats(y_test, group_models_predictions)
        return {
            'Mean': np.mean(prediction_stats.means_lst),
            'Std': np.mean(prediction_stats.stds_lst),
            'IQR': np.mean(prediction_stats.iqr_lst),
            'Entropy': np.mean(prediction_stats.entropy_lst),
            'Jitter': prediction_stats.jitter,
            'Per_Sample_Accuracy': np.mean(prediction_stats.per_sample_accuracy_lst),
            'Label_Stability': np.mean(prediction_stats.label_stability_lst),
        }

    def compute_subgroup_metrics(self, models_predictions: dict, save_results: bool,
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

        # Compute variance metrics for subgroups
        results = dict()
        results['overall'] = self.overall_variance_metrics
        for group_name in self.test_protected_groups.keys():
            X_test_group = self.test_protected_groups[group_name]
            group_models_predictions = {
                model_idx: models_predictions[model_idx][X_test_group.index].reset_index(drop=True)
                for model_idx in models_predictions.keys()
            }
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index].reset_index(drop=True),
                                                        group_models_predictions)

        self.subgroup_variance_metrics_dict = results
        if save_results:
            self.save_metrics_to_file(result_filename, save_dir_path)

        return self.subgroup_variance_metrics_dict
