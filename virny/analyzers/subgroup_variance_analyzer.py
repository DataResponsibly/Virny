import pandas as pd

from virny.configs.constants import ModelSetting
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.analyzers.subgroup_variance_calculator import SubgroupVarianceCalculator
from virny.analyzers.batch_overall_variance_analyzer import BatchOverallVarianceAnalyzer
from virny.analyzers.incremental_overall_variance_analyzer import IncrementalOverallVarianceAnalyzer


class SubgroupVarianceAnalyzer:
    """
    Analyzer to compute variance metrics for subgroups.

    Parameters
    ----------
    model_setting
        Model learning type; a constant from virny.configs.constants.ModelSetting
    n_estimators
        Number of estimators for bootstrap
    base_model
        Initialized base model to analyze
    base_model_name
        Model name
    bootstrap_fraction
        [0-1], fraction from train_pd_dataset for fitting an ensemble of base models
    dataset
        Initialized object of GenericPipeline class
    dataset_name
        Name of dataset, used for correct results naming
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    test_protected_groups
        A dictionary of protected groups where keys are subgroup names,
         and values are X_test row indexes correspondent to this subgroup.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
         As for now, 0, 1, 2 levels are supported.

    """
    def __init__(self, model_setting: ModelSetting, n_estimators: int, base_model, base_model_name: str,
                 bootstrap_fraction: float, dataset: BaseFlowDataset, dataset_name: str,
                 sensitive_attributes_dct: dict, test_protected_groups: dict, computation_mode: str = None, verbose: int = 0):
        if model_setting == ModelSetting.BATCH:
            overall_variance_analyzer = BatchOverallVarianceAnalyzer(base_model=base_model,
                                                                     base_model_name=base_model_name,
                                                                     bootstrap_fraction=bootstrap_fraction,
                                                                     X_train=dataset.X_train_val,
                                                                     y_train=dataset.y_train_val,
                                                                     X_test=dataset.X_test,
                                                                     y_test=dataset.y_test,
                                                                     dataset_name=dataset_name,
                                                                     target_column=dataset.target,
                                                                     n_estimators=n_estimators,
                                                                     verbose=verbose)
        elif model_setting == ModelSetting.INCREMENTAL:
            overall_variance_analyzer = IncrementalOverallVarianceAnalyzer(base_model=base_model,
                                                                           base_model_name=base_model_name,
                                                                           bootstrap_fraction=bootstrap_fraction,
                                                                           X_train=dataset.X_train_val,
                                                                           y_train=dataset.y_train_val,
                                                                           X_test=dataset.X_test,
                                                                           y_test=dataset.y_test,
                                                                           dataset_name=dataset_name,
                                                                           target_column=dataset.target,
                                                                           n_estimators=n_estimators,
                                                                           verbose=verbose)
        else:
            raise ValueError('model_setting is incorrect or not supported')

        self.dataset_name = overall_variance_analyzer.dataset_name
        self.n_estimators = overall_variance_analyzer.n_estimators
        self.base_model_name = overall_variance_analyzer.base_model_name

        self.__overall_variance_analyzer = overall_variance_analyzer
        self.__subgroup_variance_calculator = SubgroupVarianceCalculator(X_test=dataset.X_test,
                                                                         y_test=dataset.y_test,
                                                                         sensitive_attributes_dct=sensitive_attributes_dct,
                                                                         test_protected_groups=test_protected_groups,
                                                                         computation_mode=computation_mode)
        self.overall_variance_metrics_dct = dict()
        self.subgroup_variance_metrics_dct = dict()

    def set_test_sets(self, new_X_test, new_y_test):
        self.__overall_variance_analyzer.X_test = new_X_test
        self.__overall_variance_analyzer.y_test = new_y_test
        self.__subgroup_variance_calculator.X_test = new_X_test
        self.__subgroup_variance_calculator.y_test = new_y_test

    def set_test_protected_groups(self, new_test_protected_groups):
        self.__subgroup_variance_calculator.test_protected_groups = new_test_protected_groups

    def compute_metrics(self, save_results: bool, result_filename: str = None, save_dir_path: str = None,
                        make_plots: bool = True, with_fit: bool = True):
        """
        Measure variance metrics for subgroups for the base model. Display variance plots for analysis if needed.
         Save results to a .csv file if needed.

        Return averaged bootstrap predictions and a pandas dataframe of variance metrics for subgroups.

        Parameters
        ----------
        save_results
            If we need to save result metrics in a file
        result_filename
            [Optional] Filename for results to save
        save_dir_path
            [Optional] Location where to save the results file
        make_plots
            If to display plots for analysis
        with_fit
            If to fit estimators in bootstrap

        """
        y_preds, y_test_true = self.__overall_variance_analyzer.compute_metrics(make_plots, save_results=False, with_fit=with_fit)
        self.overall_variance_metrics_dct = self.__overall_variance_analyzer.get_metrics_dict()

        # Count and display fairness metrics
        self.__subgroup_variance_calculator.set_overall_variance_metrics(self.overall_variance_metrics_dct)
        self.subgroup_variance_metrics_dct = self.__subgroup_variance_calculator.compute_subgroup_metrics(
            self.__overall_variance_analyzer.models_predictions, save_results, result_filename, save_dir_path
        )

        return y_preds, pd.DataFrame(self.subgroup_variance_metrics_dct)
