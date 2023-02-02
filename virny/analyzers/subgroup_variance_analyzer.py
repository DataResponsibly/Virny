import pandas as pd

from virny.configs.constants import ModelSetting
from virny.custom_classes.generic_pipeline import GenericPipeline
from virny.analyzers.subgroup_variance_calculator import SubgroupVarianceCalculator
from virny.analyzers.batch_overall_variance_analyzer import BatchOverallVarianceAnalyzer


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
    base_pipeline
        Initialized object of GenericPipeline class
    dataset_name
        Name of dataset, used for correct results naming

    """
    def __init__(self, model_setting: ModelSetting, n_estimators: int, base_model, base_model_name: str,
                 bootstrap_fraction: float, base_pipeline: GenericPipeline, dataset_name: str):
        if model_setting == ModelSetting.BATCH:
            overall_variance_analyzer = BatchOverallVarianceAnalyzer(base_model, base_model_name, bootstrap_fraction,
                                                                     base_pipeline.X_train_val, base_pipeline.y_train_val,
                                                                     base_pipeline.X_test, base_pipeline.y_test,
                                                                     dataset_name, base_pipeline.target, n_estimators)
        else:
            raise ValueError('model_setting is incorrect or not supported')

        self.dataset_name = overall_variance_analyzer.dataset_name
        self.n_estimators = overall_variance_analyzer.n_estimators
        self.base_model_name = overall_variance_analyzer.base_model_name

        self.__overall_variance_analyzer = overall_variance_analyzer
        self.__subgroup_variance_calculator = SubgroupVarianceCalculator(base_pipeline.X_test, base_pipeline.y_test,
                                                                         base_pipeline.sensitive_attributes_dct,
                                                                         base_pipeline.test_protected_groups)
        self.overall_variance_metrics_dct = dict()
        self.subgroup_variance_metrics_dct = dict()

    def compute_metrics(self, save_results: bool, result_filename: str = None, save_dir_path: str = None,
                        make_plots: bool = True):
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

        """
        y_preds, y_test_true = self.__overall_variance_analyzer.compute_metrics(make_plots, save_results=False)
        self.overall_variance_metrics_dct = self.__overall_variance_analyzer.get_metrics_dict()

        # Count and display fairness metrics
        self.__subgroup_variance_calculator.set_overall_variance_metrics(self.overall_variance_metrics_dct)
        self.subgroup_variance_metrics_dct = self.__subgroup_variance_calculator.compute_subgroup_metrics(
            self.__overall_variance_analyzer.models_predictions, save_results, result_filename, save_dir_path
        )

        return y_preds, pd.DataFrame(self.subgroup_variance_metrics_dct)
