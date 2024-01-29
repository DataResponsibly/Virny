import pandas as pd

from virny.configs.constants import ModelSetting, ComputationMode
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.analyzers.subgroup_variance_calculator import SubgroupVarianceCalculator
from virny.analyzers.batch_overall_variance_analyzer import BatchOverallVarianceAnalyzer
from virny.analyzers.batch_overall_variance_analyzer_postprocessing import BatchOverallVarianceAnalyzerPostProcessing


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
    postprocessor
        One of postprocessors from aif360 (https://aif360.readthedocs.io/en/stable/modules/algorithms.html#module-aif360.algorithms.postprocessing)
    postprocessing_sensitive_attribute
        A sensitive attribute to use for post-processing
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
         As for now, 0, 1, 2 levels are supported.

    """
    def __init__(self, model_setting: ModelSetting, n_estimators: int, base_model, base_model_name: str,
                 bootstrap_fraction: float, dataset: BaseFlowDataset, dataset_name: str,
                 sensitive_attributes_dct: dict, test_protected_groups: dict, postprocessor=None,
                 postprocessing_sensitive_attribute : str = None, computation_mode: str = None,
                 notebook_logs_stdout: bool = False, verbose: int = 0):
        if model_setting == ModelSetting.BATCH:
            if postprocessor is not None:
                print('Enabled a postprocessing mode')
                overall_variance_analyzer = BatchOverallVarianceAnalyzerPostProcessing(postprocessor=postprocessor,
                                                                                       sensitive_attribute=postprocessing_sensitive_attribute,
                                                                                       base_model=base_model,
                                                                                       base_model_name=base_model_name,
                                                                                       bootstrap_fraction=bootstrap_fraction,
                                                                                       X_train=dataset.X_train_val,
                                                                                       y_train=dataset.y_train_val,
                                                                                       X_test=dataset.X_test,
                                                                                       y_test=dataset.y_test,
                                                                                       dataset_name=dataset_name,
                                                                                       target_column=dataset.target,
                                                                                       n_estimators=n_estimators,
                                                                                       with_predict_proba=False,
                                                                                       notebook_logs_stdout=notebook_logs_stdout,
                                                                                       verbose=verbose)
            else:
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
                                                                         notebook_logs_stdout=notebook_logs_stdout,
                                                                         verbose=verbose)
        else:
            raise ValueError('model_setting is incorrect or not supported')

        self.dataset_name = overall_variance_analyzer.dataset_name
        self.n_estimators = overall_variance_analyzer.n_estimators
        self.base_model_name = overall_variance_analyzer.base_model_name

        self.__overall_variance_analyzer = overall_variance_analyzer

        with_predict_proba = False if postprocessor is not None else True
        self.__subgroup_variance_calculator = SubgroupVarianceCalculator(X_test=dataset.X_test,
                                                                         y_test=dataset.y_test,
                                                                         sensitive_attributes_dct=sensitive_attributes_dct,
                                                                         test_protected_groups=test_protected_groups,
                                                                         computation_mode=computation_mode,
                                                                         with_predict_proba=with_predict_proba)
        self.overall_variance_metrics_dct = dict()
        self.subgroup_variance_metrics_dct = dict()

    def set_test_sets(self, new_X_test, new_y_test):
        self.__overall_variance_analyzer.X_test = new_X_test
        self.__overall_variance_analyzer.y_test = new_y_test
        self.__subgroup_variance_calculator.X_test = new_X_test
        self.__subgroup_variance_calculator.y_test = new_y_test

    def set_test_protected_groups(self, new_test_protected_groups):
        self.__subgroup_variance_calculator.test_protected_groups = new_test_protected_groups

    def compute_metrics(self, save_results: bool, result_filename: str = None,
                        save_dir_path: str = None, with_fit: bool = True):
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
        with_fit
            If to fit estimators in bootstrap

        """
        y_preds, y_test_true = self.__overall_variance_analyzer.compute_metrics(save_results=False, with_fit=with_fit)
        y_preds = pd.Series(y_preds, index=y_test_true.index)
        self.overall_variance_metrics_dct = self.__overall_variance_analyzer.prediction_metrics

        # Count and display fairness metrics
        self.__subgroup_variance_calculator.set_overall_variance_metrics(self.overall_variance_metrics_dct)
        self.subgroup_variance_metrics_dct = self.__subgroup_variance_calculator.compute_subgroup_metrics(
            y_preds, self.__overall_variance_analyzer.models_predictions,
            save_results, result_filename, save_dir_path
        )

        return y_preds, pd.DataFrame(self.subgroup_variance_metrics_dct)
