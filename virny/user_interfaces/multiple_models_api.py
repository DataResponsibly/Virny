import os
import sys
import traceback
import pandas as pd
from datetime import datetime, timezone

from virny.configs.constants import ModelSetting
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.analyzers.subgroup_variance_analyzer import SubgroupVarianceAnalyzer
from virny.analyzers.subgroup_error_analyzer import SubgroupErrorAnalyzer
from virny.utils.protected_groups_partitioning import create_test_protected_groups
from virny.utils.common_helpers import save_metrics_to_file


def compute_metrics_with_config(dataset: BaseFlowDataset, config, models_config: dict,
                                save_results_dir_path: str, postprocessor=None,
                                notebook_logs_stdout: bool = False, verbose: int = 0) -> dict:
    """
    Compute stability and accuracy metrics for each model in models_config. Arguments are defined as an input config object.
    Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.
    config
        Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    save_results_dir_path
        Location where to save result files with metrics
    postprocessor
        [Optional] Postprocessor object to apply to model predictions before metrics computation
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
            As for now, 0, 1, 2 levels are supported. Currently, verbose works only with notebook_logs_stdout = False.

    """
    # Currently, verbose works only with notebook_logs_stdout = False
    if notebook_logs_stdout:
        verbose = 0

    start_datetime = datetime.now(timezone.utc)
    os.makedirs(save_results_dir_path, exist_ok=True)

    # Check if a type of postprocessing_sensitive_attribute is not NoneType.
    # In other words, check if postprocessing_sensitive_attribute is defined in a config yaml.
    postprocessing_sensitive_attribute = config.postprocessing_sensitive_attribute \
        if type(config.postprocessing_sensitive_attribute) != type(None) else None

    model_metrics_dct = dict()
    models_metrics_dct = run_metrics_computation(dataset=dataset,
                                                 bootstrap_fraction=config.bootstrap_fraction,
                                                 dataset_name=config.dataset_name,
                                                 models_config=models_config,
                                                 n_estimators=config.n_estimators,
                                                 sensitive_attributes_dct=config.sensitive_attributes_dct,
                                                 model_setting=config.model_setting,
                                                 computation_mode=config.computation_mode,
                                                 postprocessor=postprocessor,
                                                 postprocessing_sensitive_attribute=postprocessing_sensitive_attribute,
                                                 save_results=False,
                                                 notebook_logs_stdout=notebook_logs_stdout,
                                                 verbose=verbose)

    # Concatenate with previous results and save them in an overwrite mode each time for backups
    for model_name in models_metrics_dct.keys():
        model_metrics_df = models_metrics_dct[model_name]
        model_metrics_dct[model_name] = model_metrics_df

        result_filename = f'Metrics_{config.dataset_name}_{model_name}_{config.n_estimators}_Estimators_{start_datetime.strftime("%Y%m%d__%H%M%S")}.csv'
        model_metrics_dct[model_name].to_csv(f'{save_results_dir_path}/{result_filename}', index=False, mode='w')

    return model_metrics_dct


def run_metrics_computation(dataset: BaseFlowDataset, bootstrap_fraction: float, dataset_name: str,
                            models_config: dict, n_estimators: int, sensitive_attributes_dct: dict,
                            model_setting: str = ModelSetting.BATCH.value, computation_mode: str = None,
                            postprocessor=None, postprocessing_sensitive_attribute: str = None,
                            save_results: bool = True, save_results_dir_path: str = None,
                            notebook_logs_stdout: bool = False, verbose: int = 0) -> dict:
    """
    Compute stability and accuracy metrics for each model in models_config.
    Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        Dataset object that contains all needed attributes like target, features, numerical_columns etc.
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    dataset_name
        Dataset name to name a result file with metrics
    models_config
        Dictionary where keys are model names, and values are initialized models
    n_estimators
        Number of estimators for bootstrap to compute subgroup stability metrics
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    model_setting
        [Optional] Currently, only batch models are supported. Default: 'batch'.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    postprocessor
        [Optional] Postprocessor object to apply to model predictions before metrics computation
    postprocessing_sensitive_attribute
        [Optional] Sensitive attribute name to apply postprocessor only to this attribute predictions
    save_results
        [Optional] If to save result metrics in a file
    save_results_dir_path
        [Optional] Location where to save result files with metrics
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
            As for now, 0, 1, 2 levels are supported.

    """
    # Set a specific tqdm type for Jupyter notebooks and python modules
    if notebook_logs_stdout:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    models_metrics_dct = dict()
    num_models = len(models_config)
    for model_idx, model_name in tqdm(enumerate(models_config.keys()),
                                      total=num_models,
                                      desc="Analyze multiple models",
                                      colour="red",
                                      file=sys.stdout):
        if verbose >= 1:
            print('\n\n', flush=True)
            print('#' * 30, f' [Model {model_idx + 1} / {num_models}] Analyze {model_name} ', '#' * 30)
        try:
            base_model = models_config[model_name]
            model_metrics_df = compute_one_model_metrics(base_model=base_model,
                                                         n_estimators=n_estimators,
                                                         dataset=dataset,
                                                         bootstrap_fraction=bootstrap_fraction,
                                                         sensitive_attributes_dct=sensitive_attributes_dct,
                                                         model_setting=model_setting,
                                                         computation_mode=computation_mode,
                                                         dataset_name=dataset_name,
                                                         base_model_name=model_name,
                                                         postprocessor=postprocessor,
                                                         postprocessing_sensitive_attribute=postprocessing_sensitive_attribute,
                                                         save_results=save_results,
                                                         save_results_dir_path=save_results_dir_path,
                                                         notebook_logs_stdout=notebook_logs_stdout,
                                                         verbose=verbose)
            models_metrics_dct[model_name] = model_metrics_df

        except Exception as err:
            print('#' * 20, f'ERROR with {model_name}', '#' * 20)
            traceback.print_exc()

        if verbose >= 1:
            print('\n\n\n')

    return models_metrics_dct


def compute_one_model_metrics(base_model, n_estimators: int, dataset: BaseFlowDataset, bootstrap_fraction: float,
                              sensitive_attributes_dct: dict, dataset_name: str, base_model_name: str,
                              postprocessor=None, postprocessing_sensitive_attribute: str = None,
                              model_setting: str = ModelSetting.BATCH.value, computation_mode: str = None, save_results: bool = True,
                              save_results_dir_path: str = None, notebook_logs_stdout: bool = False, verbose: int = 0):
    """
    Compute subgroup metrics for the base model.
    Save results in `save_results_dir_path` folder.

    Return a dataframe of model metrics.

    Parameters
    ----------
    base_model
        Base model for metrics computation
    n_estimators
        Number of estimators for bootstrap to compute subgroup variance metrics
    dataset
        BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    dataset_name
        Dataset name to name a result file with metrics
    base_model_name
        Model name to name a result file with metrics
    postprocessor
        [Optional] Postprocessor object to apply to model predictions before metrics computation
    postprocessing_sensitive_attribute
        [Optional] Sensitive attribute name to apply postprocessor only to this attribute predictions
    save_results
        [Optional] If to save result metrics in a file
    model_setting
        [Optional] Currently, only batch models are supported. Default: 'batch'.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    save_results_dir_path
        [Optional] Location where to save result files with metrics
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
            As for now, 0, 1, 2 levels are supported.

    """
    model_setting = ModelSetting.BATCH if model_setting is None else ModelSetting[model_setting.upper()]

    test_protected_groups = create_test_protected_groups(dataset.X_test, dataset.init_features_df, sensitive_attributes_dct)
    if verbose >= 2:
        print('\nProtected groups splits:')
        for g in test_protected_groups.keys():
            print(g, test_protected_groups[g].shape)

    # Compute stability metrics for subgroups
    subgroup_variance_analyzer = SubgroupVarianceAnalyzer(model_setting=model_setting,
                                                          n_estimators=n_estimators,
                                                          base_model=base_model,
                                                          base_model_name=base_model_name,
                                                          bootstrap_fraction=bootstrap_fraction,
                                                          dataset=dataset,
                                                          dataset_name=dataset_name,
                                                          sensitive_attributes_dct=sensitive_attributes_dct,
                                                          test_protected_groups=test_protected_groups,
                                                          computation_mode=computation_mode,
                                                          postprocessor=postprocessor,
                                                          postprocessing_sensitive_attribute=postprocessing_sensitive_attribute,
                                                          notebook_logs_stdout=notebook_logs_stdout,
                                                          verbose=verbose)
    y_preds, variance_metrics_df = subgroup_variance_analyzer.compute_metrics(save_results=False,
                                                                              result_filename=None,
                                                                              save_dir_path=None)

    # Compute error metrics for subgroups
    error_analyzer = SubgroupErrorAnalyzer(X_test=dataset.X_test,
                                           y_test=dataset.y_test,
                                           sensitive_attributes_dct=sensitive_attributes_dct,
                                           test_protected_groups=test_protected_groups,
                                           computation_mode=computation_mode)
    dtc_res = error_analyzer.compute_subgroup_metrics(y_preds=y_preds,
                                                      models_predictions=dict(),
                                                      save_results=False,
                                                      result_filename=None,
                                                      save_dir_path=None)
    error_metrics_df = pd.DataFrame(dtc_res)

    metrics_df = pd.concat([variance_metrics_df, error_metrics_df])
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df['Model_Name'] = base_model_name
    metrics_df['Model_Params'] = str(base_model.get_params())

    if save_results:
        # Save metrics
        result_filename = f'Metrics_{dataset_name}_{base_model_name}'
        save_metrics_to_file(metrics_df, result_filename, save_results_dir_path)

    return metrics_df
