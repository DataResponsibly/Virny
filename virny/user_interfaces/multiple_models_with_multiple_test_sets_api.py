import sys
import traceback
import pandas as pd
from datetime import datetime, timezone

from virny.configs.constants import ModelSetting
from virny.utils.protected_groups_partitioning import create_test_protected_groups
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.analyzers.subgroup_variance_analyzer import SubgroupVarianceAnalyzer
from virny.analyzers.subgroup_error_analyzer import SubgroupErrorAnalyzer


def compute_metrics_with_multiple_test_sets(dataset: BaseFlowDataset, extra_test_sets_lst,
                                            config, models_config: dict, custom_tbl_fields_dct: dict,
                                            db_writer_func, notebook_logs_stdout: bool = False, verbose: int = 0):
    """
    Compute stability and accuracy metrics for each model in models_config based on dataset.X_test and each extra test set
     in extra_test_sets_lst. Arguments are defined as an input config object. Save results to a database after each run
      appending fields and value from custom_tbl_fields_dct and using db_writer_func.
      Index of each test set is also added as a separate column in out final records in the database
      (0 index -- for dataset.X_test, 1 and greater -- for each extra test set in extra_test_sets_lst, keeping the original sequence).

    Parameters
    ----------
    dataset
        BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.
    extra_test_sets_lst
        List of extra test sets like [(X_test1, y_test1), (X_test2, y_test2), ...] to compute metrics
        that are not equal to original dataset.X_test and dataset.y_test
    config
        Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    custom_tbl_fields_dct
        Dictionary where keys are column names and values to add to inserted metrics during saving results to a database
    db_writer_func
        Python function object has one argument (run_models_metrics_df) and save this metrics df to a target database
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

    models_metrics_dct = run_metrics_computation_with_multiple_test_sets(dataset=dataset,
                                                                         bootstrap_fraction=config.bootstrap_fraction,
                                                                         dataset_name=config.dataset_name,
                                                                         extra_test_sets_lst=extra_test_sets_lst,
                                                                         models_config=models_config,
                                                                         n_estimators=config.n_estimators,
                                                                         sensitive_attributes_dct=config.sensitive_attributes_dct,
                                                                         model_setting=config.model_setting,
                                                                         computation_mode=config.computation_mode,
                                                                         notebook_logs_stdout=notebook_logs_stdout,
                                                                         verbose=verbose)
    # Concatenate current run metrics with previous results and
    # create melted_model_metrics_df to save it in a database
    run_models_metrics_df = pd.DataFrame()
    for model_name in models_metrics_dct.keys():
        model_metrics_dfs_lst = models_metrics_dct[model_name]
        for idx, model_metrics_df in enumerate(model_metrics_dfs_lst):
            model_metrics_df['Dataset_Name'] = config.dataset_name
            model_metrics_df['Num_Estimators'] = config.n_estimators
            model_metrics_df['Test_Set_Index'] = idx

            # Extend df with technical columns
            model_metrics_df['Tag'] = 'OK'
            model_metrics_df['Record_Create_Date_Time'] = datetime.now(timezone.utc)
            for column, value in custom_tbl_fields_dct.items():
                model_metrics_df[column] = value

            subgroup_names = [col for col in model_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
            melted_model_metrics_df = model_metrics_df.melt(id_vars=[col for col in model_metrics_df.columns if col not in subgroup_names],
                                                            value_vars=subgroup_names,
                                                            var_name="Subgroup",
                                                            value_name="Metric_Value")
            run_models_metrics_df = pd.concat([run_models_metrics_df, melted_model_metrics_df])

    # Save results for this run in a database
    db_writer_func(run_models_metrics_df)

    if verbose >= 1:
        print('Metrics computation interface was successfully executed!')


def run_metrics_computation_with_multiple_test_sets(dataset: BaseFlowDataset, bootstrap_fraction: float, dataset_name: str,
                                                    extra_test_sets_lst: list, models_config: dict, n_estimators: int,
                                                    sensitive_attributes_dct: dict, model_setting: str = ModelSetting.BATCH.value,
                                                    computation_mode: str = None, notebook_logs_stdout: bool = False,
                                                    verbose: int = 0) -> dict:
    """
    Compute stability and accuracy metrics for each model in models_config based on dataset.X_test and each extra test set
     in extra_test_sets_lst. Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        Dataset object that contains all needed attributes like target, features, numerical_columns etc.
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    dataset_name
        Dataset name to name a result file with metrics
    extra_test_sets_lst
        List of extra test sets like [(X_test1, y_test1), (X_test2, y_test2), ...] to compute metrics
    models_config
        Dictionary where keys are model names, and values are initialized models
    n_estimators
        Number of estimators for bootstrap to compute subgroup stability metrics
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    model_setting
        Currently, only batch models are supported. Default: 'batch'.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
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
            print('#' * 30, f' [Model {model_idx + 1} / {num_models}] Analyze {model_name} ', '#' * 30)
        try:
            base_model = models_config[model_name]
            model_metrics_dfs_lst = compute_one_model_metrics_with_multiple_test_sets(base_model=base_model,
                                                                                      n_estimators=n_estimators,
                                                                                      dataset=dataset,
                                                                                      extra_test_sets_lst=extra_test_sets_lst,
                                                                                      bootstrap_fraction=bootstrap_fraction,
                                                                                      sensitive_attributes_dct=sensitive_attributes_dct,
                                                                                      model_setting=model_setting,
                                                                                      computation_mode=computation_mode,
                                                                                      dataset_name=dataset_name,
                                                                                      base_model_name=model_name,
                                                                                      notebook_logs_stdout=notebook_logs_stdout,
                                                                                      verbose=verbose)
            models_metrics_dct[model_name] = model_metrics_dfs_lst
        except Exception as err:
            print('#' * 20, f'ERROR with {model_name}', '#' * 20)
            traceback.print_exc()

        if verbose >= 1:
            print('\n\n\n')

    return models_metrics_dct


def compute_one_model_metrics_with_multiple_test_sets(base_model, n_estimators: int,
                                                      dataset: BaseFlowDataset, extra_test_sets_lst: list,
                                                      bootstrap_fraction: float, sensitive_attributes_dct: dict,
                                                      dataset_name: str, base_model_name: str,
                                                      model_setting: str = ModelSetting.BATCH.value,
                                                      computation_mode: str = None, notebook_logs_stdout: bool = False,
                                                      verbose: int = 0):
    """
    Compute subgroup metrics for the base model based on dataset.X_test and each extra test set in extra_test_sets_lst.
    Save results in `save_results_dir_path` folder.

    Return a dataframe of model metrics.

    Parameters
    ----------
    base_model
        Base model for metrics computation
    n_estimators
        Number of estimators for bootstrap to compute subgroup stability metrics
    dataset
        BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.
    extra_test_sets_lst
        List of extra test sets like [(X_test1, y_test1, init_features_df1), (X_test2, y_test2, init_features_df2), ...]
         to compute metrics.
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    dataset_name
        Dataset name to name a result file with metrics
    base_model_name
        Model name to name a result file with metrics
    model_setting
        Currently, only batch models are supported. Default: 'batch'.
    computation_mode
        [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
            As for now, 0, 1, 2 levels are supported.

    """
    model_setting = ModelSetting.BATCH if model_setting is None else ModelSetting[model_setting.upper()]
    subgroup_variance_analyzer = SubgroupVarianceAnalyzer(model_setting=model_setting,
                                                          n_estimators=n_estimators,
                                                          base_model=base_model,
                                                          base_model_name=base_model_name,
                                                          bootstrap_fraction=bootstrap_fraction,
                                                          dataset=dataset,  # will be replaced in the below for-cycle
                                                          dataset_name=dataset_name,
                                                          sensitive_attributes_dct=sensitive_attributes_dct,
                                                          test_protected_groups=dict(),  # stub for this attribute
                                                          computation_mode=computation_mode,
                                                          notebook_logs_stdout=notebook_logs_stdout,
                                                          verbose=verbose)

    test_sets_lst = [(dataset.X_test, dataset.y_test, dataset.init_features_df)] + extra_test_sets_lst
    all_test_sets_metrics_lst = []
    for set_idx, (new_X_test, new_y_test, cur_init_features_df) in enumerate(test_sets_lst):
        new_test_protected_groups = create_test_protected_groups(new_X_test, cur_init_features_df, sensitive_attributes_dct)
        if verbose >= 2:
            print(f'\nProtected groups splits for test set index #{set_idx}:')
            for g in new_test_protected_groups.keys():
                print(g, new_test_protected_groups[g].shape)

        # Replace test sets and protected groups for each new test set
        subgroup_variance_analyzer.set_test_sets(new_X_test, new_y_test)
        subgroup_variance_analyzer.set_test_protected_groups(new_test_protected_groups)

        # Compute stability metrics for subgroups
        y_preds, variance_metrics_df = subgroup_variance_analyzer.compute_metrics(save_results=False,
                                                                                  result_filename=None,
                                                                                  save_dir_path=None,
                                                                                  with_fit=True if set_idx == 0 else False)

        # Compute accuracy metrics for subgroups
        error_analyzer = SubgroupErrorAnalyzer(X_test=new_X_test,
                                               y_test=new_y_test,
                                               sensitive_attributes_dct=sensitive_attributes_dct,
                                               test_protected_groups=new_test_protected_groups,
                                               computation_mode=computation_mode)
        dtc_res = error_analyzer.compute_subgroup_metrics(y_preds,
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

        all_test_sets_metrics_lst.append(metrics_df)

    return all_test_sets_metrics_lst
