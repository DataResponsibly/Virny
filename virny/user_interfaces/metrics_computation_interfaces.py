import os
import random
import traceback
import pandas as pd
from tqdm.notebook import tqdm
from datetime import datetime, timezone
from IPython.display import display

from virny.configs.constants import ModelSetting
from virny.utils.common_helpers import reset_model_seed
from virny.utils.custom_initializers import create_base_pipeline
from virny.custom_classes.base_dataset import BaseDataset
from virny.analyzers.subgroup_variance_analyzer import SubgroupVarianceAnalyzer
from virny.utils.common_helpers import save_metrics_to_file
from virny.analyzers.subgroup_statistical_bias_analyzer import SubgroupStatisticalBiasAnalyzer


def compute_model_metrics_with_config(base_model, model_name: str, dataset: BaseDataset, config, save_results_dir_path: str,
                                      model_seed: int = None, save_results: bool = True, debug_mode: bool = False) -> pd.DataFrame:
    """
    Compute subgroup metrics for the base model. Arguments are defined as an input config object.
    Save results in `save_results_dir_path` folder.

    Return a dataframe of model metrics.

    Parameters
    ----------
    base_model
        Base model for metrics computation
    model_name
        Model name to name a result file with metrics
    dataset
        BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.
    config
        Object that contains test_set_fraction, bootstrap_fraction, dataset_name,
         n_estimators, sensitive_attributes_dct attributes
    save_results_dir_path
        Location where to save result files with metrics
    model_seed
        [Optional] Model seed
    save_results
        [Optional] If to save result metrics in a file
    debug_mode
        [Optional] Enable or disable extra logs

    """
    if model_seed is None:
        model_seed = random.randint(1, 1000)

    return compute_model_metrics(base_model=base_model,
                                 n_estimators=config.n_estimators,
                                 dataset=dataset,
                                 test_set_fraction=config.test_set_fraction,
                                 bootstrap_fraction=config.bootstrap_fraction,
                                 sensitive_attributes_dct=config.sensitive_attributes_dct,
                                 model_seed=model_seed,
                                 dataset_name=config.dataset_name,
                                 base_model_name=model_name,
                                 save_results=save_results,
                                 save_results_dir_path=save_results_dir_path,
                                 debug_mode=debug_mode)


def compute_model_metrics(base_model, n_estimators: int, dataset: BaseDataset, test_set_fraction: float, bootstrap_fraction: float,
                          sensitive_attributes_dct: dict, model_seed: int, dataset_name: str, base_model_name: str,
                          model_setting: str = ModelSetting.BATCH.value, save_results: bool = True,
                          save_results_dir_path: str = None, debug_mode: bool = False):
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
        BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.
    test_set_fraction
        Fraction of the whole dataset in range [0.0 - 1.0] to create a test set
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    model_seed
        Model seed
    dataset_name
        Dataset name to name a result file with metrics
    base_model_name
        Model name to name a result file with metrics
    save_results
        [Optional] If to save result metrics in a file
    save_results_dir_path
        [Optional] Location where to save result files with metrics
    debug_mode
        [Optional] Enable or disable extra logs

    """
    model_setting = ModelSetting.BATCH if model_setting is None else ModelSetting[model_setting.upper()]

    base_model = reset_model_seed(base_model, model_seed)
    base_pipeline = create_base_pipeline(dataset, sensitive_attributes_dct, model_seed, test_set_fraction)
    if debug_mode:
        print('\nProtected groups splits:')
        for g in base_pipeline.test_protected_groups.keys():
            print(g, base_pipeline.test_protected_groups[g].shape)

        print('\n\nTop rows of processed X train + validation set: ')
        display(base_pipeline.X_train_val.head(10))

    # Compute variance metrics for subgroups
    subgroup_variance_analyzer = SubgroupVarianceAnalyzer(model_setting, n_estimators, base_model, base_model_name,
                                                          bootstrap_fraction, base_pipeline, dataset_name)
    y_preds, variance_metrics_df = subgroup_variance_analyzer.compute_metrics(save_results=False,
                                                                              result_filename=None,
                                                                              save_dir_path=None,
                                                                              make_plots=False)

    # Compute bias metrics for subgroups
    bias_analyzer = SubgroupStatisticalBiasAnalyzer(base_pipeline.X_test, base_pipeline.y_test,
                                                    base_pipeline.sensitive_attributes_dct, base_pipeline.test_protected_groups)
    dtc_res = bias_analyzer.compute_subgroup_metrics(y_preds,
                                                     save_results=False,
                                                     result_filename=None,
                                                     save_dir_path=None)
    bias_metrics_df = pd.DataFrame(dtc_res)

    metrics_df = pd.concat([variance_metrics_df, bias_metrics_df])
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df['Model_Seed'] = model_seed
    metrics_df['Model_Name'] = base_model_name

    if save_results:
        # Save metrics
        result_filename = f'Metrics_{dataset_name}_{base_model_name}'
        save_metrics_to_file(metrics_df, result_filename, save_results_dir_path)

    return metrics_df


def run_metrics_computation_with_config(dataset: BaseDataset, config, models_config: dict, save_results_dir_path: str,
                                        run_seed: int = None, debug_mode: bool = False) -> dict:
    """
    Find variance and statistical bias metrics for each model in models_config.
    Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        Dataset object that contains all needed attributes like target, features, numerical_columns etc
    config
        Object that contains test_set_fraction, bootstrap_fraction, dataset_name,
         n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    save_results_dir_path
        Location where to save result files with metrics
    run_seed
        [Optional] Base seed for this run
    debug_mode
        [Optional] Enable or disable extra logs

    """
    if run_seed is None:
        run_seed = random.randint(1, 1000)
    # Create a directory for results if not exists
    os.makedirs(save_results_dir_path, exist_ok=True)
    # Parse config and execute the main run_metrics_computation function
    return run_metrics_computation(dataset=dataset,
                                   test_set_fraction=config.test_set_fraction,
                                   bootstrap_fraction=config.bootstrap_fraction,
                                   dataset_name=config.dataset_name,
                                   models_config=models_config,
                                   n_estimators=config.n_estimators,
                                   sensitive_attributes_dct=config.sensitive_attributes_dct,
                                   model_setting=config.model_setting,
                                   model_seed=run_seed,
                                   save_results_dir_path=save_results_dir_path,
                                   save_results=True,
                                   debug_mode=debug_mode)


def run_metrics_computation(dataset: BaseDataset, test_set_fraction: float, bootstrap_fraction: float, dataset_name: str,
                            models_config: dict, n_estimators: int, sensitive_attributes_dct: dict,
                            model_setting: str = ModelSetting.BATCH.value, model_seed: int = None,
                            save_results: bool = True, save_results_dir_path: str = None, debug_mode: bool = False) -> dict:
    """
    Find variance and statistical bias metrics for each model in models_config.
    Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        Dataset object that contains all needed attributes like target, features, numerical_columns etc.
    test_set_fraction
        Fraction of the whole dataset in range [0.0 - 1.0] to create a test set
    bootstrap_fraction
        Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap
    dataset_name
        Dataset name to name a result file with metrics
    models_config
        Dictionary where keys are model names, and values are initialized models
    n_estimators
        Number of estimators for bootstrap to compute subgroup variance metrics
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    model_seed
        [Optional] Model seed
    save_results
        [Optional] If to save result metrics in a file
    save_results_dir_path
        [Optional] Location where to save result files with metrics
    debug_mode
        [Optional] Enable or disable extra logs

    """
    models_metrics_dct = dict()
    num_models = len(models_config)
    for model_idx, model_name in tqdm(enumerate(models_config.keys()),
                                      total=num_models,
                                      desc="Analyze models in one run",
                                      colour="red"):
        print('#' * 30, f' [Model {model_idx + 1} / {num_models}] Analyze {model_name} ', '#' * 30)
        model_seed += 1
        try:
            base_model = models_config[model_name]
            model_metrics_df = compute_model_metrics(base_model=base_model,
                                                     n_estimators=n_estimators,
                                                     dataset=dataset,
                                                     test_set_fraction=test_set_fraction,
                                                     bootstrap_fraction=bootstrap_fraction,
                                                     sensitive_attributes_dct=sensitive_attributes_dct,
                                                     model_setting=model_setting,
                                                     model_seed=model_seed,
                                                     dataset_name=dataset_name,
                                                     base_model_name=model_name,
                                                     save_results=save_results,
                                                     save_results_dir_path=save_results_dir_path,
                                                     debug_mode=debug_mode)
            models_metrics_dct[model_name] = model_metrics_df
            if debug_mode:
                print(f'\n[{model_name}] Metrics matrix:')
                display(model_metrics_df)
        except Exception as err:
            print('#' * 20, f'ERROR with {model_name}', '#' * 20)
            traceback.print_exc()

        print('\n\n\n')

    return models_metrics_dct


def compute_metrics_multiple_runs(dataset: BaseDataset, config, models_config: dict,
                                  save_results_dir_path: str, debug_mode=False) -> dict:
    """
    Find variance and statistical bias metrics for each model in models_config. Arguments are defined as an input config object.
    Save results in `save_results_dir_path` folder.

    Return a dictionary where keys are model names, and values are metrics for multiple runs and sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.
    config
        Object that contains test_set_fraction, bootstrap_fraction, dataset_name,
         n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    save_results_dir_path
        Location where to save result files with metrics
    debug_mode
        [Optional] Enable or disable extra logs

    """
    start_datetime = datetime.now(timezone.utc)
    os.makedirs(save_results_dir_path, exist_ok=True)

    multiple_runs_metrics_dct = dict()
    for run_num, run_seed in tqdm(enumerate(config.runs_seed_lst),
                                  total=len(config.runs_seed_lst),
                                  desc="Multiple runs progress",
                                  colour="green"):
        models_metrics_dct = run_metrics_computation(dataset=dataset,
                                                     test_set_fraction=config.test_set_fraction,
                                                     bootstrap_fraction=config.bootstrap_fraction,
                                                     dataset_name=config.dataset_name,
                                                     models_config=models_config,
                                                     n_estimators=config.n_estimators,
                                                     sensitive_attributes_dct=config.sensitive_attributes_dct,
                                                     model_setting=config.model_setting,
                                                     model_seed=run_seed,
                                                     save_results=False,
                                                     debug_mode=debug_mode)

        # Concatenate with previous results and save them in an overwrite mode each time for backups
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            model_metrics_df['Run_Number'] = f'Run_{run_num + 1}'

            if multiple_runs_metrics_dct.get(model_name) is None:
                multiple_runs_metrics_dct[model_name] = model_metrics_df
            else:
                multiple_runs_metrics_dct[model_name] = pd.concat([multiple_runs_metrics_dct[model_name], model_metrics_df])

            result_filename = f'Metrics_{config.dataset_name}_{model_name}_{config.n_estimators}_Estimators_{start_datetime.strftime("%Y%m%d__%H%M%S")}.csv'
            multiple_runs_metrics_dct[model_name].to_csv(f'{save_results_dir_path}/{result_filename}', index=False, mode='w')

    return multiple_runs_metrics_dct


def compute_metrics_multiple_runs_with_db_writer(dataset: BaseDataset, config, models_config: dict,
                                                 custom_tbl_fields_dct: dict, db_writer_func, debug_mode=False) -> dict:
    """
    Find variance and statistical bias metrics for each model in models_config. Arguments are defined as an input config object.
    Save results to a database after each run appending fields and value from custom_tbl_fields_dct and using db_writer_func.

    Return a dictionary where keys are model names, and values are metrics for multiple runs and sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.
    config
        Object that contains test_set_fraction, bootstrap_fraction, dataset_name,
         n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    custom_tbl_fields_dct
        Dictionary where keys are column names and values to add to inserted metrics during saving results to a database
    db_writer_func
        Python function object has one argument (run_models_metrics_df) and save this metrics df to a target database
    debug_mode
        [Optional] Enable or disable extra logs

    """
    multiple_runs_metrics_dct = dict()
    for run_num, run_seed in tqdm(enumerate(config.runs_seed_lst),
                                  total=len(config.runs_seed_lst),
                                  desc="Multiple runs progress",
                                  colour="green"):
        run_models_metrics_df = pd.DataFrame()
        models_metrics_dct = run_metrics_computation(dataset=dataset,
                                                     test_set_fraction=config.test_set_fraction,
                                                     bootstrap_fraction=config.bootstrap_fraction,
                                                     dataset_name=config.dataset_name,
                                                     models_config=models_config,
                                                     n_estimators=config.n_estimators,
                                                     sensitive_attributes_dct=config.sensitive_attributes_dct,
                                                     model_setting=config.model_setting,
                                                     model_seed=run_seed,
                                                     save_results=False,
                                                     debug_mode=debug_mode)

        # Concatenate current run metrics with previous results and create melted_model_metrics_df to save it in a database
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            model_metrics_df['Run_Number'] = f'Run_{run_num + 1}'
            model_metrics_df['Dataset_Name'] = config.dataset_name
            model_metrics_df['Num_Estimators'] = config.n_estimators

            model_metrics_df_copy = model_metrics_df.copy(deep=True) # Version copy for multiple_runs_metrics_dct
            # Append current run metrics to multiple_runs_metrics_dct
            if multiple_runs_metrics_dct.get(model_name) is None:
                multiple_runs_metrics_dct[model_name] = model_metrics_df_copy
            else:
                multiple_runs_metrics_dct[model_name] = pd.concat([multiple_runs_metrics_dct[model_name], model_metrics_df_copy])

            # Extend df with technical columns
            model_metrics_df['Model_Name'] = model_name
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

    return multiple_runs_metrics_dct
