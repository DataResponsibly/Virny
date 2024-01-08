import pandas as pd
from datetime import datetime, timezone

from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.user_interfaces.multiple_models_api import run_metrics_computation


def compute_metrics_with_db_writer(dataset: BaseFlowDataset, config, models_config: dict,
                                   custom_tbl_fields_dct: dict, db_writer_func,
                                   postprocessor=None, notebook_logs_stdout: bool = False, verbose: int = 0) -> dict:
    """
    Compute stability and accuracy metrics for each model in models_config. Arguments are defined as an input config object.
    Save results to a database after each run appending fields and value from custom_tbl_fields_dct and using db_writer_func.

    Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.
    config
        Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    custom_tbl_fields_dct
        Dictionary where keys are column names and values to add to inserted metrics during saving results to a database
    db_writer_func
        Python function object has one argument (run_models_metrics_df) and save this metrics df to a target database
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

    # Check if a type of postprocessing_sensitive_attribute is not NoneType.
    # In other words, check if postprocessing_sensitive_attribute is defined in a config yaml.
    postprocessing_sensitive_attribute = config.postprocessing_sensitive_attribute \
        if type(config.postprocessing_sensitive_attribute) != type(None) else None

    multiple_models_metrics_dct = dict()
    run_models_metrics_df = pd.DataFrame()
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

    # Concatenate current run metrics with previous results and
    # create melted_model_metrics_df to save it in a database
    for model_name in models_metrics_dct.keys():
        model_metrics_df = models_metrics_dct[model_name]
        model_metrics_df['Dataset_Name'] = config.dataset_name
        model_metrics_df['Num_Estimators'] = config.n_estimators

        model_metrics_df_copy = model_metrics_df.copy(deep=True)  # Version copy for multiple_models_metrics_dct
        # Append current run metrics to multiple_models_metrics_dct
        if multiple_models_metrics_dct.get(model_name) is None:
            multiple_models_metrics_dct[model_name] = model_metrics_df_copy
        else:
            multiple_models_metrics_dct[model_name] = pd.concat(
                [multiple_models_metrics_dct[model_name], model_metrics_df_copy])

        # Extend df with technical columns
        model_metrics_df['Tag'] = 'OK'
        model_metrics_df['Record_Create_Date_Time'] = datetime.now(timezone.utc)

        for column, value in custom_tbl_fields_dct.items():
            model_metrics_df[column] = value

        subgroup_names = [col for col in model_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
        melted_model_metrics_df = model_metrics_df.melt(
            id_vars=[col for col in model_metrics_df.columns if col not in subgroup_names],
            value_vars=subgroup_names,
            var_name="Subgroup",
            value_name="Metric_Value")
        run_models_metrics_df = pd.concat([run_models_metrics_df, melted_model_metrics_df])

    # Save results for this run in a database
    db_writer_func(run_models_metrics_df)

    return multiple_models_metrics_dct
