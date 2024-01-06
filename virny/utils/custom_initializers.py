import os
import yaml
import pandas as pd
from munch import DefaultMunch

from virny.custom_classes.fair_inprocessing_wrapper import FairInprocessingWrapper
from virny.utils.common_helpers import validate_config
from virny.configs.constants import INTERSECTION_SIGN


def create_config_obj(config_yaml_path: str):
    """
    Return a config object created based on a config yaml file.

    Parameters
    ----------
    config_yaml_path
        Path to a config yaml file

    """
    with open(config_yaml_path) as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)

    config_obj = DefaultMunch.fromDict(config_dct)
    validate_config(config_obj)

    # Fix formatting
    sensitive_attributes_dct_keys = list(config_obj.sensitive_attributes_dct.keys())
    for attr in sensitive_attributes_dct_keys:
        if INTERSECTION_SIGN in attr:
            attrs = attr.strip().split(INTERSECTION_SIGN)
            cleaned_attr = INTERSECTION_SIGN.join(attr.strip() for attr in attrs)
            if cleaned_attr != attr:
                config_obj.sensitive_attributes_dct[cleaned_attr] = config_obj.sensitive_attributes_dct.pop(attr)

    return config_obj


def create_base_model_wrapper(base_model, inprocessing_sensitive_attribute):
    """
    Check if a model is from aif360.algorithms.inprocessing package.
    If yes, create a wrapper to align with sklearn models API.
    Otherwise, return the same base_model.

    Parameters
    ----------
    base_model
        An initialized model to profile.
    inprocessing_sensitive_attribute
        Sensitive attribute name to use in the fairness in-processing intervention.

    """
    model_class = getattr(base_model, '__module__', None)
    if model_class is None:
        return base_model

    model_source_package = '.'.join(model_class.split('.')[:3])
    if model_source_package == 'aif360.algorithms.inprocessing':
        if inprocessing_sensitive_attribute is None:
            raise ValueError('Sensitive attribute for inprocessing is not defined. '
                             'Please, set inprocessing_sensitive_attribute argument in the metric computation config.')

        new_base_model = FairInprocessingWrapper(inprocessor=base_model,
                                                 sensitive_attr_for_intervention=inprocessing_sensitive_attribute)
        return new_base_model

    return base_model


def read_model_metric_dfs(metrics_path, model_names):
    # Read models metrics dfs
    metrics_filenames = [filename for filename in os.listdir(metrics_path)]
    models_metrics_dct = dict()
    for model_name in model_names:
        for filename in metrics_filenames:
            if model_name in filename:
                models_metrics_dct[model_name] = pd.read_csv(f'{metrics_path}/{filename}')
                break

    return models_metrics_dct


def create_models_metrics_dct_from_database_df(model_metric_dfs):
    """
    Create a models_metrics_dct from a metrics dataframe written to a database.
    Return the models_metrics_dct where a key is a model name and a value is a metrics dataframe for the model.

    Parameters
    ----------
    model_metric_dfs
        Path to a config yaml file

    """
    # Create columns based on values in the Subgroup column
    pivoted_model_metric_dfs = model_metric_dfs.pivot(columns='Subgroup', values='Metric_Value',
                                                      index=[col for col in model_metric_dfs.columns
                                                             if col not in ('Subgroup', 'Metric_Value')]).reset_index()
    pivoted_model_metric_dfs = pivoted_model_metric_dfs.rename_axis(None, axis=1)

    # Create a dict of metrics for each model
    models_metrics_dct = dict()
    for model_name in pivoted_model_metric_dfs['Model_Name'].unique():
        models_metrics_dct[model_name] = pivoted_model_metric_dfs[pivoted_model_metric_dfs['Model_Name'] == model_name]

    return models_metrics_dct


def create_models_config_from_tuned_params_df(models_config_for_tuning: dict, tuned_params_df_path: str):
    models_tuned_params_df = pd.read_csv(tuned_params_df_path)
    experiment_models_config = dict()
    for model_name, model_params in models_config_for_tuning.items():
        base_model = create_tuned_base_model(model_params['model'], model_name, models_tuned_params_df)
        experiment_models_config[model_name] = base_model

    return experiment_models_config


def create_tuned_base_model(init_model, model_name, models_tuned_params_df):
    model_params = eval(models_tuned_params_df.loc[models_tuned_params_df['Model_Name'] == model_name,
                                                   'Model_Best_Params'].iloc[0])
    return init_model.set_params(**model_params)
