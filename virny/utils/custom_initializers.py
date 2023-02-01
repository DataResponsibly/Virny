import os
import yaml
import pandas as pd
from munch import DefaultMunch

from virny.custom_classes.generic_pipeline import GenericPipeline
from virny.utils.common_helpers import validate_config


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

    return config_obj


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


def create_models_config_from_tuned_params_df(models_config_for_tuning, models_tuned_params_df):
    experiment_models_config = dict()
    for model_idx in range(len(models_config_for_tuning)):
        model_name = models_config_for_tuning[model_idx]["model_name"]
        base_model = create_tuned_base_model(models_config_for_tuning[model_idx]['model'], model_name, models_tuned_params_df)
        experiment_models_config[model_name] = base_model

    return experiment_models_config


def create_base_pipeline(dataset, sensitive_attributes_dct, model_seed, test_set_fraction):
    base_pipeline = GenericPipeline(dataset, sensitive_attributes_dct)
    _ = base_pipeline.create_preprocessed_train_test_split(dataset, test_set_fraction, seed=model_seed)

    return base_pipeline


def create_tuned_base_model(init_model, model_name, models_tuned_params_df):
    model_params = eval(models_tuned_params_df.loc[models_tuned_params_df['Model_Name'] == model_name,
                                                   'Model_Best_Params'].iloc[0])
    return init_model.set_params(**model_params)
