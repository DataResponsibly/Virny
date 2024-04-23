import os
import re

from datetime import datetime, timezone

from virny.configs.constants import *


def validate_config(config_obj):
    """
    Validate parameters types and values in config yaml file.

    Extra details:
    * config_obj.model_setting is an optional argument that defines a type of models to use
      to compute fairness and stability metrics. Default: 'batch'.

    * config_obj.computation_mode is an optional argument that defines a non-default mode for metrics computation.
      Currently, only 'error_analysis' mode is supported.

    Parameters
    ----------
    config_obj
        Object with parameters defined in a yaml file

    """
    # ============================================================================================================
    # Required parameters
    # ============================================================================================================
    if not isinstance(config_obj.dataset_name, str):
        raise ValueError('dataset_name must be string')

    if not isinstance(config_obj.bootstrap_fraction, float) \
            or config_obj.bootstrap_fraction < 0.0 \
            or config_obj.bootstrap_fraction > 1.0:
        raise ValueError('bootstrap_fraction must be float in [0.0, 1.0] range')

    if not isinstance(config_obj.n_estimators, int) or config_obj.n_estimators <= 1:
        raise ValueError('n_estimators must be integer greater than 1')

    if not isinstance(config_obj.sensitive_attributes_dct, dict):
        raise ValueError('sensitive_attributes_dct must be python dictionary')

    if isinstance(config_obj.sensitive_attributes_dct, dict):
        intersectional_attrs = [attr for attr in config_obj.sensitive_attributes_dct.keys()
                                if INTERSECTION_SIGN in attr]
        for intersectional_attr in intersectional_attrs:
            intersectional_attr = intersectional_attr.strip()
            attrs = intersectional_attr.split(INTERSECTION_SIGN)
            attrs = [attr.strip() for attr in attrs]
            if len(attrs) != intersectional_attr.count(INTERSECTION_SIGN) + 1:
                raise ValueError(f"Incorrect format for an intersectional attribute name -- {intersectional_attr}."
                                 f"Intersectional signs must be between all attributes in this intersectional attribute.")

            for attr in attrs:
                if attr not in config_obj.sensitive_attributes_dct.keys():
                    raise ValueError('Intersectional attributes in sensitive_attributes_dct must contain '
                                     'single sensitive attributes that also exist in sensitive_attributes_dct')

    # ============================================================================================================
    # Optional parameters
    # ============================================================================================================
    if config_obj.model_setting is not None \
            and not isinstance(config_obj.model_setting, str) \
            and config_obj.model_setting not in ModelSetting:
        raise ValueError('model_setting must be a string that is included in the ModelSetting enum. '
                         'Refer to this function documentation for more details!')

    if config_obj.computation_mode is not None \
            and not isinstance(config_obj.computation_mode, str) \
            and config_obj.computation_mode not in ComputationMode:
        raise ValueError('computation_mode must be a string that is included in the ComputationMode enum. '
                         'Refer to this function documentation for more details!')

    # ============================================================================================================
    # Default parameters
    # ============================================================================================================
    if 'postprocessing_sensitive_attribute' not in config_obj:
        config_obj.postprocessing_sensitive_attribute = None
    if 'random_state' not in config_obj:
        config_obj.random_state = None

    return True


def isfloat_regex(string):
    # We have defined a pattern for float value
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    # Find the match and convert to boolean
    return bool(re.match(pattern, string))


def str_to_float(str_var: str, var_name: str):
    if isfloat_regex(str_var):
        return float(str_var)
    else:
        raise ValueError(f"{var_name} must be a float number with a '.' separator.")


def save_metrics_to_file(metrics_df, result_filename, save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_time_str = now.strftime("%Y%m%d__%H%M%S")
    filename = f"{result_filename}_{date_time_str}.csv"
    metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)


def check_substring_in_list(val_to_check: str, allowed_lst: list):
    # Case-insensitive check if a val_to_check substring is in allowed_lst
    val_to_check = val_to_check.lower()
    for allowed_val in allowed_lst:
        if allowed_val.lower() in val_to_check:
            return True
    return False
