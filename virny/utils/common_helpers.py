import os
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import confusion_matrix

from virny.configs.constants import INTERSECTION_SIGN


def validate_config(config_obj):
    """
    Validate parameters types and values in config yaml file.

    Parameters
    ----------
    config_obj
        Object with parameters defined in a yaml file

    """
    if not isinstance(config_obj.dataset_name, str):
        raise ValueError('dataset_name must be string')

    elif not isinstance(config_obj.test_set_fraction, float) \
            or config_obj.test_set_fraction < 0.0 \
            or config_obj.test_set_fraction > 1.0:
        raise ValueError('test_set_fraction must be float in [0.0, 1.0] range')

    elif not isinstance(config_obj.bootstrap_fraction, float) \
            or config_obj.bootstrap_fraction < 0.0 \
            or config_obj.bootstrap_fraction > 1.0:
        raise ValueError('bootstrap_fraction must be float in [0.0, 1.0] range')

    elif not isinstance(config_obj.n_estimators, int) or config_obj.n_estimators <= 1:
        raise ValueError('n_estimators must be integer greater than 1')

    elif config_obj.runs_seed_lst is not None and not isinstance(config_obj.runs_seed_lst, list):
        raise ValueError('runs_seed_lst must be python list')

    elif not isinstance(config_obj.sensitive_attributes_dct, dict):
        raise ValueError('sensitive_attributes_dct must be python dictionary')

    elif isinstance(config_obj.sensitive_attributes_dct, dict):
        for sensitive_attr in config_obj.sensitive_attributes_dct.keys():
            if sensitive_attr.count(INTERSECTION_SIGN) > 1:
                raise ValueError('sensitive_attributes_dct must contain only plain sensitive attributes or '
                                 'intersections of two sensitive attributes (not more attributes intersections)')
        intersectional_attrs = [attr for attr in config_obj.sensitive_attributes_dct.keys()
                                if INTERSECTION_SIGN in attr]
        for intersectional_attr in intersectional_attrs:
            attr1, attr2 = intersectional_attr.split(INTERSECTION_SIGN)
            if attr1 not in config_obj.sensitive_attributes_dct.keys() or \
                attr2 not in config_obj.sensitive_attributes_dct.keys():
                raise ValueError('intersectional attributes in sensitive_attributes_dct must contain '
                                 'sensitive attributes that also exist in sensitive_attributes_dct')
    return True


def reset_model_seed(model, new_seed):
    if 'random_state' in model.get_params():
        model.set_params(random_state=new_seed)
    return model


def save_metrics_to_file(metrics_df, result_filename, save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_time_str = now.strftime("%Y%m%d__%H%M%S")
    filename = f"{result_filename}_{date_time_str}.csv"
    metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)


def partition_by_group_intersectional(df, attr1, attr2, priv_value1, priv_value2):
    priv = df[(df[attr1] == priv_value1) & (df[attr2] == priv_value2)]
    dis = df[(df[attr1] != priv_value1) & (df[attr2] != priv_value2)]

    return priv, dis


def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv) + len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def check_sensitive_attrs_in_columns(df_columns, sensitive_attributes_dct):
    for sensitive_attr in sensitive_attributes_dct.keys():
        if INTERSECTION_SIGN not in sensitive_attr and sensitive_attr not in df_columns:
            return False
    return True


def create_test_protected_groups(X_test: pd.DataFrame, full_df: pd.DataFrame, sensitive_attributes_dct: dict):
    """
    Create protected groups based on a test feature set.

    Return a dictionary where keys are subgroup names, and values are X_test rows correspondent to this subgroup.

    Parameters
    ----------
    X_test
        Test feature set
    full_df
        Full dataset
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes

    """

    # Check if input sensitive attributes are in X_test.columns.
    # If no, add them only to create test groups
    if check_sensitive_attrs_in_columns(X_test.columns, sensitive_attributes_dct):
        X_test_with_sensitive_attrs = X_test
    else:
        plain_sensitive_attributes = [attr for attr in sensitive_attributes_dct.keys() if INTERSECTION_SIGN not in attr]
        cols_with_sensitive_attrs = set(list(X_test.columns) + plain_sensitive_attributes)
        X_test_with_sensitive_attrs = full_df[cols_with_sensitive_attrs].loc[X_test.index]

    groups = dict()
    for attr in sensitive_attributes_dct.keys():
        if INTERSECTION_SIGN in attr:
            if attr.count(INTERSECTION_SIGN) == 1:
                attr1, attr2 = attr.split(INTERSECTION_SIGN)
                groups[attr1 + INTERSECTION_SIGN + attr2 + '_priv'], groups[attr1 + INTERSECTION_SIGN + attr2 + '_dis'] = \
                    partition_by_group_intersectional(X_test_with_sensitive_attrs, attr1, attr2,
                                                      sensitive_attributes_dct[attr1], sensitive_attributes_dct[attr2])

                if groups[attr1 + INTERSECTION_SIGN + attr2 + '_priv'].shape[0] == 0:
                    raise ValueError(f"Protected group ({attr1 + INTERSECTION_SIGN + attr2 + '_priv'}) from X_test is empty. "
                                     f"Please check types of sensitive attributes in config, or replace the sensitive attribute, or extend test_set_fraction")
                if groups[attr1 + INTERSECTION_SIGN + attr2 + '_dis'].shape[0] == 0:
                    raise ValueError(f"Protected group ({attr1 + INTERSECTION_SIGN + attr2 + '_dis'}) from X_test is empty. "
                                     f"Please check types of sensitive attributes in config, or replace the sensitive attribute, or extend test_set_fraction")
        else:
            groups[attr + '_priv'], groups[attr + '_dis'] = \
                partition_by_group_binary(X_test_with_sensitive_attrs, attr, sensitive_attributes_dct[attr])

            if groups[attr + '_priv'].shape[0] == 0:
                raise ValueError(f"Protected group ({attr + '_priv'}) from X_test is empty. "
                                 f"Please check types of sensitive attributes in config, or replace the sensitive attribute, or extend test_set_fraction")
            if groups[attr + '_dis'].shape[0] == 0:
                raise ValueError(f"Protected group ({attr + '_dis'}) from X_test is empty. "
                                 f"Please check types of sensitive attributes in config, or replace the sensitive attribute, or extend test_set_fraction")

    return groups


def confusion_matrix_metrics(y_true, y_preds):
    metrics = {}
    TN, FP, FN, TP = confusion_matrix(y_true, y_preds).ravel()
    metrics['TPR'] = TP/(TP+FN)
    metrics['TNR'] = TN/(TN+FP)
    metrics['PPV'] = TP/(TP+FP)
    metrics['FNR'] = FN/(FN+TP)
    metrics['FPR'] = FP/(FP+TN)
    metrics['Accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    metrics['F1'] = (2*TP)/(2*TP+FP+FN)
    metrics['Selection-Rate'] = (TP+FP)/(TP+FP+TN+FN)
    metrics['Positive-Rate'] = (TP+FP)/(TP+FN)

    return metrics
