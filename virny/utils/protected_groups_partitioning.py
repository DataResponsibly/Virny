import pandas as pd

from virny.configs.constants import INTERSECTION_SIGN


def get_df_condition(df: pd.DataFrame, col: str, dis, include_dis: bool):
    if isinstance(dis, list):
        return df[col].isin(dis) if include_dis else ~df[col].isin(dis)
    else:
        return df[col] == dis if include_dis else df[col] != dis


def partition_by_group_intersectional(df, attrs, dis_values):
    """
    After a partitioning on intersectional groups, a dis group is formed based on the values in sensitive_attributes_dct,
    and a priv group includes all other records, which are not included to a dis group.

    :param df: n initial df
    :param attrs: sensitive attributes
    :param dis_values: disadvantage values for input sensitive attributes

    """
    # Construct complex df conditions
    dis_condition = get_df_condition(df, attrs[0], dis_values[0], include_dis=True)
    for idx in range(1, len(attrs)):
        dis_condition &= get_df_condition(df, attrs[idx], dis_values[idx], include_dis=True)

    priv_condition = (get_df_condition(df, attrs[0], dis_values[0], include_dis=False))
    for idx in range(1, len(attrs)):
        priv_condition |= get_df_condition(df, attrs[idx], dis_values[idx], include_dis=False)

    priv = df[priv_condition]
    dis = df[dis_condition]
    if len(priv) + len(dis) != len(df):
        raise ValueError("Error! Not a partition")

    return priv, dis


def partition_by_group_binary(df, attr, dis_value):
    dis = df[get_df_condition(df, attr, dis_value, include_dis=True)]
    priv = df[get_df_condition(df, attr, dis_value, include_dis=False)]
    if len(priv) + len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def check_sensitive_attrs_in_columns(df_columns, sensitive_attributes_dct):
    for sensitive_attr in sensitive_attributes_dct.keys():
        if INTERSECTION_SIGN not in sensitive_attr and sensitive_attr not in df_columns:
            return False
    return True


def create_test_protected_groups(X_test: pd.DataFrame, init_features_df: pd.DataFrame, sensitive_attributes_dct: dict):
    """
    Create protected groups based on a test feature set. Use a disadvantaged group as a reference group.

    Return a dictionary where keys are subgroup names, and values are X_test row indexes correspondent to this subgroup.

    Parameters
    ----------
    X_test
        Test feature set
    init_features_df
        Initial full dataset without preprocessing
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are disadvantaged values for these attributes

    """
    plain_sensitive_attributes = [attr for attr in sensitive_attributes_dct.keys() if INTERSECTION_SIGN not in attr]
    X_test_with_sensitive_attrs = init_features_df[plain_sensitive_attributes].loc[X_test.index]

    groups = dict()
    for attr in sensitive_attributes_dct.keys():
        attr = attr.strip()
        if INTERSECTION_SIGN in attr:
            single_attrs = attr.split(INTERSECTION_SIGN)
            single_attrs = [single_attr.strip() for single_attr in single_attrs]
            priv_grp_name = INTERSECTION_SIGN.join(single_attrs) + '_priv'
            dis_grp_name = INTERSECTION_SIGN.join(single_attrs) + '_dis'
            groups[priv_grp_name], groups[dis_grp_name] = \
                partition_by_group_intersectional(X_test_with_sensitive_attrs, single_attrs,
                                                  dis_values=[sensitive_attributes_dct[attr] for attr in single_attrs])
        else:
            priv_grp_name = attr + '_priv'
            dis_grp_name = attr + '_dis'
            groups[priv_grp_name], groups[dis_grp_name] = \
                partition_by_group_binary(X_test_with_sensitive_attrs, attr, sensitive_attributes_dct[attr])

        if groups[priv_grp_name].shape[0] == 0:
            raise ValueError(f"Protected group ({priv_grp_name}) from X_test is empty. "
                             f"Please check types of sensitive attributes in config or replace the sensitive attribute")
        if groups[dis_grp_name].shape[0] == 0:
            raise ValueError(f"Protected group ({dis_grp_name}) from X_test is empty. "
                             f"Please check types of sensitive attributes in config or replace the sensitive attribute")

    return groups
