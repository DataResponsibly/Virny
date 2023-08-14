from sklearn.model_selection import train_test_split
from virny.datasets import ACSEmploymentDataset
from virny.utils.protected_groups_partitioning import check_sensitive_attrs_in_columns, create_test_protected_groups

from tests import config_params, folk_emp_config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class


def test_check_sensitive_attrs_in_columns_true(config_params):
    df_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                  'race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                  'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
    actual = check_sensitive_attrs_in_columns(df_columns,
                                              config_params.sensitive_attributes_dct)
    assert actual == True


def test_check_sensitive_attrs_in_columns_false(config_params):
    df_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                  'race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                  'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M']
    actual = check_sensitive_attrs_in_columns(df_columns,
                                              config_params.sensitive_attributes_dct)
    assert actual == False


def test_create_test_protected_groups_true(compas_dataset_class, config_params):
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(compas_dataset_class.X_data,
                                                        compas_dataset_class.y_data,
                                                        test_size=config_params.test_set_fraction,
                                                        random_state=seed)
    actual_test_protected_groups = create_test_protected_groups(X_test, compas_dataset_class.full_df,
                                                                config_params.sensitive_attributes_dct)

    assert len(actual_test_protected_groups) == len(config_params.sensitive_attributes_dct.keys()) * 2

    assert actual_test_protected_groups['sex_priv'].shape[0] == 211
    assert actual_test_protected_groups['sex_dis'].shape[0] == 845
    assert actual_test_protected_groups['race_priv'].shape[0] == 414
    assert actual_test_protected_groups['race_dis'].shape[0] == 642
    assert actual_test_protected_groups['sex&race_priv'].shape[0] == 526
    assert actual_test_protected_groups['sex&race_dis'].shape[0] == 530


def test_create_test_protected_groups_true_without_sensitive_attrs(compas_without_sensitive_attrs_dataset_class, config_params):
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(compas_without_sensitive_attrs_dataset_class.X_data,
                                                        compas_without_sensitive_attrs_dataset_class.y_data,
                                                        test_size=config_params.test_set_fraction,
                                                        random_state=seed)
    actual_test_protected_groups = create_test_protected_groups(X_test, compas_without_sensitive_attrs_dataset_class.full_df,
                                                                config_params.sensitive_attributes_dct)

    assert len(actual_test_protected_groups) == len(config_params.sensitive_attributes_dct.keys()) * 2

    assert actual_test_protected_groups['sex_priv'].shape[0] == 211
    assert actual_test_protected_groups['sex_dis'].shape[0] == 845
    assert actual_test_protected_groups['race_priv'].shape[0] == 414
    assert actual_test_protected_groups['race_dis'].shape[0] == 642
    assert actual_test_protected_groups['sex&race_priv'].shape[0] == 526
    assert actual_test_protected_groups['sex&race_dis'].shape[0] == 530


def test_create_test_protected_groups_true2(compas_without_sensitive_attrs_dataset_class, config_params):
    new_sensitive_attributes_dct = {'sex': 1, 'race': 'African-American'}

    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(compas_without_sensitive_attrs_dataset_class.X_data,
                                                        compas_without_sensitive_attrs_dataset_class.y_data,
                                                        test_size=config_params.test_set_fraction,
                                                        random_state=seed)
    actual_test_protected_groups = create_test_protected_groups(X_test,
                                                                compas_without_sensitive_attrs_dataset_class.full_df,
                                                                new_sensitive_attributes_dct)

    assert len(actual_test_protected_groups) == len(new_sensitive_attributes_dct.keys()) * 2

    assert actual_test_protected_groups['sex_priv'].shape[0] == 211
    assert actual_test_protected_groups['sex_dis'].shape[0] == 845
    assert actual_test_protected_groups['race_priv'].shape[0] == 414
    assert actual_test_protected_groups['race_dis'].shape[0] == 642
