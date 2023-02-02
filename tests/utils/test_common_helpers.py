import pytest
from munch import DefaultMunch

from virny.utils.common_helpers import check_sensitive_attrs_in_columns, validate_config


@pytest.fixture(scope='module')
def config_params():
    config_dct = {
        "dataset_name": 'COMPAS',
        "test_set_fraction": 0.2,
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "runs_seed_lst": [100, 200, 300, 400, 500, 600],
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian', 'sex&race': None},
    }
    return DefaultMunch.fromDict(config_dct)


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


def test_validate_config_true1(config_params):
    actual = validate_config(config_params)
    assert actual == True


def test_validate_config_true2():
    config_dct = {
        "dataset_name": 'COMPAS',
        "test_set_fraction": 0.2,
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian'},
    }
    config = DefaultMunch.fromDict(config_dct)

    actual = validate_config(config)
    assert actual == True
