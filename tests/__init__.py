import os
import pytest
import pandas as pd

from munch import DefaultMunch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from virny.datasets.base import BaseDataLoader


def get_root_dir():
    # Configure a location of root dir
    cur_folder_name = os.getcwd().split('/')[-1]
    if cur_folder_name == "tests":
        root_dir = os.path.join('..')
    else:
        # For the root repo path
        if os.path.exists(os.path.join('virny', 'datasets')):
            root_dir = os.getcwd()
        else:
            root_dir = os.path.join('..', '..')
            
    return root_dir


ROOT_DIR = get_root_dir()


@pytest.fixture(scope='package')
def config_params():
    config_dct = {
        "dataset_name": 'COMPAS',
        "test_set_fraction": 0.2,
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "runs_seed_lst": [100, 200, 300, 400, 500, 600],
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian', 'sex&race': None},
    }
    return DefaultMunch.fromDict(config_dct)\


@pytest.fixture(scope='package')
def models_config():
    return {
        'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                         max_depth=20,
                                                         max_features=0.6,
                                                         min_samples_split=0.1),
        'LogisticRegression': LogisticRegression(C=1,
                                                 max_iter=50,
                                                 penalty='l2',
                                                 solver='newton-cg')
    }


@pytest.fixture(scope='package')
def compas_dataset_class():
    dataset_path = os.path.join(ROOT_DIR, 'virny', 'datasets', 'COMPAS.csv')
    df = pd.read_csv(dataset_path)

    int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                   'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
    int_columns_dct = {col: "int" for col in int_columns}
    df = df.astype(int_columns_dct)

    target = 'recidivism'
    numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                           'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']

    return BaseDataLoader(full_df=df,
                          target=target,
                          numerical_columns=numerical_columns,
                          categorical_columns=categorical_columns)


@pytest.fixture(scope='package')
def compas_without_sensitive_attrs_dataset_class():
    dataset_path = os.path.join(ROOT_DIR, 'virny', 'datasets', 'COMPAS.csv')
    df = pd.read_csv(dataset_path)

    int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                   'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
    int_columns_dct = {col: "int" for col in int_columns}
    df = df.astype(int_columns_dct)

    target = 'recidivism'
    numerical_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    categorical_columns = ['age_cat_25 - 45', 'age_cat_Greater than 45',
                           'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M']

    return BaseDataLoader(full_df=df,
                          target=target,
                          numerical_columns=numerical_columns,
                          categorical_columns=categorical_columns)
