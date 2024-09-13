import os
import pytest
import pathlib
import numpy as np
import pandas as pd

from munch import DefaultMunch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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


def compare_metric_dfs_v2(df1, df2):
    # Check shape
    if not df1.shape == df2.shape:
        return False

    # Check column names
    if not sorted(df1.columns.tolist()) == sorted(df2.columns.tolist()):
        return False

    # Check values
    df1 = df1.sort_values(by='Metric').reset_index(drop=True)
    df2 = df2.sort_values(by='Metric').reset_index(drop=True)

    if not df1.equals(df2):
        return False

    return True


def compare_metric_dfs_with_tolerance(df1, df2, tolerance=1e-6):
    # Check shape
    if not df1.shape == df2.shape:
        return False

    # Check column names
    if not sorted(df1.columns.tolist()) == sorted(df2.columns.tolist()):
        return False

    # Check values
    df1 = df1.sort_values(by='Metric').reset_index(drop=True)
    df2 = df2.sort_values(by='Metric').reset_index(drop=True)

    categorical_cols = ['Metric', 'Model_Name', 'Model_Params']
    numerical_cols = [col for col in df1.columns if col not in categorical_cols]

    # Compare numerical columns with tolerance
    close_numerical = np.isclose(df1[numerical_cols], df2[numerical_cols], atol=tolerance).all(axis=0)

    # Compare categorical columns directly
    equal_categorical = (df1[categorical_cols] == df2[categorical_cols]).all()

    # Combine both results
    overall_equal = close_numerical.all() and equal_categorical.all()

    if not overall_equal:
        return False

    return True


def compare_metric_dfs(expected_composed_metrics_df, actual_composed_metrics_df,
                       model_name, metrics_lst, groups, alpha=0.000_001):
    for metric_name in metrics_lst:
        for group in groups:
            expected_metric_val = expected_composed_metrics_df[
                (expected_composed_metrics_df['Model_Name'] == model_name) &
                (expected_composed_metrics_df['Metric'] == metric_name)
                ][group].values[0]
            actual_metric_val = actual_composed_metrics_df[
                (actual_composed_metrics_df['Model_Name'] == model_name) &
                (actual_composed_metrics_df['Metric'] == metric_name)
                ][group].values[0]

            assert abs(expected_metric_val - actual_metric_val) < alpha, f"Assert for {metric_name} metric and {group} group"


ROOT_DIR = get_root_dir()


@pytest.fixture(scope='package')
def config_params():
    config_dct = {
        "dataset_name": 'COMPAS',
        "test_set_fraction": 0.2,
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "runs_seed_lst": [100, 200, 300, 400, 500, 600],
        "sensitive_attributes_dct": {'sex': 1, 'race': 'African-American', 'sex&race': None},
    }
    return DefaultMunch.fromDict(config_dct)


@pytest.fixture(scope='package')
def folk_emp_config_params():
    config_dct = {
        "dataset_name": 'Folktables_NY_2018_Employment',
        "test_set_fraction": 0.2,
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "num_runs": 1,
        "runs_seed_lst": [100],
        "sensitive_attributes_dct": {'SEX': '2', 'RAC1P': '2', 'SEX & RAC1P': None},
    }
    return DefaultMunch.fromDict(config_dct)


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
                                                 solver='newton-cg'),
        'XGBClassifier': XGBClassifier(learning_rate=0.1,
                                       n_estimators=200,
                                       max_depth=7),
    }


@pytest.fixture(scope='package')
def folk_employment_NY_2018_loader():
    df_path = pathlib.Path(__file__).parent.joinpath('files_for_tests').joinpath('folk_employment_NY_2018.csv')
    full_df = pd.read_csv(df_path, header=0)

    target = 'ESR'
    numerical_columns = ['AGEP']
    categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P',
                           'RELP', 'CIT', 'ANC', 'SCHL']

    full_df[categorical_columns] = full_df[categorical_columns].astype('str')
    return BaseDataLoader(full_df=full_df,
                          target=target,
                          numerical_columns=numerical_columns,
                          categorical_columns=categorical_columns)


@pytest.fixture(scope='package')
def compas_dataset_class():
    dataset_path = os.path.join(ROOT_DIR, 'virny', 'datasets', 'data', 'COMPAS.csv')
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
    dataset_path = os.path.join(ROOT_DIR, 'virny', 'datasets', 'data', 'COMPAS.csv')
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

@pytest.fixture(scope='package')
def COMPAS_y_test():
    y_test = pd.read_csv(os.path.join(ROOT_DIR, 'tests', 'files_for_tests', 'COMPAS_use_case', 'COMPAS_y_test.csv'), header=0)
    y_test = y_test.set_index("0")
    return y_test


@pytest.fixture(scope='package')
def COMPAS_RF_expected_preds():
    expected_preds = pd.read_csv(os.path.join(ROOT_DIR, 'tests', 'files_for_tests', 'COMPAS_use_case',
                                 'COMPAS_RF_expected_preds.csv'), header=0)
    expected_preds = expected_preds.set_index("0")
    return expected_preds


@pytest.fixture(scope='package')
def COMPAS_RF_bootstrap_predictions():
    models_predictions = pd.read_csv(os.path.join(ROOT_DIR, 'tests', 'files_for_tests', 'COMPAS_use_case',
                                                  'COMPAS_RF_predictions.csv'), header=0)
    models_predictions = models_predictions.reset_index(drop=True)
    models_predictions_dct = dict()
    for col in models_predictions.columns:
        models_predictions_dct[int(col)] = models_predictions[col].to_numpy()

    return models_predictions_dct


@pytest.fixture(scope='package')
def COMPAS_RF_expected_metrics():
    return pd.read_csv(os.path.join(ROOT_DIR, 'tests', 'files_for_tests', 'COMPAS_use_case',
                                    'COMPAS_RF_expected_metrics.csv'), header=0)
