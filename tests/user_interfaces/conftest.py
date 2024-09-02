import pytest
import pathlib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.utils.custom_initializers import create_config_obj
from virny.datasets import LawSchoolDataset


@pytest.fixture(scope="function")
def common_seed():
    return 42


@pytest.fixture(scope="function")
def law_school_dataset_1k_params(common_seed):
    # Define init variables
    dataset_split_seed = 42
    test_set_fraction = 0.2
    data_loader = LawSchoolDataset(subsample_size=1_000, subsample_seed=common_seed)
    config = create_config_obj(config_yaml_path=str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'configs', 'law_school_config.yaml')))
    models_config = {
        'LogisticRegression': LogisticRegression(penalty='l2',
                                                 C=0.1,
                                                 max_iter=250),
    }
    save_results_dir_path = str(pathlib.Path(__file__).parent.parent.joinpath('results', 'law_school_dataset_1k'))

    # Preprocess the dataset
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse_output=False), data_loader.categorical_columns),
        ('numerical_features', StandardScaler(), data_loader.numerical_columns),
    ])
    base_flow_dataset = preprocess_dataset(data_loader=data_loader,
                                           column_transformer=column_transformer,
                                           sensitive_attributes_dct=config.sensitive_attributes_dct,
                                           test_set_fraction=test_set_fraction,
                                           dataset_split_seed=dataset_split_seed)

    return base_flow_dataset, config, models_config, save_results_dir_path


@pytest.fixture(scope="function")
def law_school_dataset_20k_params(common_seed):
    # Define init variables
    dataset_split_seed = 42
    test_set_fraction = 0.2
    data_loader = LawSchoolDataset()
    config = create_config_obj(config_yaml_path=str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'configs', 'law_school_config.yaml')))
    models_config = {
        'LogisticRegression': LogisticRegression(penalty='l2',
                                                 C=0.1,
                                                 max_iter=250),
        'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                         max_depth=20,
                                                         max_features=0.6,
                                                         min_samples_split=0.1),
    }
    save_results_dir_path = str(pathlib.Path(__file__).parent.parent.joinpath('results', 'law_school_dataset_20k'))

    # Preprocess the dataset
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse_output=False), data_loader.categorical_columns),
        ('numerical_features', StandardScaler(), data_loader.numerical_columns),
    ])
    base_flow_dataset = preprocess_dataset(data_loader=data_loader,
                                           column_transformer=column_transformer,
                                           sensitive_attributes_dct=config.sensitive_attributes_dct,
                                           test_set_fraction=test_set_fraction,
                                           dataset_split_seed=dataset_split_seed)

    return base_flow_dataset, config, models_config, save_results_dir_path
