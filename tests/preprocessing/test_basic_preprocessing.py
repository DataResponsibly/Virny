import numpy as np
from sklearn.model_selection import train_test_split

from tests import config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class
from virny.preprocessing.basic_preprocessing import make_features_dfs


# ========================== Test make_features_dfs ==========================
def test_make_features_dfs_true1(compas_without_sensitive_attrs_dataset_class, config_params):
    dataset = compas_without_sensitive_attrs_dataset_class
    X_train, X_test, y_train, y_test = train_test_split(dataset.X_data, dataset.y_data,
                                                        test_size=config_params.test_set_fraction,
                                                        random_state=42)
    X_train_features, X_test_features = make_features_dfs(X_train, X_test, dataset)

    # Number of columns and their names are the same in train and test sets
    assert np.array_equal(X_train_features.columns, X_test_features.columns)

    # Check if values changed after normalization
    assert (X_train['juv_fel_count'].values == X_train_features['juv_fel_count'].values).any() == False
    assert (X_test['juv_fel_count'].values == X_test_features['juv_fel_count'].values).any() == False


def test_make_features_dfs_true2(compas_dataset_class, config_params):
    dataset = compas_dataset_class
    X_train, X_test, y_train, y_test = train_test_split(dataset.X_data, dataset.y_data,
                                                        test_size=config_params.test_set_fraction,
                                                        random_state=42)
    X_train_features, X_test_features = make_features_dfs(X_train, X_test, dataset)

    # Number of columns and their names are the same in train and test sets
    assert np.array_equal(X_train_features.columns, X_test_features.columns)

    # Check if values changed after normalization
    assert (X_train['juv_fel_count'].values == X_train_features['juv_fel_count'].values).any() == False
    assert (X_test['juv_fel_count'].values == X_test_features['juv_fel_count'].values).any() == False
