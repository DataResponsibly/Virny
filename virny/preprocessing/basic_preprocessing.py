import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from virny.datasets.data_loaders import BaseDataLoader
from virny.custom_classes.base_dataset import BaseFlowDataset


def preprocess_dataset(data_loader: BaseDataLoader, column_transformer: ColumnTransformer,
                       test_set_fraction: float, dataset_split_seed: int) -> BaseFlowDataset:
    """
    Preprocess an input dataset using sklearn ColumnTransformer. Split the dataset on train and test using test_set_fraction.
     Create an instance of BaseFlowDataset.

    Parameters
    ----------
    data_loader
        Instance of BaseDataLoader that contains a target, numerical, and categorical columns.
    column_transformer
        Instance of sklearn ColumnTransformer to preprocess categorical and numerical columns.
    test_set_fraction
        Fraction from 0 to 1. Used to split the input dataset on the train and test sets.
    dataset_split_seed
        Seed for dataset splitting.

    Return
    ----------
    An instance of BaseFlowDataset.

    """
    if test_set_fraction < 0.0 or test_set_fraction > 1.0:
        raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=dataset_split_seed)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    X_train_features = column_transformer.fit_transform(X_train_val)
    X_test_features = column_transformer.transform(X_test)

    return BaseFlowDataset(init_features_df=data_loader.full_df.drop(data_loader.target, axis=1, errors='ignore'),
                           X_train_val=X_train_features,
                           X_test=X_test_features,
                           y_train_val=y_train_val,
                           y_test=y_test,
                           target=data_loader.target,
                           numerical_columns=data_loader.numerical_columns,
                           categorical_columns=data_loader.categorical_columns)


def get_dummies(data: pd.DataFrame, categorical_columns: list, numerical_columns: list):
    """
    Return a dataset made by one-hot encoding for categorical columns and concatenate with numerical columns.

    Parameters
    ----------
    data
        Dataframe for one-hot encoding
    categorical_columns
        List of categorical column names
    numerical_columns
        List of numerical column names

    """
    feature_df = pd.get_dummies(data[categorical_columns], columns=categorical_columns)
    for col in numerical_columns:
        if col in data.columns:
            feature_df[col] = data[col]
    return feature_df


def make_features_dfs(X_train: pd.DataFrame, X_test: pd.DataFrame, dataset: BaseFlowDataset):
    """
    Return preprocessed train and test feature dataframes after one-hot encoding and standard scaling.

    Parameters
    ----------
    X_train

    X_test

    dataset

    """
    X_train_features = get_dummies(X_train, dataset.categorical_columns, dataset.numerical_columns)
    X_test_features = get_dummies(X_test, dataset.categorical_columns, dataset.numerical_columns)

    # Align columns
    features_columns = list(set(X_train_features.columns) & set(X_test_features.columns))
    X_train_features = X_train_features[features_columns]
    X_test_features = X_test_features[features_columns]

    scaler = StandardScaler()
    X_train_features[dataset.numerical_columns] = scaler.fit_transform(X_train_features[dataset.numerical_columns])
    X_test_features[dataset.numerical_columns] = scaler.transform(X_test_features[dataset.numerical_columns])

    return X_train_features, X_test_features
