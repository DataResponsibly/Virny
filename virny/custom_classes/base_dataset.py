import pandas as pd


class BaseDataset:
    """
    Base dataset class that is used as input for user interfaces.
    Inherit from it to create a dataset class for your dataset file.

    Parameters
    ----------
    pandas_df
        Full dataset in a pandas dataframe format
    features
        List of feature column names
    target
        Name of the target column name
    numerical_columns
        List of numerical column names
    categorical_columns
        List of categorical column names
    X_data
        [Optional] Dataframe of features
    y_data
        [Optional] Dataframe with a target column
    columns_with_nulls
        [Optional] List of column names that contains nulls

    """
    def __init__(self, pandas_df: pd.DataFrame, features: list, target: str, numerical_columns: list,
                 categorical_columns: list, X_data: pd.DataFrame = None, y_data: pd.DataFrame = None,
                 columns_with_nulls: list = None):
        self.dataset = pandas_df
        self.target = target
        self.features = features
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.X_data = self.dataset[features] if X_data is None else X_data
        self.y_data = self.dataset[target] if y_data is None else y_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list() \
            if columns_with_nulls is None else columns_with_nulls
