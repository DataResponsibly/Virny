import pandas as pd


class BaseDataLoader:
    """
    Base data loader class that helps to unify the logic for various datasets.

    Parameters
    ----------
    full_df
        Full dataset in a pandas dataframe format
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
    def __init__(self, full_df: pd.DataFrame, target: str, numerical_columns: list,
                 categorical_columns: list, X_data: pd.DataFrame = None, y_data: pd.DataFrame = None,
                 columns_with_nulls: list = None):
        self.full_df = full_df
        self.target = target
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.features = numerical_columns + categorical_columns

        self.X_data = self.full_df[self.features] if X_data is None else X_data
        self.y_data = self.full_df[self.target] if y_data is None else y_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list() \
            if columns_with_nulls is None else columns_with_nulls
