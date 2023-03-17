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

        # By default, dataset splits should be None for BaseDataset.
        # If you want to define them, please use CustomSplitsDataset.
        self.X_train_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_test = None


class CustomSplitsDataset(BaseDataset):
    """
    Dataset class with custom train and test splits that is used as input for user interfaces.
    Inherit from it if you want to use your custom dataset train and test splits.

    Parameters
    ----------
    full_features_df
        Full train + test dataset of features without the target column. It is used for creating test groups.
    X_train_val
        Train dataframe of features
    X_test
        Test dataframe of features
    y_train_val
        Train dataframe with a target column
    y_test
        Test dataframe with a target column
    target
        Name of the target column name
    numerical_columns
        List of numerical column names
    categorical_columns
        List of categorical column names

    """
    def __init__(self, full_features_df: pd.DataFrame, X_train_val: pd.DataFrame, X_test: pd.DataFrame,
                 y_train_val: pd.DataFrame, y_test: pd.DataFrame,
                 target: str, numerical_columns: list, categorical_columns: list):
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=full_features_df,
            X_data=pd.DataFrame(),
            y_data=pd.DataFrame(),
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            columns_with_nulls=[]
        )

        self.X_train_val = X_train_val
        self.X_test = X_test
        self.y_train_val = y_train_val
        self.y_test = y_test
