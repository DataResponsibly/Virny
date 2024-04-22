import pandas as pd


class BaseFlowDataset:
    """
    Dataset class with custom train and test splits that is used as input for metrics computation interfaces.
    Create your dataset class based on this one to use it for metrics computation interfaces.

    Parameters
    ----------
    init_features_df
        Full train + test non-preprocessed dataset of features without the target column.
         It is used for creating test groups.
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
    ordered_categories_dct
        [Optional] Dictionary where keys are ordinal column names,
         and values are lists of ordered categories for these columns

    """

    def __init__(self, init_features_df: pd.DataFrame, X_train_val: pd.DataFrame, X_test: pd.DataFrame,
                 y_train_val: pd.DataFrame, y_test: pd.DataFrame, target: str,
                 numerical_columns: list, categorical_columns: list, ordered_categories_dct: dict = dict()):
        # Validate input sets
        if not isinstance(init_features_df, pd.DataFrame) or not isinstance(X_train_val, pd.DataFrame) \
                or not isinstance(X_test, pd.DataFrame):
            raise ValueError("Input feature sets must be in a pd.DataFrame format")

        assert X_test.index.isin(init_features_df.index).all(), \
            ("Not all indexes of X_test are present in init_features_df. "
             "It is important to correctly compute metrics for protected groups in the test set.")
        assert y_test.index.isin(init_features_df.index).all(), \
            ("Not all indexes of y_test are present in init_features_df. "
             "It is important to correctly compute metrics for protected groups in the test set.")

        assert X_train_val.index.equals(y_train_val.index) is True, \
            "Indexes of X_train_val and y_train_val are different"
        assert X_test.index.equals(y_test.index) is True, \
            "Indexes of X_test and y_test should be the same to correctly compute metrics for protected groups in the test set"

        # Define parameters
        self.init_features_df = init_features_df
        self.X_train_val = X_train_val
        self.X_test = X_test
        self.y_train_val = y_train_val
        self.y_test = y_test

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.features = numerical_columns + categorical_columns
        self.ordered_categories_dct = ordered_categories_dct
        self.target = target
