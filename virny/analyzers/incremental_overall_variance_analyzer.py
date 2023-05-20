import numpy as np
import pandas as pd

from virny.custom_classes.incremental_pandas_dataset import IncrementalPandasDataset
from virny.analyzers.abstract_overall_variance_analyzer import AbstractOverallVarianceAnalyzer


class IncrementalOverallVarianceAnalyzer(AbstractOverallVarianceAnalyzer):
    """
    Analyzer to compute subgroup variance metrics for incremental learning models.

    Parameters
    ----------
    base_model
        Base model for stability measuring
    base_model_name
        Model name like 'HoeffdingTreeClassifier' or 'LogisticRegression'
    bootstrap_fraction
        [0-1], fraction from train_pd_dataset for fitting an ensemble of base models
    X_train
        Processed features train set
    y_train
        Targets train set
    X_test
        Processed features test set
    y_test
        Targets test set
    target_column
        Name of the target column
    dataset_name
        Name of dataset, used for correct results naming
    n_estimators
        Number of estimators in ensemble to measure base_model stability
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
         As for now, 0, 1, 2 levels are supported.

    """
    def __init__(self, base_model, base_model_name: str, bootstrap_fraction: float,
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 target_column: str, dataset_name: str, n_estimators: int, verbose: int = 0):
        super().__init__(base_model=base_model,
                         base_model_name=base_model_name,
                         bootstrap_fraction=bootstrap_fraction,
                         X_train=X_train,
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         dataset_name=dataset_name,
                         n_estimators=n_estimators,
                         verbose=verbose)
        self.target_column = target_column
        self.dataset_reader = IncrementalPandasDataset

        # Create converters for the train set to apply them for train incremental datasets
        train_df_for_types = X_train.astype('object')
        train_converters = {str(col): type(train_df_for_types.loc[train_df_for_types.index[0], col])
                                 for col in train_df_for_types}
        train_converters[self.target_column] = type(y_train.astype('object')[y_train.index[0]])
        self.train_converters = train_converters

    def _fit_model(self, classifier, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit an incremental classifier that is an instance of self.base_model
        """
        train_df = pd.DataFrame(X_train, columns=[key for key in self.train_converters.keys()
                                                  if key != self.target_column])
        train_df[self.target_column] = y_train
        train_dataset = self.dataset_reader(pd_dataset=train_df, target=self.target_column, converters=self.train_converters)
        for x, y_true in train_dataset:
            classifier.learn_one(x=x, y=y_true)

        return classifier

    def _batch_predict(self, classifier, X_test: pd.DataFrame):
        """
        Predict with the incremental classifier for X_test set.
        Return predictions.
        """
        predictions = []
        test_df_for_types = X_test.astype('object')
        converters = {col: type(test_df_for_types.loc[test_df_for_types.index[0], col]) for col in test_df_for_types}
        test_dataset = self.dataset_reader(pd_dataset=X_test, target=None, converters=converters)
        for x, _ in test_dataset:
            y_pred = classifier.predict_one(x)
            predictions.append(y_pred)

        return predictions

    def _batch_predict_proba(self, classifier, X_test: pd.DataFrame):
        """
        Predict with the incremental classifier for X_test set.
        Return predicted probabilities for each class for each test point.
        """
        # Create converters for the test set to apply them for an incremental dataset
        test_df_for_types = X_test.astype('object')
        converters = {col: type(test_df_for_types.loc[test_df_for_types.index[0], col]) for col in test_df_for_types}

        predictions = []
        test_dataset = self.dataset_reader(pd_dataset=X_test, target=None, converters=converters)
        for x, _ in test_dataset:
            predict_proba = classifier.predict_proba_one(x)
            y_pred = predict_proba[0]
            predictions.append(y_pred)

        return predictions
