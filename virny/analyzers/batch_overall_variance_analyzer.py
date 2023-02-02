import pandas as pd

from virny.analyzers.abstract_overall_variance_analyzer import AbstractOverallVarianceAnalyzer


class BatchOverallVarianceAnalyzer(AbstractOverallVarianceAnalyzer):
    """
    Analyzer to compute subgroup variance metrics for batch learning models.

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

    """
    def __init__(self, base_model, base_model_name: str, bootstrap_fraction: float,
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 target_column: str, dataset_name: str, n_estimators: int):
        super().__init__(base_model, base_model_name, bootstrap_fraction,
                         X_train, y_train, X_test, y_test, dataset_name, n_estimators)
        self.target_column = target_column

    def _fit_model(self, classifier, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fit a classifier that is an instance of self.base_model.
        """
        return classifier.fit(X_train, y_train)

    def _batch_predict(self, classifier, X_test: pd.DataFrame):
        """
        Predict with the classifier for X_test set and return predictions
        """
        return classifier.predict(X_test)

    def _batch_predict_proba(self, classifier, X_test: pd.DataFrame):
        """
        Predict with the classifier for X_test set and return probabilities for each class for each test point
        """
        return classifier.predict_proba(X_test)[:, 0]
