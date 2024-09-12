import inspect
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
        Model name like 'DecisionTreeClassifier' or 'LogisticRegression'
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
    random_state
        [Optional] Controls the randomness of the bootstrap approach for model arbitrariness evaluation
    with_predict_proba
        [Optional] A flag if model can return probabilities for its predictions.
         If no, only metrics based on labels (not labels and probabilities) will be computed.
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
         As for now, 0, 1, 2 levels are supported.

    """
    def __init__(self, base_model, base_model_name: str, bootstrap_fraction: float,
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 target_column: str, dataset_name: str, n_estimators: int, random_state: int = None,
                 with_predict_proba: bool = True, notebook_logs_stdout: bool = False, verbose: int = 0):
        super().__init__(base_model=base_model,
                         base_model_name=base_model_name,
                         bootstrap_fraction=bootstrap_fraction,
                         X_train=X_train,
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         dataset_name=dataset_name,
                         n_estimators=n_estimators,
                         random_state=random_state,
                         with_predict_proba=with_predict_proba,
                         notebook_logs_stdout=notebook_logs_stdout,
                         verbose=verbose)
        self.target_column = target_column

    def _fit_model(self, classifier, X_train: pd.DataFrame, y_train: pd.DataFrame, random_state: int):
        """
        Fit a classifier that is an instance of self.base_model
        """
        # Get the signature of the function
        signature = inspect.signature(classifier.fit)
        if 'random_state' in signature.parameters:
            return classifier.fit(X_train, y_train, random_state=random_state)
        elif 'seed' in signature.parameters:
            return classifier.fit(X_train, y_train, seed=random_state)

        # Sklearn API
        return classifier.fit(X_train, y_train.values.ravel())

    def _batch_predict(self, classifier, X_test: pd.DataFrame, random_state: int):
        """
        Predict with the classifier for X_test set and return predictions
        """
        # Get the signature of the function
        signature = inspect.signature(classifier.predict)
        if 'random_state' in signature.parameters:
            return classifier.predict(X_test, random_state=random_state)
        elif 'seed' in signature.parameters:
            return classifier.predict(X_test, seed=random_state)

        # Sklearn API
        return classifier.predict(X_test)

    def _batch_predict_proba(self, classifier, X_test: pd.DataFrame, random_state: int):
        """
        Predict with the classifier for X_test set and return probabilities for each class for each test point
        """
        # Get the signature of the function
        signature = inspect.signature(classifier.predict_proba)
        if 'random_state' in signature.parameters:
            return classifier.predict_proba(X_test, random_state=random_state)[:, 0]
        elif 'seed' in signature.parameters:
            return classifier.predict_proba(X_test, seed=random_state)[:, 0]

        # Sklearn API
        return classifier.predict_proba(X_test)[:, 0]
