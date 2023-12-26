import os
import gc
import sys
import pandas as pd

from copy import deepcopy
from abc import ABCMeta, abstractmethod

from virny.custom_classes.custom_logger import get_logger
from virny.utils.stability_utils import generate_bootstrap
from virny.utils.stability_utils import count_prediction_metrics


class AbstractOverallVarianceAnalyzer(metaclass=ABCMeta):
    """
    Abstract class for an analyzer that computes overall variance metrics for subgroups.

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
    dataset_name
        Name of dataset, used for correct results naming
    n_estimators
        Number of estimators in ensemble to measure base_model stability
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
                 dataset_name: str, n_estimators: int, with_predict_proba: bool = True,
                 notebook_logs_stdout: bool = False, verbose: int = 0):
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.bootstrap_fraction = bootstrap_fraction
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]
        self.models_predictions = None
        self.prediction_metrics = None
        self.with_predict_proba = with_predict_proba

        self._notebook_logs_stdout = notebook_logs_stdout
        self._verbose = verbose
        self._logger = get_logger(verbose)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @abstractmethod
    def _fit_model(self, classifier, X_train, y_train):
        pass

    @abstractmethod
    def _batch_predict(self, classifier, X_test):
        pass

    @abstractmethod
    def _batch_predict_proba(self, classifier, X_test):
        pass

    def compute_metrics(self, save_results: bool = True, with_fit: bool = True):
        """
        Measure metrics for the base model. Save results to a .csv file.

        Parameters
        ----------
        save_results
            If to save result metrics in a file
        with_fit
            If to fit estimators in bootstrap

        """
        # Quantify uncertainty for the base model
        boostrap_size = int(self.bootstrap_fraction * self.X_train.shape[0])
        self.models_predictions = self.UQ_by_boostrap(boostrap_size, with_replacement=True, with_fit=with_fit)

        # Count metrics based on prediction proba results
        y_preds, self.prediction_metrics = count_prediction_metrics(self.y_test.values, self.models_predictions, self.with_predict_proba)
        self._logger.info(f'Successfully computed predict proba metrics')

        if save_results:
            self.save_metrics_to_file()
        else:
            return y_preds, self.y_test

    def UQ_by_boostrap(self, boostrap_size: int, with_replacement: bool, with_fit: bool = True) -> dict:
        """
        Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples.

        Return a dictionary where keys are models indexes, and values are lists of
         correspondent model predictions for X_test set.

        Parameters
        ----------
        boostrap_size
            Number of records in bootstrap splits
        with_replacement
            Enable replacement or not
        with_fit
            Whether to fit estimators in bootstrap

        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        if self._verbose >= 1:
            print('\n', flush=True)
        self._logger.info('Start classifiers testing by bootstrap')

        # Remove a progress bar for UQ without estimators fitting
        if self._notebook_logs_stdout:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        cycle_range = range(self.n_estimators) if with_fit is False else \
            tqdm(range(self.n_estimators),
                 desc="Classifiers testing by bootstrap",
                 colour="blue",
                 mininterval=10,
                 file=sys.stdout)

        # Train and test each estimator in models_predictions
        for idx in cycle_range:
            classifier = self.models_lst[idx]
            if with_fit:
                X_sample, y_sample = generate_bootstrap(self.X_train, self.y_train, boostrap_size, with_replacement)
                classifier = self._fit_model(classifier, X_sample, y_sample)
            models_predictions[idx] = self._batch_predict_proba(classifier, self.X_test)
            self.models_lst[idx] = classifier
            # Force garbage collection to avoid out of memory error
            if with_fit and ((idx + 1) % 10 == 0 or (idx + 1) == self.n_estimators):
                gc.collect()

        if self._verbose >= 1:
            print('\n', flush=True)
        self._logger.info('Successfully tested classifiers by bootstrap')

        return models_predictions

    def save_metrics_to_file(self):
        metrics_to_report = dict()
        metrics_to_report['Dataset_Name'] = [self.dataset_name]
        metrics_to_report['Base_Model_Name'] = [self.base_model_name]
        metrics_to_report['N_Estimators'] = [self.n_estimators]

        for metric in self.prediction_metrics:
            metrics_to_report[metric] = self.prediction_metrics[metric]

        metrics_df = pd.DataFrame(metrics_to_report)
        dir_path = os.path.join('..', '..', 'results', 'models_stability_metrics')
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{self.dataset_name}_{self.n_estimators}_estimators_{self.base_model_name}_base_model_stability_metrics.csv"
        metrics_df.to_csv(f'{dir_path}/{filename}', index=False)
