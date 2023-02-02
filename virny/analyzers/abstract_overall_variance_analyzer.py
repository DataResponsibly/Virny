import os
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm.notebook import tqdm
from abc import ABCMeta, abstractmethod

from virny.custom_classes.custom_logger import get_logger
from virny.utils.data_viz_utils import plot_generic
from virny.utils.stability_utils import generate_bootstrap
from virny.utils.stability_utils import count_prediction_stats, compute_std_mean_iqr_metrics


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

    """
    def __init__(self, base_model, base_model_name: str, bootstrap_fraction: float,
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 dataset_name: str, n_estimators: int):
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.bootstrap_fraction = bootstrap_fraction
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]
        self.models_predictions = None

        self.__logger = get_logger()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Metrics
        self.mean = None
        self.std = None
        self.iqr = None
        self.entropy = None
        self.jitter = None
        self.per_sample_accuracy = None
        self.label_stability = None

    @abstractmethod
    def _fit_model(self, classifier, X_train, y_train):
        pass

    @abstractmethod
    def _batch_predict(self, classifier, X_test):
        pass

    @abstractmethod
    def _batch_predict_proba(self, classifier, X_test):
        pass

    def compute_metrics(self, make_plots: bool = False, save_results: bool = True):
        """
        Measure metrics for the base model. Display plots for analysis if needed. Save results to a .pkl file

        Parameters
        ----------
        make_plots
            bool, if to display plots for analysis
        save_results
            If to save result metrics in a file

        """
        # Quantify uncertainty for the base model
        boostrap_size = int(self.bootstrap_fraction * self.X_train.shape[0])
        self.models_predictions = self.UQ_by_boostrap(boostrap_size, with_replacement=True)

        # Count metrics based on prediction proba results
        y_preds, uq_labels, prediction_stats = count_prediction_stats(self.y_test.values, self.models_predictions)
        self.__logger.info(f'Successfully computed predict proba metrics')

        self.__update_metrics(prediction_stats.means_lst,
                              prediction_stats.stds_lst,
                              prediction_stats.iqr_lst,
                              prediction_stats.entropy_lst,
                              prediction_stats.jitter,
                              prediction_stats.per_sample_accuracy_lst,
                              prediction_stats.label_stability_lst)

        # Display plots if needed
        if make_plots:
            self.print_metrics()

            # Count metrics based on label predictions to visualize plots
            labels_means_lst, labels_stds_lst, labels_iqr_lst = compute_std_mean_iqr_metrics(uq_labels)
            self.__logger.info(f'Successfully computed predict labels metrics')
            per_sample_accuracy_lst = prediction_stats.per_sample_accuracy_lst
            label_stability_lst = prediction_stats.label_stability_lst

            plot_generic(labels_means_lst, labels_stds_lst, "Mean of probability", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Probability mean vs Standard deviation")
            plot_generic(labels_stds_lst, label_stability_lst, "Standard deviation", "Label stability", x_lim=0.5, y_lim=1.01, plot_title="Standard deviation vs Label stability")
            plot_generic(labels_means_lst, label_stability_lst, "Mean", "Label stability", x_lim=1.01, y_lim=1.01, plot_title="Mean vs Label stability")
            plot_generic(per_sample_accuracy_lst, labels_stds_lst, "Accuracy", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Accuracy vs Standard deviation")
            plot_generic(per_sample_accuracy_lst, labels_iqr_lst, "Accuracy", "Inter quantile range", x_lim=1.01, y_lim=1.01, plot_title="Accuracy vs Inter quantile range")

        if save_results:
            self.save_metrics_to_file()
        else:
            return y_preds, self.y_test

    def UQ_by_boostrap(self, boostrap_size: int, with_replacement: bool) -> dict:
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

        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        print('\n', flush=True)
        self.__logger.info('Start classifiers testing by bootstrap')
        # Train and test each estimator in models_predictions
        for idx in tqdm(range(self.n_estimators),
                        desc="Classifiers testing by bootstrap",
                        colour="blue",
                        mininterval=10):
            classifier = self.models_lst[idx]
            X_sample, y_sample = generate_bootstrap(self.X_train, self.y_train, boostrap_size, with_replacement)
            classifier = self._fit_model(classifier, X_sample, y_sample)
            models_predictions[idx] = self._batch_predict_proba(classifier, self.X_test)

        print('\n', flush=True)
        self.__logger.info('Successfully tested classifiers by bootstrap')

        return models_predictions

    def __update_metrics(self, means_lst, stds_lst, iqr_lst, entropy_lst, jitter_lst,
                         per_sample_accuracy, label_stability):
        self.mean = np.mean(means_lst)
        self.std = np.mean(stds_lst)
        self.iqr = np.mean(iqr_lst)
        self.entropy = np.mean(entropy_lst)
        self.jitter = jitter_lst
        self.per_sample_accuracy = np.mean(per_sample_accuracy)
        self.label_stability = np.mean(label_stability)

    def print_metrics(self):
        precision = 4
        print('\n')
        print("#" * 30, " Stability metrics ", "#" * 30)
        print(f'Mean: {np.round(self.mean, precision)}\n'
              f'Std: {np.round(self.std, precision)}\n'
              f'IQR: {np.round(self.iqr, precision)}\n'
              f'Entropy: {np.round(self.entropy, precision)}\n'
              f'Jitter: {np.round(self.jitter, precision)}\n'
              f'Per sample accuracy: {np.round(self.per_sample_accuracy, precision)}\n'
              f'Label stability: {np.round(self.label_stability, precision)}\n\n')

    def get_metrics_dict(self):
        return {
            'Mean': self.mean,
            'Std': self.std,
            'IQR': self.iqr,
            'Entropy': self.entropy,
            'Jitter': self.jitter,
            'Per_Sample_Accuracy': self.per_sample_accuracy,
            'Label_Stability': self.label_stability,
        }

    def save_metrics_to_file(self):
        metrics_to_report = dict()
        metrics_to_report['Dataset_Name'] = [self.dataset_name]
        metrics_to_report['Base_Model_Name'] = [self.base_model_name]
        metrics_to_report['N_Estimators'] = [self.n_estimators]

        metrics_to_report['Mean'] = [self.mean]
        metrics_to_report['Std'] = [self.std]
        metrics_to_report['IQR'] = [self.iqr]
        metrics_to_report['Entropy'] = [self.entropy]
        metrics_to_report['Jitter'] = [self.jitter]
        metrics_to_report['Per_Sample_Accuracy'] = [self.per_sample_accuracy]
        metrics_to_report['Label_Stability'] = [self.label_stability]
        metrics_df = pd.DataFrame(metrics_to_report)

        dir_path = os.path.join('..', '..', 'results', 'models_stability_metrics')
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{self.dataset_name}_{self.n_estimators}_estimators_{self.base_model_name}_base_model_stability_metrics.csv"
        metrics_df.to_csv(f'{dir_path}/{filename}', index=False)
