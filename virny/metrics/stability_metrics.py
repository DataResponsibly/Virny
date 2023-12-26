import itertools
import numpy as np
import pandas as pd
import scipy as sp


def std(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute standard deviation of predictive variance.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap

    """
    return np.mean(uq_predict_probas.std().values)


def iqr(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute inter-quantile range (IQR) of predictive variance.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap

    """
    return np.mean(sp.stats.iqr(uq_predict_probas, axis=0))


def churn(predicted_labels_1: list, predicted_labels_2: list):
    """
    Pairwise stability metric for two model predictions.

    Parameters
    ----------
    predicted_labels_1

    predicted_labels_2

    """
    return np.sum([int(predicted_labels_1[i] != predicted_labels_2[i])
                   for i in range(len(predicted_labels_1))]) / len(predicted_labels_1)


def jitter(y_true: pd.DataFrame, uq_labels: pd.DataFrame) -> float:
    """
    Jitter is a stability metric that shows how the base model predictions fluctuate.
    Values closer to 0 -- perfect stability, values closer to 1 -- extremely bad stability.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_labels
        `uq_labels` variable from count_prediction_metrics()

    """
    models_prediction_labels = uq_labels.values
    n_models = len(models_prediction_labels)
    models_idx_lst = [i for i in range(n_models)]
    churns_sum = 0
    for i, j in itertools.combinations(models_idx_lst, 2):
        churns_sum += churn(models_prediction_labels[i], models_prediction_labels[j])

    return churns_sum / (n_models * (n_models - 1) * 0.5)


def per_sample_label_stability(predicted_labels: list) -> float:
    """
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1.
    If the absolute difference is large, the label is more stable.
    If the difference is exactly zero, then it's extremely unstable --- equally likely to be classified as 0 or 1.

    Parameters
    ----------
    predicted_labels

    """
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos

    return np.abs(count_pos - count_neg) / len(predicted_labels)


def label_stability(y_true: pd.DataFrame, uq_labels: pd.DataFrame) -> float:
    """
    Compute per-sample accuracy for each model predictions.

    Return per_sample_accuracy and label_stability (refer to https://www.osti.gov/servlets/purl/1527311)

    Parameters
    ----------
    y_true
        y test dataset
    uq_labels
        `uq_labels` variable from count_prediction_metrics()

    """
    label_stability_lst = []
    for sample in range(len(y_true)):
        per_sample_predictions = list(uq_labels[sample].values)
        label_stability_lst.append(per_sample_label_stability(per_sample_predictions))

    return np.mean(label_stability_lst)
