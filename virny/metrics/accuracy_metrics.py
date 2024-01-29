import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from virny.configs.constants import *


def mean_prediction(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute mean predictions of all estimators in the boostrap.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap

    """
    return np.mean(uq_predict_probas.mean().values)


def statistical_bias_from_predict_proba(x, y_true):
    """
    Compute statistical bias from predicted probability

    Parameters
    ----------
    x
        Probability of 0 class
    y_true
        True label

    """
    # If x (main prediction) = 0.4, then expected value = 0 * 0.4 + 1 * (1 - 0.4) = 0.6.
    # For true label = 0, we get bias = abs(0 - 0.6) = 0.6.
    # For true label = 1, we get bias = abs(1 - 0.6) = 0.4.
    expected_val = 0 * x + 1 * (1 - x)
    return abs(y_true - expected_val)


def statistical_bias(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute statistical bias.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap

    """
    main_predictions = uq_predict_probas.mean().values
    statistical_bias_lst =  np.array(
        [statistical_bias_from_predict_proba(x, y_true) for x, y_true in np.column_stack((main_predictions, y_true))]
    )
    return np.mean(statistical_bias_lst)


def confusion_matrix_metrics(y_true, y_preds):
    """
    Compute accuracy metrics based on the confusion matrix.

    Parameters
    ----------
    y_true
        True labels.
    y_preds
        Predicted labels.

    """
    metrics = {}
    TN, FP, FN, TP = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()

    metrics[TPR] = TP/(TP+FN)
    metrics[TNR] = TN/(TN+FP)
    metrics[PPV] = TP/(TP+FP)
    metrics[FNR] = FN/(FN+TP)
    metrics[FPR] = FP/(FP+TN)
    metrics[ACCURACY] = (TP+TN)/(TP+TN+FP+FN)
    metrics[F1] = (2*TP)/(2*TP+FP+FN)
    metrics[SELECTION_RATE] = (TP+FP)/(TP+FP+TN+FN)
    metrics[POSITIVE_RATE] = (TP+FP)/(TP+FN)

    return metrics
