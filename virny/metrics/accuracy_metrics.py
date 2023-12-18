import numpy as np
import pandas as pd


def mean_prediction(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
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
    main_predictions = uq_predict_probas.mean().values
    statistical_bias_lst =  np.array(
        [statistical_bias_from_predict_proba(x, y_true) for x, y_true in np.column_stack((main_predictions, y_true))]
    )
    return np.mean(statistical_bias_lst)
