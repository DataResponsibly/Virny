import numpy as np
import pandas as pd
import scipy as sp


def entropy_from_predicted_probability(x):
    """
    Compute entropy from predicted probability

    Parameters
    ----------
    x
        Probability of 0 class

    """
    return sp.stats.entropy([x, 1-x], base=2)


def conf_interval(labels):
    """
    Create 95% confidence interval for population mean weight.

    Parameters
    ----------
    labels

    """
    return sp.stats.norm.interval(alpha=0.95, loc=np.mean(labels), scale=sp.stats.sem(labels))


def aleatoric_uncertainty(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute aleatoric uncertainty as average predictive entropy.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap.

    """
    return np.mean(uq_predict_probas.apply(entropy_from_predicted_probability).mean().values)


def overall_uncertainty(y_true: pd.DataFrame, uq_predict_probas: pd.DataFrame) -> float:
    """
    Compute overall uncertainty as predictive entropy.

    Parameters
    ----------
    y_true
        A pandas dataframe of true labels. Is not used in this function, required for consistency.
    uq_predict_probas
        A pandas dataframe of predictions (probabilities) from all estimators in the bootstrap.

    """
    main_predictions = uq_predict_probas.mean().values
    overall_entropy_lst = np.array([entropy_from_predicted_probability(x) for x in main_predictions])
    return np.mean(overall_entropy_lst)
