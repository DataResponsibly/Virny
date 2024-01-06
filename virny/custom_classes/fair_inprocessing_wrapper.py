import numpy as np

from virny.utils.postprocessing_intervention_utils import construct_binary_label_dataset_from_df


class FairInprocessingWrapper:
    """
    A wrapper for fair inprocessors from aif360. The wrapper aligns fit, predict, and predict_proba methods
    to be compatible with sklearn models.

    Parameters
    ----------
    inprocessor
        An initialized inprocessor from aif360.
    sensitive_attr_for_intervention
        A sensitive attribute name to use in the fairness in-processing intervention.

    """

    def __init__(self, inprocessor, sensitive_attr_for_intervention):
        self.sensitive_attr_for_intervention = sensitive_attr_for_intervention
        self.inprocessor = inprocessor
        self.y_shape = None

    def fit(self, X, y):
        # Create a binary label dataset based on aif360 API
        self.y_shape = np.shape(y)
        train_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                      y_sample=y,
                                                                      target_column='target',
                                                                      sensitive_attribute=self.sensitive_attr_for_intervention)
        # Fit an inprocessor
        self.inprocessor.fit(train_binary_dataset)
        return self

    def predict_proba(self, X):
        y_empty = np.zeros(shape=self.y_shape)  # Create an empty target column to align with aif360 API
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        return test_dataset_pred.scores

    def predict(self, X):
        y_empty = np.zeros(shape=self.y_shape)
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        return test_dataset_pred.labels
