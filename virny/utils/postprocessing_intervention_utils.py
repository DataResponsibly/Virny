import copy

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset


def construct_binary_label_dataset_from_samples(X_sample, y_sample, column_names, target_column, sensitive_attribute):
    df = pd.DataFrame(X_sample, columns=column_names)
    df[target_column] = y_sample

    binary_label_dataset = BinaryLabelDataset(df=df,
                                              label_names=[target_column],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)

    return binary_label_dataset


def construct_binary_label_dataset_from_df(X_sample, y_sample, target_column, sensitive_attribute):
    df = X_sample
    df[target_column] = y_sample

    binary_label_dataset = BinaryLabelDataset(df=df,
                                              label_names=[target_column],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)

    return binary_label_dataset


def predict_on_binary_label_dataset(model, orig_dataset, threshold=0.5):
    orig_dataset_pred = copy.deepcopy(orig_dataset)

    fav_idx = np.where(model.classes_ == orig_dataset.favorable_label)[0][0]
    y_pred_prob = model.predict_proba(orig_dataset.features)[:, fav_idx]
    orig_dataset.scores = y_pred_prob.reshape(-1, 1)

    y_pred = np.zeros_like(orig_dataset.labels)
    y_pred[y_pred_prob >= threshold] = orig_dataset.favorable_label
    y_pred[~(y_pred_prob >= threshold)] = orig_dataset.unfavorable_label
    orig_dataset_pred.labels = y_pred

    return orig_dataset_pred
