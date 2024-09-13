import copy
import inspect
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset

from virny.utils.common_helpers import has_method


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


def predict_on_binary_label_dataset(model, orig_dataset, random_state, threshold=0.5):
    orig_dataset_pred = copy.deepcopy(orig_dataset)
    fav_idx = np.where(model.classes_ == orig_dataset.favorable_label)[0][0]

    # PyTorch Tabular API
    if not has_method(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(orig_dataset.features, tta_seed=random_state)[f'{fav_idx}_probability']
    else:
        # Get the signature of the function
        signature = inspect.signature(model.predict_proba)
        if 'random_state' in signature.parameters:
            y_pred_prob = model.predict_proba(orig_dataset.features, random_state=random_state)[:, fav_idx]
        elif 'seed' in signature.parameters:
            y_pred_prob = model.predict_proba(orig_dataset.features, seed=random_state)[:, fav_idx]
        else:
            y_pred_prob = model.predict_proba(orig_dataset.features)[:, fav_idx]

    orig_dataset.scores = y_pred_prob.reshape(-1, 1)

    y_pred = np.zeros_like(orig_dataset.labels)
    y_pred[y_pred_prob >= threshold] = orig_dataset.favorable_label
    y_pred[~(y_pred_prob >= threshold)] = orig_dataset.unfavorable_label
    orig_dataset_pred.labels = y_pred

    return orig_dataset_pred
