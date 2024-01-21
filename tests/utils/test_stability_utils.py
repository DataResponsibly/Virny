import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tests import (config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class,
                   COMPAS_y_test, COMPAS_RF_bootstrap_predictions, COMPAS_RF_expected_preds)
from virny.utils.stability_utils import count_prediction_metrics, generate_bootstrap
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.configs.constants import *


# ========================== Test count_prediction_metrics ==========================
def test_count_prediction_metrics_true1():
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    uq_results = np.array([[0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.7, 0.6, 0.4, 0.4],
                           [0.7, 0.6, 0.4, 0.4, 0.5, 0.3, 0.2, 0.6, 0.4, 0.4]])
    y_preds, prediction_metrics = count_prediction_metrics(y_test, uq_results)

    assert np.array_equal(y_preds, np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]))

    # Check stability and uncertainty metrics
    alpha = 0.000_001
    assert abs(prediction_metrics[MEAN_PREDICTION] - 0.47000000000000003) < alpha
    assert abs(prediction_metrics[STATISTICAL_BIAS] - 0.42000000000000004) < alpha
    assert abs(prediction_metrics[JITTER] - 0.1) < alpha
    assert abs(prediction_metrics[LABEL_STABILITY] - 0.9) < alpha
    assert abs(prediction_metrics[STD] - 0.0565685424949238) < alpha
    assert abs(prediction_metrics[IQR] - 0.03999999999999998) < alpha
    assert abs(prediction_metrics[ALEATORIC_UNCERTAINTY] - 0.9345065014636438) < alpha
    assert abs(prediction_metrics[EPISTEMIC_UNCERTAINTY] - 0.02150068825) < alpha
    assert abs(prediction_metrics[OVERALL_UNCERTAINTY] - 0.9560071897163649) < alpha


def test_count_prediction_metrics_false1():
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    uq_results = np.array([[0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.7, 0.6, 0.4, 0.4]])

    try:
        y_preds, prediction_stats = count_prediction_metrics(y_test, uq_results)
        actual = True
    except ZeroDivisionError:
        actual = False

    assert actual == False


def test_count_prediction_metrics_true2(COMPAS_y_test, COMPAS_RF_bootstrap_predictions, COMPAS_RF_expected_preds):
    y_preds, prediction_metrics = count_prediction_metrics(COMPAS_y_test, COMPAS_RF_bootstrap_predictions)

    alpha = 0.000_001
    assert np.array_equal(y_preds, COMPAS_RF_expected_preds['1'].to_numpy())
    assert abs(prediction_metrics[MEAN_PREDICTION] - 0.5233320457326472) < alpha
    assert abs(prediction_metrics[STATISTICAL_BIAS] - 0.4044558084608265) < alpha
    assert abs(prediction_metrics[JITTER] - 0.11221320346320351) < alpha
    assert abs(prediction_metrics[LABEL_STABILITY] - 0.8408712121212122) < alpha
    assert abs(prediction_metrics[STD] - 0.06812843675435999) < alpha
    assert abs(prediction_metrics[IQR] - 0.08940024816894414) < alpha
    assert abs(prediction_metrics[ALEATORIC_UNCERTAINTY] - 0.8369703514251653) < alpha
    assert abs(prediction_metrics[EPISTEMIC_UNCERTAINTY] - 0.02378108452) < alpha
    assert abs(prediction_metrics[OVERALL_UNCERTAINTY] - 0.8607514359506866) < alpha


# ========================== Test generate_bootstrap ==========================
def test_generate_bootstrap_true1(compas_without_sensitive_attrs_dataset_class, config_params):
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), compas_without_sensitive_attrs_dataset_class.categorical_columns),
        ('numerical_features', StandardScaler(), compas_without_sensitive_attrs_dataset_class.numerical_columns),
    ])
    base_flow_ds = preprocess_dataset(compas_without_sensitive_attrs_dataset_class,
                                      column_transformer,
                                      test_set_fraction=0.2,
                                      dataset_split_seed=42)
    boostrap_size = int(config_params.bootstrap_fraction * base_flow_ds.X_train_val.shape[0])
    X_sample, y_sample = generate_bootstrap(base_flow_ds.X_train_val,
                                            base_flow_ds.y_train_val,
                                            boostrap_size,
                                            with_replacement=True)

    assert X_sample.shape[0] == boostrap_size
    assert X_sample.shape[0] == y_sample.shape[0]
    assert X_sample.shape == (3377, 14)
