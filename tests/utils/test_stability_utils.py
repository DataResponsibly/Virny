import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from tests import config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class
from virny.utils.stability_utils import count_prediction_stats, generate_bootstrap
from virny.preprocessing.basic_preprocessing import preprocess_dataset


# ========================== Test count_prediction_stats ==========================
def test_count_prediction_stats_true1():
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    uq_results = np.array([[0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.7, 0.6, 0.4, 0.4],
                           [0.7, 0.6, 0.4, 0.4, 0.5, 0.3, 0.2, 0.6, 0.4, 0.4]])
    y_preds, uq_labels, prediction_stats = count_prediction_stats(y_test, uq_results)

    mean = np.mean(prediction_stats.means_lst)
    std = np.mean(prediction_stats.stds_lst)
    iqr = np.mean(prediction_stats.iqr_lst)
    aleatoric_uncertainty = np.mean(prediction_stats.mean_ensemble_entropy_lst)
    overall_uncertainty = np.mean(prediction_stats.overall_entropy_lst)
    statistical_bias = np.mean(prediction_stats.statistical_bias_lst)
    per_sample_accuracy = np.mean(prediction_stats.per_sample_accuracy_lst)
    label_stability = np.mean(prediction_stats.label_stability_lst)

    assert np.array_equal(y_preds, np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]))
    assert np.array_equal( uq_labels, np.array([[0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]]) )

    alpha = 0.000_001
    assert abs(prediction_stats.jitter - 0.1) < alpha
    assert abs(mean - 0.47000000000000003) < alpha
    assert abs(std - 0.0565685424949238) < alpha
    assert abs(iqr - 0.03999999999999998) < alpha
    assert abs(aleatoric_uncertainty - 0.9345065014636438) < alpha
    assert abs(overall_uncertainty - 0.9560071897163649) < alpha
    assert abs(statistical_bias - 0.42000000000000004) < alpha
    assert abs(per_sample_accuracy - 0.85) < alpha
    assert abs(label_stability - 0.9) < alpha


def test_count_prediction_stats_true2():
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    uq_results = np.array([[0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.7, 0.6, 0.4, 0.4]])

    try:
        y_preds, uq_labels, prediction_stats = count_prediction_stats(y_test, uq_results)
        actual = True
    except ZeroDivisionError:
        actual = False

    assert actual == False


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
