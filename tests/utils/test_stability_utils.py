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

    assert np.array_equal(y_preds, np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1]))
    assert np.array_equal( uq_labels, np.array([[0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]]) )

    assert prediction_stats.jitter is not None
    assert prediction_stats.means_lst is not None
    assert prediction_stats.stds_lst is not None
    assert prediction_stats.iqr_lst is not None
    assert prediction_stats.entropy_lst is not None
    assert prediction_stats.per_sample_accuracy_lst is not None
    assert prediction_stats.label_stability_lst is not None


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
