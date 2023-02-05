import numpy as np

from tests import config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class
from virny.utils.stability_utils import count_prediction_stats, generate_bootstrap
from virny.utils.custom_initializers import create_base_pipeline


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
    base_pipeline = create_base_pipeline(compas_without_sensitive_attrs_dataset_class,
                                         config_params.sensitive_attributes_dct,
                                         model_seed=config_params.runs_seed_lst[0],
                                         test_set_fraction=config_params.test_set_fraction)
    boostrap_size = int(config_params.bootstrap_fraction * base_pipeline.X_train_val.shape[0])
    X_sample, y_sample = generate_bootstrap(base_pipeline.X_train_val,
                                            base_pipeline.y_train_val,
                                            boostrap_size,
                                            with_replacement=True)

    assert X_sample.shape[0] == boostrap_size
    assert X_sample.shape[0] == y_sample.shape[0]
    assert X_sample.shape == (3377, 14)
