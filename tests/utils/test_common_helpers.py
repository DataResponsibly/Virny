import numpy as np
from munch import DefaultMunch
from sklearn.model_selection import train_test_split

from tests import config_params, compas_dataset_class, compas_without_sensitive_attrs_dataset_class
from virny.utils.common_helpers import validate_config, confusion_matrix_metrics


def test_validate_config_true1(config_params):
    actual = validate_config(config_params)
    assert actual == True


def test_validate_config_true2():
    config_dct = {
        "dataset_name": 'COMPAS',
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian'},
    }
    config = DefaultMunch.fromDict(config_dct)

    actual = validate_config(config)
    assert actual == True


def test_validate_config_false1():
    config_dct = {
        "dataset_name": 'COMPAS',
        "bootstrap_fraction": 0.8,
        "n_estimators": 100,
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian', 'sex&race&age': None},
    }
    config = DefaultMunch.fromDict(config_dct)

    try:
        actual = validate_config(config)
    except ValueError:
        actual = False

    assert actual == False


def test_validate_config_false2():
    config_dct = {
        "dataset_name": 'COMPAS',
        "bootstrap_fraction": 1.8,
        "n_estimators": 100,
        "sensitive_attributes_dct": {'sex': 0, 'race': 'Caucasian'},
    }
    config = DefaultMunch.fromDict(config_dct)

    try:
        actual = validate_config(config)
    except ValueError:
        actual = False

    assert actual == False


def test_validate_config_false3():
    config_dct = {
        "dataset_name": 'COMPAS',
        "bootstrap_fraction": 1.8,
        "n_estimators": 100,
        "sensitive_attributes_dct": {'sex': 0, 'sex&race': None},
    }
    config = DefaultMunch.fromDict(config_dct)

    try:
        actual = validate_config(config)
    except ValueError:
        actual = False

    assert actual == False


def test_confusion_matrix_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    actual_metrics = confusion_matrix_metrics(y_true, y_preds)

    required_fields = ['TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1', 'Selection-Rate', 'Positive-Rate']
    for field in required_fields:
        assert field in actual_metrics.keys()
