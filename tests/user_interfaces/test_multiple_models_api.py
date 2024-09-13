import sys
import copy
import pathlib

from virny.utils.custom_initializers import read_model_metric_dfs
from virny.user_interfaces import compute_metrics_with_config

from tests import compare_metric_dfs_v2, compare_metric_dfs_with_tolerance


def test_compute_metrics_with_config_same_seed(law_school_dataset_1k_params):
    base_flow_dataset, config, models_config, save_results_dir_path = law_school_dataset_1k_params

    config.random_state = 100
    metrics_dct1 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)
    metrics_dct2 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)

    # Drop technical columns
    metrics_dct1['LogisticRegression'] = metrics_dct1['LogisticRegression'].drop('Runtime_in_Mins', axis=1)
    metrics_dct2['LogisticRegression'] = metrics_dct2['LogisticRegression'].drop('Runtime_in_Mins', axis=1)

    assert compare_metric_dfs_v2(metrics_dct1['LogisticRegression'], metrics_dct2['LogisticRegression'])


def test_compute_metrics_with_config_diff_seeds(law_school_dataset_1k_params):
    base_flow_dataset, config, models_config, save_results_dir_path = law_school_dataset_1k_params

    config.random_state = 100
    metrics_dct1 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)
    config.random_state = 200
    metrics_dct2 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)

    # Drop technical columns
    metrics_dct1['LogisticRegression'] = metrics_dct1['LogisticRegression'].drop('Runtime_in_Mins', axis=1)
    metrics_dct2['LogisticRegression'] = metrics_dct2['LogisticRegression'].drop('Runtime_in_Mins', axis=1)

    assert not compare_metric_dfs_v2(metrics_dct1['LogisticRegression'], metrics_dct2['LogisticRegression'])


def test_compute_metrics_with_config_none_seeds(law_school_dataset_1k_params):
    base_flow_dataset, config, models_config, save_results_dir_path = law_school_dataset_1k_params

    metrics_dct1 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)
    metrics_dct2 = compute_metrics_with_config(dataset=base_flow_dataset,
                                               config=config,
                                               models_config=copy.deepcopy(models_config),
                                               save_results_dir_path=save_results_dir_path)

    # Drop technical columns
    metrics_dct1['LogisticRegression'] = metrics_dct1['LogisticRegression'].drop('Runtime_in_Mins', axis=1)
    metrics_dct2['LogisticRegression'] = metrics_dct2['LogisticRegression'].drop('Runtime_in_Mins', axis=1)

    assert not compare_metric_dfs_v2(metrics_dct1['LogisticRegression'], metrics_dct2['LogisticRegression'])


def test_compute_metrics_with_config_should_equal_prev_release_results(law_school_dataset_20k_params):
    base_flow_dataset, config, models_config, save_results_dir_path = law_school_dataset_20k_params

    config.random_state = 100
    metrics_dct = compute_metrics_with_config(dataset=base_flow_dataset,
                                              config=config,
                                              models_config=copy.deepcopy(models_config),
                                              save_results_dir_path=save_results_dir_path)

    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print("Python 3.12 or newer is installed.")
        metrics_path = str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'law_school_dataset_20k', 'python_3_12'))
    elif sys.version_info.major == 3 and sys.version_info.minor == 11:
        print("Python 3.11 or newer is installed.")
        metrics_path = str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'law_school_dataset_20k', 'python_3_11'))
    elif sys.version_info.major == 3 and sys.version_info.minor == 10:
        print("Python 3.10 or newer is installed.")
        metrics_path = str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'law_school_dataset_20k', 'python_3_10'))
    elif sys.version_info.major == 3 and sys.version_info.minor == 9:
        print("Python 3.9 or newer is installed.")
        metrics_path = str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'law_school_dataset_20k', 'python_3_9'))
    else:
        print("Older version of Python is installed.")
        metrics_path = str(pathlib.Path(__file__).parent.parent.joinpath('files_for_tests', 'law_school_dataset_20k', 'python_3_8'))

    expected_metrics_dct = read_model_metric_dfs(metrics_path, model_names=['LogisticRegression', 'DecisionTreeClassifier'])

    # Drop technical columns
    metrics_dct['LogisticRegression'] = metrics_dct['LogisticRegression'].drop('Runtime_in_Mins', axis=1)
    metrics_dct['DecisionTreeClassifier'] = metrics_dct['DecisionTreeClassifier'].drop('Runtime_in_Mins', axis=1)

    assert compare_metric_dfs_with_tolerance(expected_metrics_dct['LogisticRegression'], metrics_dct['LogisticRegression'])
    assert compare_metric_dfs_with_tolerance(expected_metrics_dct['DecisionTreeClassifier'], metrics_dct['DecisionTreeClassifier'])
