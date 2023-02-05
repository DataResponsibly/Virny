import os
import pytest

from tests import config_params, models_config, ROOT_DIR
from virny.utils.custom_initializers import read_model_metric_dfs
from virny.custom_classes.metrics_composer import MetricsComposer


@pytest.fixture(scope='module')
def models_metrics_dct(models_config):
    metrics_dir_path = os.path.join(ROOT_DIR, 'tests', 'files_for_tests',
                                    'COMPAS_Without_Sensitive_Attributes_Metrics_20230202__094821')
    models_metrics_dct = read_model_metric_dfs(metrics_dir_path, model_names=list(models_config.keys()))
    return models_metrics_dct


# ========================== Test compose_metrics ==========================
def test_compose_metrics_true1(models_metrics_dct, config_params):
    metrics_composer = MetricsComposer(models_metrics_dct, config_params.sensitive_attributes_dct)
    models_composed_metrics_df = metrics_composer.compose_metrics()

    # Check shape
    assert models_composed_metrics_df.shape == (20, 5)

    # Check column names
    assert sorted(models_composed_metrics_df.columns.tolist()) == sorted(['Metric', 'Model_Name', 'sex', 'race', 'sex&race'])

    # Check unique Model_Name
    assert sorted(models_composed_metrics_df['Model_Name'].unique().tolist()) == sorted(['DecisionTreeClassifier', 'LogisticRegression'])

    # Check all metrics presence
    assert sorted(models_composed_metrics_df['Metric'].unique().tolist()) == sorted(['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact',
                                                                                     'Statistical_Parity_Difference', 'Accuracy_Parity', 'Label_Stability_Ratio',
                                                                                     'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity'])


def test_compose_metrics_true2(models_metrics_dct, config_params):
    metrics_composer = MetricsComposer(models_metrics_dct, {'sex': 0, 'race': 'Caucasian'})
    models_composed_metrics_df = metrics_composer.compose_metrics()

    # Check shape
    assert models_composed_metrics_df.shape == (20, 4)

    # Check column names
    assert sorted(models_composed_metrics_df.columns.tolist()) == sorted(['Metric', 'Model_Name', 'sex', 'race'])

    # Check unique Model_Name
    assert sorted(models_composed_metrics_df['Model_Name'].unique().tolist()) == sorted(['DecisionTreeClassifier', 'LogisticRegression'])

    # Check all metrics presence
    assert sorted(models_composed_metrics_df['Metric'].unique().tolist()) == sorted(['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact',
                                                                                     'Statistical_Parity_Difference', 'Accuracy_Parity', 'Label_Stability_Ratio',
                                                                                     'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity'])
