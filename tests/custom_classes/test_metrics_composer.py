import os

import pandas as pd
import pytest

from tests import config_params, models_config, ROOT_DIR, compare_metric_dfs
from virny.utils.custom_initializers import read_model_metric_dfs
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.configs.constants import *


@pytest.fixture(scope='module')
def models_metrics_dct1(models_config):
    metrics_dir_path = os.path.join(ROOT_DIR, 'tests', 'files_for_tests',
                                    'COMPAS_Without_Sensitive_Attributes_Metrics_20230202__094821')
    models_metrics_dct = read_model_metric_dfs(metrics_dir_path, model_names=list(models_config.keys()))
    return models_metrics_dct


@pytest.fixture(scope='module')
def models_metrics_dct2(models_config):
    metrics_dir_path = os.path.join(ROOT_DIR, 'tests', 'files_for_tests',
                                    'COMPAS_Without_Sensitive_Attributes_Metrics_20231021__205806')
    models_metrics_dct = read_model_metric_dfs(metrics_dir_path, model_names=list(models_config.keys()))
    return models_metrics_dct


# ========================== Test compose_metrics ==========================
def test_compose_metrics_true1(models_metrics_dct1, config_params):
    metrics_composer = MetricsComposer(models_metrics_dct1, config_params.sensitive_attributes_dct)
    models_composed_metrics_df = metrics_composer.compose_metrics()

    # Check shape
    assert models_composed_metrics_df.shape == (26, 5)

    # Check column names
    assert sorted(models_composed_metrics_df.columns.tolist()) == sorted(['Metric', 'Model_Name', 'sex', 'race', 'sex&race'])

    # Check unique Model_Name
    assert sorted(models_composed_metrics_df['Model_Name'].unique().tolist()) == sorted(['DecisionTreeClassifier', 'LogisticRegression'])

    # Check all metrics presence
    assert sorted(models_composed_metrics_df['Metric'].unique().tolist()) == (
        sorted([EQUALIZED_ODDS_TPR, EQUALIZED_ODDS_TNR, EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_FNR,
                DISPARATE_IMPACT, STATISTICAL_PARITY_DIFFERENCE, ACCURACY_DIFFERENCE, LABEL_STABILITY_DIFFERENCE,
                LABEL_STABILITY_RATIO, IQR_DIFFERENCE, STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE])
    )


def test_compose_metrics_true2(models_metrics_dct1, config_params):
    metrics_composer = MetricsComposer(models_metrics_dct1, {'sex': 0, 'race': 'Caucasian'})
    models_composed_metrics_df = metrics_composer.compose_metrics()

    # Check shape
    assert models_composed_metrics_df.shape == (26, 4)

    # Check column names
    assert sorted(models_composed_metrics_df.columns.tolist()) == sorted(['Metric', 'Model_Name', 'sex', 'race'])

    # Check unique Model_Name
    assert sorted(models_composed_metrics_df['Model_Name'].unique().tolist()) == sorted(['DecisionTreeClassifier', 'LogisticRegression'])

    # Check all metrics presence
    assert sorted(models_composed_metrics_df['Metric'].unique().tolist()) == (
        sorted([EQUALIZED_ODDS_TPR, EQUALIZED_ODDS_TNR, EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_FNR,
                DISPARATE_IMPACT, STATISTICAL_PARITY_DIFFERENCE, ACCURACY_DIFFERENCE, LABEL_STABILITY_DIFFERENCE,
                LABEL_STABILITY_RATIO, IQR_DIFFERENCE, STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE])
    )


def test_compose_metrics_true3(models_metrics_dct2, config_params):
    metrics_composer = MetricsComposer(models_metrics_dct2, config_params.sensitive_attributes_dct)
    models_composed_metrics_df = metrics_composer.compose_metrics()

    # Check shape
    assert models_composed_metrics_df.shape == (38, 5)

    # Check column names
    assert sorted(models_composed_metrics_df.columns.tolist()) == sorted(['Metric', 'Model_Name', 'sex', 'race', 'sex&race'])

    # Check unique Model_Name
    assert sorted(models_composed_metrics_df['Model_Name'].unique().tolist()) == sorted(['LogisticRegression', 'XGBClassifier'])

    # Check all metrics presence
    assert sorted(models_composed_metrics_df['Metric'].unique().tolist()) == (
        sorted([EQUALIZED_ODDS_TPR, EQUALIZED_ODDS_TNR, EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_FNR,
                DISPARATE_IMPACT, STATISTICAL_PARITY_DIFFERENCE, ACCURACY_DIFFERENCE, LABEL_STABILITY_DIFFERENCE,
                LABEL_STABILITY_RATIO, IQR_DIFFERENCE, STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE,
                ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO,
                OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO,
                EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO])
    )

    expected_composed_metrics_df = pd.read_csv(os.path.join(ROOT_DIR, 'tests', 'files_for_tests', 'composed_metrics',
                                                            'Multiple_Models_Interface_Use_Case.csv'), header=0)
    # Check error disparity metrics
    compare_metric_dfs(expected_composed_metrics_df=expected_composed_metrics_df,
                       actual_composed_metrics_df=models_composed_metrics_df,
                       model_name='XGBClassifier',
                       groups=['sex', 'race', 'sex&race'],
                       metrics_lst=[EQUALIZED_ODDS_TPR,
                                    EQUALIZED_ODDS_TNR,
                                    EQUALIZED_ODDS_FPR,
                                    EQUALIZED_ODDS_FNR,
                                    DISPARATE_IMPACT,
                                    STATISTICAL_PARITY_DIFFERENCE,
                                    ACCURACY_DIFFERENCE])
    # Check stability disparity metrics
    compare_metric_dfs(expected_composed_metrics_df=expected_composed_metrics_df,
                       actual_composed_metrics_df=models_composed_metrics_df,
                       model_name='XGBClassifier',
                       groups=['sex', 'race', 'sex&race'],
                       metrics_lst=[LABEL_STABILITY_RATIO,
                                    IQR_DIFFERENCE,
                                    STD_DIFFERENCE,
                                    STD_RATIO,
                                    JITTER_DIFFERENCE])
    # Check uncertainty disparity metrics
    compare_metric_dfs(expected_composed_metrics_df=expected_composed_metrics_df,
                       actual_composed_metrics_df=models_composed_metrics_df,
                       model_name='XGBClassifier',
                       groups=['sex', 'race', 'sex&race'],
                       metrics_lst=[OVERALL_UNCERTAINTY_DIFFERENCE,
                                    OVERALL_UNCERTAINTY_RATIO,
                                    ALEATORIC_UNCERTAINTY_DIFFERENCE,
                                    ALEATORIC_UNCERTAINTY_RATIO])
