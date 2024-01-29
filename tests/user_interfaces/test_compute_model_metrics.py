import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from tests import (COMPAS_y_test, COMPAS_RF_bootstrap_predictions, COMPAS_RF_expected_preds, compare_metric_dfs,
                   COMPAS_RF_expected_metrics)

from virny.configs.constants import *
from virny.utils.protected_groups_partitioning import create_test_protected_groups
from virny.analyzers.subgroup_variance_calculator import SubgroupVarianceCalculator
from virny.analyzers.subgroup_error_analyzer import SubgroupErrorAnalyzer
from virny.utils.stability_utils import count_prediction_metrics
from virny.datasets.data_loaders import CompasWithoutSensitiveAttrsDataset
from virny.preprocessing.basic_preprocessing import preprocess_dataset


def test_subgroup_variance_and_error_analyzers(COMPAS_y_test, COMPAS_RF_bootstrap_predictions, COMPAS_RF_expected_preds,
                                               COMPAS_RF_expected_metrics):
    dataset_split_seed = 42
    test_set_fraction = 0.2

    data_loader = CompasWithoutSensitiveAttrsDataset()
    sensitive_attributes_dct = {'sex': 1, 'race': 'African-American', 'sex&race': None}
    column_transformer = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('numerical_features', StandardScaler(), data_loader.numerical_columns),
    ])
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, dataset_split_seed)
    test_protected_groups = create_test_protected_groups(base_flow_dataset.X_test, base_flow_dataset.init_features_df,
                                                         sensitive_attributes_dct)

    y_preds, prediction_metrics = count_prediction_metrics(COMPAS_y_test, COMPAS_RF_bootstrap_predictions)
    y_preds = pd.Series(y_preds, index=base_flow_dataset.y_test.index)
    subgroup_variance_calculator = SubgroupVarianceCalculator(X_test=base_flow_dataset.X_test,
                                                              y_test=base_flow_dataset.y_test,
                                                              sensitive_attributes_dct=sensitive_attributes_dct,
                                                              test_protected_groups=test_protected_groups,
                                                              computation_mode=None)
    subgroup_variance_calculator.set_overall_variance_metrics(prediction_metrics)
    subgroup_variance_metrics_dct = subgroup_variance_calculator.compute_subgroup_metrics(
        y_preds, COMPAS_RF_bootstrap_predictions,
        save_results=False, result_filename=None, save_dir_path=None
    )
    variance_metrics_df = pd.DataFrame(subgroup_variance_metrics_dct)

    # Compute error metrics for subgroups
    error_analyzer = SubgroupErrorAnalyzer(X_test=base_flow_dataset.X_test,
                                           y_test=base_flow_dataset.y_test,
                                           sensitive_attributes_dct=sensitive_attributes_dct,
                                           test_protected_groups=test_protected_groups,
                                           computation_mode=None)
    dtc_res = error_analyzer.compute_subgroup_metrics(y_preds=y_preds,
                                                      models_predictions=dict(),
                                                      save_results=False,
                                                      result_filename=None,
                                                      save_dir_path=None)
    error_metrics_df = pd.DataFrame(dtc_res)

    metrics_df = pd.concat([variance_metrics_df, error_metrics_df])
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df['Model_Name'] = 'RandomForestClassifier'

    # Check accuracy metrics
    compare_metric_dfs(expected_composed_metrics_df=COMPAS_RF_expected_metrics,
                       actual_composed_metrics_df=metrics_df,
                       model_name='RandomForestClassifier',
                       groups=['overall', 'sex_priv', 'sex_dis', 'race_priv', 'race_dis', 'sex&race_priv', 'sex&race_dis'],
                       metrics_lst=[MEAN_PREDICTION,
                                    STATISTICAL_BIAS,
                                    TPR,
                                    TNR,
                                    PPV,
                                    FNR,
                                    FPR,
                                    F1,
                                    ACCURACY,
                                    SELECTION_RATE,
                                    POSITIVE_RATE])
    # Check stability metrics
    compare_metric_dfs(expected_composed_metrics_df=COMPAS_RF_expected_metrics,
                       actual_composed_metrics_df=metrics_df,
                       model_name='RandomForestClassifier',
                       groups=['overall', 'sex_priv', 'sex_dis', 'race_priv', 'race_dis', 'sex&race_priv', 'sex&race_dis'],
                       metrics_lst=[STD, IQR, JITTER, LABEL_STABILITY])
    # Check uncertainty metrics
    compare_metric_dfs(expected_composed_metrics_df=COMPAS_RF_expected_metrics,
                       actual_composed_metrics_df=metrics_df,
                       model_name='RandomForestClassifier',
                       groups=['overall', 'sex_priv', 'sex_dis', 'race_priv', 'race_dis', 'sex&race_priv', 'sex&race_dis'],
                       metrics_lst=[ALEATORIC_UNCERTAINTY, OVERALL_UNCERTAINTY, EPISTEMIC_UNCERTAINTY])
