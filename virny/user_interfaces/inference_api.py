import pandas as pd

from virny.configs.constants import ModelSetting
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.analyzers.subgroup_error_analyzer import SubgroupErrorAnalyzer
from virny.analyzers.subgroup_variance_analyzer import SubgroupVarianceAnalyzer
from virny.utils.protected_groups_partitioning import create_test_protected_groups


def compute_metrics_with_fitted_bootstrap(fitted_bootstrap: list, test_base_flow_dataset: BaseFlowDataset,
                                          config, with_predict_proba: bool = True, verbose: int = 0):
    model_setting = ModelSetting.BATCH
    X_test, y_test = test_base_flow_dataset.X_test, test_base_flow_dataset.y_test
    test_protected_groups = create_test_protected_groups(X_test, config.init_sensitive_attrs_df, config.sensitive_attributes_dct)

    subgroup_variance_analyzer = SubgroupVarianceAnalyzer(model_setting=model_setting,
                                                          n_estimators=config.n_estimators,
                                                          base_model=None,
                                                          base_model_name=None,
                                                          bootstrap_fraction=config.bootstrap_fraction,
                                                          dataset=test_base_flow_dataset,
                                                          dataset_name=config.dataset_name,
                                                          sensitive_attributes_dct=config.sensitive_attributes_dct,
                                                          test_protected_groups=test_protected_groups,
                                                          random_state=config.random_state,
                                                          computation_mode=config.computation_mode,
                                                          with_predict_proba=with_predict_proba,
                                                          notebook_logs_stdout=False,
                                                          verbose=verbose)

    # Compute stability metrics for subgroups
    subgroup_variance_analyzer.set_fitted_bootstrap(fitted_bootstrap)
    y_preds, variance_metrics_df, _ = subgroup_variance_analyzer.compute_metrics(save_results=False,
                                                                                 result_filename=None,
                                                                                 save_dir_path=None,
                                                                                 with_fit=False)
    # Compute accuracy metrics for subgroups
    error_analyzer = SubgroupErrorAnalyzer(X_test=X_test,
                                           y_test=y_test,
                                           sensitive_attributes_dct=config.sensitive_attributes_dct,
                                           test_protected_groups=test_protected_groups,
                                           computation_mode=config.computation_mode)
    dtc_res = error_analyzer.compute_subgroup_metrics(y_preds,
                                                      models_predictions=dict(),
                                                      save_results=False,
                                                      result_filename=None,
                                                      save_dir_path=None)
    error_metrics_df = pd.DataFrame(dtc_res)

    metrics_df = pd.concat([variance_metrics_df, error_metrics_df])
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df['Model_Params'] = str(fitted_bootstrap[0]['model_obj'].get_params())
    metrics_df['Virny_Random_State'] = config.random_state

    return metrics_df
