# Overview

## analyzers


Subgroup Statistical Bias and Variance Analyzers.

This module contains fairness and stability analysing methods for defined subgroups.
The purpose of an analyzer is to analyse defined metrics for defined subgroups.


- [AbstractOverallVarianceAnalyzer](../analyzers/AbstractOverallVarianceAnalyzer)
- [AbstractSubgroupAnalyzer](../analyzers/AbstractSubgroupAnalyzer)
- [BatchOverallVarianceAnalyzer](../analyzers/BatchOverallVarianceAnalyzer)
- [SubgroupStatisticalBiasAnalyzer](../analyzers/SubgroupStatisticalBiasAnalyzer)
- [SubgroupVarianceAnalyzer](../analyzers/SubgroupVarianceAnalyzer)
- [SubgroupVarianceCalculator](../analyzers/SubgroupVarianceCalculator)

## custom_classes


This module contains custom classes for metrics computation interfaces.
The purpose is to split metrics computation and visualization pipeline on components
that are highly  customizable for future library features.


- [BaseDataset](../custom-classes/BaseDataset)
- [CompasDataset](../custom-classes/CompasDataset)
- [CompasWithoutSensitiveAttrsDataset](../custom-classes/CompasWithoutSensitiveAttrsDataset)
- [GenericPipeline](../custom-classes/GenericPipeline)
- [MetricsComposer](../custom-classes/MetricsComposer)
- [MetricsVisualizer](../custom-classes/MetricsVisualizer)

## metrics


This module contains functions for variance and statistical bias metrics.


- [compute_churn](../metrics/compute-churn)
- [compute_conf_interval](../metrics/compute-conf-interval)
- [compute_entropy](../metrics/compute-entropy)
- [compute_jitter](../metrics/compute-jitter)
- [compute_per_sample_accuracy](../metrics/compute-per-sample-accuracy)
- [compute_std_mean_iqr_metrics](../metrics/compute-std-mean-iqr-metrics)

## preprocessing


Preprocessing techniques.

This module contains function for input dataset preprocessing.


- [get_dummies](../preprocessing/get-dummies)
- [make_features_dfs](../preprocessing/make-features-dfs)

## user_interfaces


User interfaces.

This module contains user interfaces for metrics computation.


- [compute_metrics_multiple_runs](../user-interfaces/compute-metrics-multiple-runs)
- [compute_model_metrics](../user-interfaces/compute-model-metrics)
- [compute_model_metrics_with_config](../user-interfaces/compute-model-metrics-with-config)
- [run_metrics_computation](../user-interfaces/run-metrics-computation)
- [run_metrics_computation_with_config](../user-interfaces/run-metrics-computation-with-config)

## utils


Common helpers and utils.


- [count_prediction_stats](../utils/count-prediction-stats)
- [create_test_protected_groups](../utils/create-test-protected-groups)
- [validate_config](../utils/validate-config)

