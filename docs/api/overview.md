# Overview

## analyzers


Subgroup Error and Variance Analyzers.

This module contains fairness and stability analysing methods for defined subgroups.
The purpose of an analyzer is to analyse defined metrics for defined subgroups.


- [AbstractOverallVarianceAnalyzer](../analyzers/AbstractOverallVarianceAnalyzer)
- [AbstractSubgroupAnalyzer](../analyzers/AbstractSubgroupAnalyzer)
- [BatchOverallVarianceAnalyzer](../analyzers/BatchOverallVarianceAnalyzer)
- [SubgroupErrorAnalyzer](../analyzers/SubgroupErrorAnalyzer)
- [SubgroupVarianceAnalyzer](../analyzers/SubgroupVarianceAnalyzer)
- [SubgroupVarianceCalculator](../analyzers/SubgroupVarianceCalculator)

## configs


Configs amd constants for the source code logic.



## custom_classes


This module contains custom classes for metrics computation interfaces.
The purpose is to split metrics computation and visualization pipeline on components
that are highly  customizable for future library features.


- [BaseFlowDataset](../custom-classes/BaseFlowDataset)
- [MetricsComposer](../custom-classes/MetricsComposer)
- [MetricsVisualizer](../custom-classes/MetricsVisualizer)

## datasets


This module contains sample datasets and data loaders.
The purpose is to provide sample datasets for functionality testing and show examples of data loaders (aka dataset classes).


- [ACSEmploymentDataset](../datasets/ACSEmploymentDataset)
- [ACSIncomeDataset](../datasets/ACSIncomeDataset)
- [ACSMobilityDataset](../datasets/ACSMobilityDataset)
- [ACSPublicCoverageDataset](../datasets/ACSPublicCoverageDataset)
- [ACSTravelTimeDataset](../datasets/ACSTravelTimeDataset)
- [CompasDataset](../datasets/CompasDataset)
- [CompasWithoutSensitiveAttrsDataset](../datasets/CompasWithoutSensitiveAttrsDataset)
- [DiabetesDataset](../datasets/DiabetesDataset)
- [LawSchoolDataset](../datasets/LawSchoolDataset)
- [RicciDataset](../datasets/RicciDataset)

## incremental_ml


## metrics


This module contains functions for computing subgroup variance and error metrics.


- [compute_churn](../metrics/compute-churn)
- [compute_conf_interval](../metrics/compute-conf-interval)
- [compute_entropy_from_predicted_probability](../metrics/compute-entropy-from-predicted-probability)
- [compute_jitter](../metrics/compute-jitter)
- [compute_per_sample_accuracy](../metrics/compute-per-sample-accuracy)
- [compute_std_mean_iqr_metrics](../metrics/compute-std-mean-iqr-metrics)

## preprocessing


Preprocessing techniques.

This module contains function for input dataset preprocessing.


- [get_dummies](../preprocessing/get-dummies)
- [make_features_dfs](../preprocessing/make-features-dfs)
- [preprocess_dataset](../preprocessing/preprocess-dataset)

## user_interfaces


User interfaces.

This module contains user interfaces for metrics computation.


- [compute_metrics_multiple_runs_with_db_writer](../user-interfaces/compute-metrics-multiple-runs-with-db-writer)
- [compute_metrics_multiple_runs_with_multiple_test_sets](../user-interfaces/compute-metrics-multiple-runs-with-multiple-test-sets)
- [compute_metrics_with_config](../user-interfaces/compute-metrics-with-config)
- [compute_model_metrics](../user-interfaces/compute-model-metrics)
- [compute_model_metrics_with_config](../user-interfaces/compute-model-metrics-with-config)
- [run_metrics_computation](../user-interfaces/run-metrics-computation)

## utils


Common helpers and utils.


- [count_prediction_stats](../utils/count-prediction-stats)
- [create_test_protected_groups](../utils/create-test-protected-groups)
- [validate_config](../utils/validate-config)

