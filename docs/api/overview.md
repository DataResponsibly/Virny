# Overview

## analyzers


Subgroup Error and Variance Analyzers.

This module contains fairness and stability analysing methods for defined subgroups.
The purpose of an analyzer is to analyse defined metrics for defined subgroups.


- [AbstractOverallVarianceAnalyzer](../analyzers/AbstractOverallVarianceAnalyzer)
- [AbstractSubgroupAnalyzer](../analyzers/AbstractSubgroupAnalyzer)
- [BatchOverallVarianceAnalyzer](../analyzers/BatchOverallVarianceAnalyzer)
- [BatchOverallVarianceAnalyzerPostProcessing](../analyzers/BatchOverallVarianceAnalyzerPostProcessing)
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
- [MetricsInteractiveVisualizer](../custom-classes/MetricsInteractiveVisualizer)
- [MetricsVisualizer](../custom-classes/MetricsVisualizer)

## datasets


This module contains sample datasets and data loaders.
The purpose is to provide sample datasets for functionality testing and show examples of data loaders (aka dataset classes).


- [ACSEmploymentDataset](../datasets/ACSEmploymentDataset)
- [ACSIncomeDataset](../datasets/ACSIncomeDataset)
- [ACSMobilityDataset](../datasets/ACSMobilityDataset)
- [ACSPublicCoverageDataset](../datasets/ACSPublicCoverageDataset)
- [ACSTravelTimeDataset](../datasets/ACSTravelTimeDataset)
- [BankMarketingDataset](../datasets/BankMarketingDataset)
- [CardiovascularDiseaseDataset](../datasets/CardiovascularDiseaseDataset)
- [CompasDataset](../datasets/CompasDataset)
- [CompasWithoutSensitiveAttrsDataset](../datasets/CompasWithoutSensitiveAttrsDataset)
- [DiabetesDataset2019](../datasets/DiabetesDataset2019)
- [GermanCreditDataset](../datasets/GermanCreditDataset)
- [LawSchoolDataset](../datasets/LawSchoolDataset)
- [RicciDataset](../datasets/RicciDataset)
- [StudentPerformancePortugueseDataset](../datasets/StudentPerformancePortugueseDataset)

## metrics


This module contains functions for computing subgroup variance and error metrics.


- [aleatoric_uncertainty](../metrics/aleatoric-uncertainty)
- [confusion_matrix_metrics](../metrics/confusion-matrix-metrics)
- [iqr](../metrics/iqr)
- [jitter](../metrics/jitter)
- [label_stability](../metrics/label-stability)
- [mean_prediction](../metrics/mean-prediction)
- [overall_uncertainty](../metrics/overall-uncertainty)
- [statistical_bias](../metrics/statistical-bias)
- [std](../metrics/std)

## preprocessing


Preprocessing techniques.

This module contains function for input dataset preprocessing.


- [get_dummies](../preprocessing/get-dummies)
- [make_features_dfs](../preprocessing/make-features-dfs)
- [preprocess_dataset](../preprocessing/preprocess-dataset)

## user_interfaces


User interfaces.

This module contains user interfaces for metrics computation.


- [compute_metrics_with_config](../user-interfaces/compute-metrics-with-config)
- [compute_metrics_with_db_writer](../user-interfaces/compute-metrics-with-db-writer)
- [compute_metrics_with_multiple_test_sets](../user-interfaces/compute-metrics-with-multiple-test-sets)

## utils


Common helpers and utils.


- [count_prediction_metrics](../utils/count-prediction-metrics)
- [create_test_protected_groups](../utils/create-test-protected-groups)
- [tune_ML_models](../utils/tune-ML-models)
- [validate_config](../utils/validate-config)

