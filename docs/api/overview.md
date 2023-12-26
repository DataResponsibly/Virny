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
- [CompasDataset](../datasets/CompasDataset)
- [CompasWithoutSensitiveAttrsDataset](../datasets/CompasWithoutSensitiveAttrsDataset)
- [CreditCardDefaultDataset](../datasets/CreditCardDefaultDataset)
- [DiabetesDataset](../datasets/DiabetesDataset)
- [LawSchoolDataset](../datasets/LawSchoolDataset)
- [RicciDataset](../datasets/RicciDataset)

## metrics


This module contains functions for computing subgroup variance and error metrics.


- [aleatoric_uncertainty](../metrics/aleatoric-uncertainty)
- [churn](../metrics/churn)
- [conf_interval](../metrics/conf-interval)
- [entropy_from_predicted_probability](../metrics/entropy-from-predicted-probability)
- [iqr](../metrics/iqr)
- [jitter](../metrics/jitter)
- [label_stability](../metrics/label-stability)
- [mean_prediction](../metrics/mean-prediction)
