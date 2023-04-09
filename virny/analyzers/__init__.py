"""
Subgroup Error and Variance Analyzers.

This module contains fairness and stability analysing methods for defined subgroups.
The purpose of an analyzer is to analyse defined metrics for defined subgroups.
"""
from .abstract_overall_variance_analyzer import AbstractOverallVarianceAnalyzer
from .abstract_subgroup_analyzer import AbstractSubgroupAnalyzer
from .batch_overall_variance_analyzer import BatchOverallVarianceAnalyzer
from .subgroup_error_analyzer import SubgroupErrorAnalyzer
from .subgroup_variance_analyzer import SubgroupVarianceAnalyzer
from .subgroup_variance_calculator import SubgroupVarianceCalculator

__all__ = [
    "AbstractOverallVarianceAnalyzer",
    "AbstractSubgroupAnalyzer",
    "BatchOverallVarianceAnalyzer",
    "SubgroupErrorAnalyzer",
    "SubgroupVarianceAnalyzer",
    "SubgroupVarianceCalculator",
]
