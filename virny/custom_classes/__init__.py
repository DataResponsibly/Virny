"""
This module contains custom classes for metrics computation interfaces.
The purpose is to split metrics computation and visualization pipeline on components
that are highly  customizable for future library features.
"""
from .base_dataset import BaseDataset
from .generic_pipeline import GenericPipeline
from .metrics_composer import MetricsComposer
from .metrics_visualizer import MetricsVisualizer


__all__ = [
    "BaseDataset",
    "GenericPipeline",
    "MetricsComposer",
    "MetricsVisualizer",
]
