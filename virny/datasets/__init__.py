"""
This module contains sample datasets and data loaders.
The purpose is to provide sample datasets for functionality testing and show examples of data loaders (aka dataset classes).
"""
from .data_loaders import CompasWithoutSensitiveAttrsDataset, DiabetesDataset, CompasDataset, \
    ACSIncomeDataset, ACSEmploymentDataset, ACSMobilityDataset, ACSTravelTimeDataset, ACSPublicCoverageDataset, \
    RicciDataset, LawSchoolDataset


__all__ = [
    "CompasWithoutSensitiveAttrsDataset",
    "CompasDataset",
    "DiabetesDataset",
    "RicciDataset",
    "LawSchoolDataset",
    "ACSIncomeDataset",
    "ACSEmploymentDataset",
    "ACSMobilityDataset",
    "ACSTravelTimeDataset",
    "ACSPublicCoverageDataset",
]
