"""
This module contains sample datasets and data loaders.
The purpose is to provide sample datasets for functionality testing and show examples of data loaders (aka dataset classes).
"""
from .finance import CreditCardDefaultDataset
from .compas import CompasWithoutSensitiveAttrsDataset, CompasDataset
from .healthcare import DiabetesDataset, RicciDataset
from .education import LawSchoolDataset, StudentPerformancePortugueseDataset
from .folktables import (ACSIncomeDataset, ACSEmploymentDataset, ACSMobilityDataset, ACSTravelTimeDataset,
                         ACSPublicCoverageDataset)


__all__ = [
    "CompasWithoutSensitiveAttrsDataset",
    "CompasDataset",
    "CreditCardDefaultDataset",
    "DiabetesDataset",
    "RicciDataset",
    "StudentPerformancePortugueseDataset",
    "LawSchoolDataset",
    "ACSIncomeDataset",
    "ACSEmploymentDataset",
    "ACSMobilityDataset",
    "ACSTravelTimeDataset",
    "ACSPublicCoverageDataset",
]
