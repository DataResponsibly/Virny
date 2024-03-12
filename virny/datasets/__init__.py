"""
This module contains sample datasets and data loaders.
The purpose is to provide sample datasets for functionality testing and show examples of data loaders (aka dataset classes).
"""
from .finance import GermanCreditDataset, BankMarketingDataset
from .compas import CompasWithoutSensitiveAttrsDataset, CompasDataset
from .healthcare import CardiovascularDiseaseDataset, RicciDataset
from .education import LawSchoolDataset, StudentPerformancePortugueseDataset
from .folktables import (ACSIncomeDataset, ACSEmploymentDataset, ACSMobilityDataset, ACSTravelTimeDataset,
                         ACSPublicCoverageDataset)


__all__ = [
    "GermanCreditDataset",
    "BankMarketingDataset",
    "CompasWithoutSensitiveAttrsDataset",
    "CompasDataset",
    "CardiovascularDiseaseDataset",
    "RicciDataset",
    "LawSchoolDataset",
    "StudentPerformancePortugueseDataset",
    "ACSIncomeDataset",
    "ACSEmploymentDataset",
    "ACSMobilityDataset",
    "ACSTravelTimeDataset",
    "ACSPublicCoverageDataset",
]
