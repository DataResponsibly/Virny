import pathlib

import numpy as np
import pandas as pd

from virny.datasets.base import BaseDataLoader


class GermanCreditDataset(BaseDataLoader):
    """
    Dataset class for the German Credit dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/german_data_credit.csv
    General description and analysis: https://arxiv.org/pdf/2110.00530.pdf (Section 3.1.3)
    Broad description: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'german_data_credit.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)
        df = pd.read_csv(dataset_path)

        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Shortening long categorical values
        df = df.replace({
            'credit-history': {
                'critical account': '1',
                'existing credits paid back duly till now': '2',
                'delay in paying off': '3',
                'no credits taken': '4',
                'all credits at this bank paid back duly': '5'
            },
            'property': {
                'real estate': '1',
                'savings agreement/life insurance': '2',
                'unknown / no property': '3',
                'car or other': '4'
            },
            'job': {
                'skilled employee / official': '1',
                'unskilled - resident': '2',
                'management/ highly qualified employee': '3',
                'unemployed/ unskilled  - non-resident': '4'
            }
        })

        target = 'class-label'
        numerical_columns = ['duration', 'credit-amount', 'installment-rate', 'residence-since',
                             'age', 'existing-credits', 'number-people-provide-maintenance-for']
        categorical_columns = [column for column in df.columns if column not in numerical_columns + [target]]

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class BankMarketingDataset(BaseDataLoader):
    """
    Dataset class for the Bank Marketing dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/bank-full.csv
    General description and analysis: https://arxiv.org/pdf/2110.00530.pdf (Section 3.1.5)
    Broad description: https://archive.ics.uci.edu/dataset/222/bank+marketing

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'bank-full.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)
        df = pd.read_csv(dataset_path)

        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # 'duration' attribute highly affects the output target (e.g., if duration=0 then y='no').
        # Yet, the duration is not known before a call is performed.
        # Also, after the end of the call y is obviously known.
        # Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
        df = df.drop(columns=['duration'])

        # Remove columns with a great percentage of 'unknown' values
        df = df.drop(columns=['contact', 'poutcome'])

        # Replace 'unknown' marker with NaN
        for col in df.columns:
            if 'unknown' in df[col].unique():
                df[col] = df[col].replace({'unknown': np.nan})

        # Target preprocessing
        target = 'y'
        df = df.replace({
            target: {
                'no': 0,
                'yes': 1,
            },
        })

        numerical_columns = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
        categorical_columns = [column for column in df.columns if column not in numerical_columns + [target]]

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
