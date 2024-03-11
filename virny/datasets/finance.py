import pathlib
import pandas as pd

from virny.datasets.base import BaseDataLoader


class CreditCardDefaultDataset(BaseDataLoader):
    """
    Dataset class for the Credit Card Default dataset that contains sensitive attributes among feature columns.
    Description: https://arxiv.org/pdf/2110.00530.pdf (Section 3.1.6)

    Parameters
    ----------
    dataset_path
        [Optional] Path to a file with the data

    """
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            filename = 'credit_card_default_clean.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path)
        target = 'default_payment'
        numerical_columns = [
            "limit_bal", "age",
            "bill_amt1", "bill_amt2", "bill_amt3",
            "bill_amt4", "bill_amt5", "bill_amt6",
            "pay_amt1", "pay_amt2", "pay_amt3",
            "pay_amt4", "pay_amt5", "pay_amt6"
        ]
        categorical_columns = [
            "sex", "education", "marriage",
            "pay_0", "pay_2", "pay_3",
            "pay_4", "pay_5", "pay_6"
        ]

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class CreditDataset(BaseDataLoader):
    """
    Dataset class for the Credit dataset that contains sensitive attributes among feature columns.
    Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'givemesomecredit.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path, index_col=0)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        target = 'SeriousDlqin2yrs'
        numerical_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                             'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                             'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines']
        categorical_columns = []

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
