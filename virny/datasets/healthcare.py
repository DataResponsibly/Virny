import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from virny.datasets.base import BaseDataLoader


class CardiovascularDiseaseDataset(BaseDataLoader):
    """
    Dataset class for the Cardiovascular Disease dataset that contains sensitive attributes among feature columns.
    Source and broad description: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'cardio.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)
        df = pd.read_csv(dataset_path, delimiter=';')

        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Preprocessing
        df = df.drop(['id'], axis=1)
        df['age'] = (df['age'] / 365).astype(int)

        columns_to_cast = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        columns_to_cast_dct = {col: str for col in columns_to_cast}
        df = df.astype(columns_to_cast_dct)

        target = 'cardio'
        numerical_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        categorical_columns = [column for column in df.columns if column not in numerical_columns + [target]]

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class RicciDataset(BaseDataLoader):
    """
    Dataset class for the Ricci dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/ricci_race.csv
    Description: https://arxiv.org/pdf/2110.00530.pdf

    Parameters
    ----------
    dataset_path
        [Optional] Path to a file with the data

    """
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            filename = 'ricci_race.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)

        df = pd.read_csv(dataset_path)

        target = 'Promoted'
        df[target] = df[target].replace([-1], 0)
        numerical_columns = ['Oral', 'Written', 'Combine']
        categorical_columns = ['Position', 'Race']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
