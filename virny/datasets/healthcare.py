import pathlib
import pandas as pd

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


class DiabetesDataset2019(BaseDataLoader):
    """
    Dataset class for the Diabetes dataset 2019 that contains sensitive attributes among feature columns.
    Source and broad description: https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019/data

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas
    with_nulls
        Whether to keep nulls in the dataset or drop rows with any nulls. Default: True.

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None, with_nulls: bool = True):
        filename = 'diabetes_dataset__2019.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)
        df = pd.read_csv(dataset_path)

        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)
        if not with_nulls:
            df = df.dropna(how='any',axis=0)
            df = df.reset_index(drop=True)

        # Preprocessing
        df = df.rename(columns={'Pregancies': 'Pregnancies'})
        df['Diabetic'] = df['Diabetic'].str.strip()
        df['Diabetic'].replace('no', 0, inplace=True)
        df['Diabetic'].replace('yes', 1, inplace=True)
        df['BPLevel'] = df['BPLevel'].str.strip().str.lower()

        target = 'Diabetic'
        numerical_columns = ['BMI', 'Sleep', 'SoundSleep', 'Pregnancies']
        categorical_columns = [column for column in df.columns if column not in numerical_columns + [target]]

        # Create a dictionary of ordered categories for ordinal categorical columns.
        # It can be useful for preprocessing of ordinal categorical columns if exist.
        ordered_categories_dct = {
            'PhysicallyActive': ['none', 'less than half an hr', 'more than half an hr', 'one hr or more'],
            'JunkFood': ['occasionally', 'often', 'very often', 'always'],
            'Stress': ['not at all', 'sometimes', 'very often', 'always'],
            'BPLevel': ['low', 'normal', 'high'],
            'UriationFreq': ['not much', 'quite often'],
        }

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            ordered_categories_dct=ordered_categories_dct,
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