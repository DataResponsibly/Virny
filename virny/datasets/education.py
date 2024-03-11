import pathlib
import pandas as pd

from virny.datasets.base import BaseDataLoader


class StudentPerformancePortugueseDataset(BaseDataLoader):
    """
    Dataset class for the Student Performance Portuguese dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/student_por_new.csv
    Description: https://arxiv.org/pdf/2110.00530.pdf (Section 3.4.1)

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'student_por_new.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath(filename)
        df = pd.read_csv(dataset_path)

        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Preprocessing
        df['class'] = [1 if v == "High" else 0 for v in df['class']]

        target = 'class'
        numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
                             'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
        categorical_columns = [column for column in df.columns if column not in numerical_columns + [target]]

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class LawSchoolDataset(BaseDataLoader):
    """
    Dataset class for the Law School dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/law_school_clean.csv
    Description: https://arxiv.org/pdf/2110.00530.pdf

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using the sample() method from pandas
    dataset_path
        [Optional] Path to a file with the data

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None, dataset_path=None):
        if dataset_path is None:
            filename = 'law_school_clean.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Cast columns
        columns_to_cast = ['fulltime', 'fam_inc', 'male', 'tier']
        columns_to_cast_dct = {col: "str" for col in columns_to_cast}
        df = df.astype(columns_to_cast_dct)

        target = 'pass_bar'
        df = df.astype({target: int})

        numerical_columns = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa']
        categorical_columns = ['fulltime', 'fam_inc', 'male', 'tier', 'race']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

