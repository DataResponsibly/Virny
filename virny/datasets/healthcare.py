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


class DiabetesDataset(BaseDataLoader):
    """
    Dataset class for the Diabetes dataset that contains sensitive attributes among feature columns.
    Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/diabetes-clean.csv
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
            filename = 'diabetes-clean.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath('data').joinpath(filename)

        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Cast columns
        columns_to_cast = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
        columns_to_cast_dct = {col: "str" for col in columns_to_cast}
        df = df.astype(columns_to_cast_dct)
        df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

        # Encode labels
        le = LabelEncoder()
        for i in ['diag_1', 'diag_2', 'diag_3']:
            df[i] = le.fit_transform(df[i])

        target = 'readmitted'
        df[target] = df[target].replace(['<30'], 1)
        df[target] = df[target].replace(['>30'], 0)
        numerical_columns = ['number_diagnoses', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                             'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
        categorical_columns = [
            'race',	'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
            'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol',	'troglitazone', 'tolazamide', 'examide', 'citoglipton',
            'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
        ]

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
