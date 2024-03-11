import pathlib
import pandas as pd

from sklearn import preprocessing
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
        numerical_columns = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa']
        categorical_columns = ['fulltime', 'fam_inc', 'male', 'tier', 'race']

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
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

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
        le = preprocessing.LabelEncoder()
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
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

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


class CompasDataset(BaseDataLoader):
    """
    Dataset class for the COMPAS dataset that contains sensitive attributes among feature columns.

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
            filename = 'COMPAS.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        target = 'recidivism'
        numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                               'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class CompasWithoutSensitiveAttrsDataset(BaseDataLoader):
    """
    Dataset class for the COMPAS dataset that does not contain sensitive attributes among feature columns
     to test blind classifiers

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
            filename = 'COMPAS.csv'
            dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        # Initial data types transformation
        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        # Define params
        target = 'recidivism'
        numerical_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count']
        categorical_columns = ['age_cat_25 - 45', 'age_cat_Greater than 45','age_cat_Less than 25',
                               'c_charge_degree_F', 'c_charge_degree_M']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
