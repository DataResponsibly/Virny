import pathlib
import pandas as pd
import numpy as np

from sklearn import preprocessing
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility, \
    employment_filter, adult_filter, public_coverage_filter
from virny.datasets.base import BaseDataLoader


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


class ACSIncomeDataset(BaseDataLoader):
    """
    Dataset class for the income task from the folktables dataset.
    Target: binary classification, predict if a person has an annual income > $50,000.
    Source of the dataset: https://github.com/socialfoundations/folktables

    Parameters
    ----------
    state
        State in the US for which to get the data. All states in the US are available.
    year
        Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.
    root_dir
        Path to the root directory where to store the extracted dataset or where it is stored.
    with_nulls
        Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.
    with_filter
        Whether to use a folktables filter for this task. Default: True.
    optimize
        Whether to optimize the dataset size by downcasting categorical columns. Default: True.
    subsample_size
        Subsample size to create based on the input dataset.
    subsample_seed
        Seed for sampling using the sample() method from pandas.

    """
    def __init__(self, state, year, root_dir=None, with_nulls=False, with_filter=True,
                 optimize=True, subsample_size: int = None, subsample_seed: int = None):
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if with_filter:
            acs_data = adult_filter(acs_data)
        if subsample_size:
            acs_data = acs_data.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else acs_data.sample(subsample_size)
            acs_data = acs_data.reset_index(drop=True)

        features = ACSIncome.features
        target = ACSIncome.target
        categorical_columns = ['SCHL', 'COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
        numerical_columns = ['AGEP', 'WKHP']

        if with_nulls:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize:
            X_data = optimize_data_loading(X_data, categorical_columns)

        optimized_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            optimized_X_data[col] = X_data[col]
        y_data = acs_data[target].apply(lambda x: int(x > 50_000))

        columns_with_nulls = optimized_X_data.columns[optimized_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=optimized_X_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=optimized_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSEmploymentDataset(BaseDataLoader):
    """
    Dataset class for the employment task from the folktables dataset.
    Target: binary classification, predict if a person is employed.
    Source of the dataset: https://github.com/socialfoundations/folktables

    Parameters
    ----------
    state
        State in the US for which to get the data. All states in the US are available.
    year
        Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.
    root_dir
        Path to the root directory where to store the extracted dataset or where it is stored.
    with_nulls
        Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.
    with_filter
        Whether to use a folktables filter for this task. Default: True.
    optimize
        Whether to optimize the dataset size by downcasting categorical columns. Default: True.
    subsample_size
        Subsample size to create based on the input dataset.
    subsample_seed
        Seed for sampling using the sample() method from pandas.

    """
    def __init__(self, state, year, root_dir=None, with_nulls=False, with_filter=True,
                 optimize=True, subsample_size: int = None, subsample_seed: int = None):
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if with_filter:
            acs_data = employment_filter(acs_data)
        if subsample_size:
            acs_data = acs_data.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else acs_data.sample(subsample_size)
            acs_data = acs_data.reset_index(drop=True)

        features = ACSEmployment.features
        target = ACSEmployment.target
        categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC','SCHL']
        numerical_columns = ['AGEP']

        if with_nulls is True:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize:
            X_data = optimize_data_loading(X_data, categorical_columns)

        optimized_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            optimized_X_data[col] = X_data[col]
        y_data = acs_data[target].apply(lambda x: int(x == 1))

        columns_with_nulls = optimized_X_data.columns[optimized_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=optimized_X_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=optimized_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSMobilityDataset(BaseDataLoader):
    """
    Dataset class for the mobility task from the folktables dataset.
    Target: binary classification, predict whether a young adult moved addresses in the last year.
    Source of the dataset: https://github.com/socialfoundations/folktables

    Parameters
    ----------
    state
        State in the US for which to get the data. All states in the US are available.
    year
        Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.
    root_dir
        Path to the root directory where to store the extracted dataset or where it is stored.
    with_nulls
        Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.

    """
    def __init__(self, state, year, root_dir=None, with_nulls=False):
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        features = ACSMobility.features
        target = ACSMobility.target
        categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','COW','ESR']
        numerical_columns = ['AGEP', 'SCHL', 'PINCP', 'WKHP', 'JWMNP']

        if with_nulls:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        filtered_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            filtered_X_data[col] = X_data[col]

        y_data = acs_data[target].apply(lambda x: int(x == 1))

        columns_with_nulls = filtered_X_data.columns[filtered_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=acs_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=filtered_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSPublicCoverageDataset(BaseDataLoader):
    """
    Dataset class for the public coverage task from the folktables dataset.
    Target: binary classification, predict whether a low-income individual, not eligible for Medicare,
        has coverage from public health insurance.
    Source of the dataset: https://github.com/socialfoundations/folktables

    Parameters
    ----------
    state
        State in the US for which to get the data. All states in the US are available.
    year
        Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.
    root_dir
        Path to the root directory where to store the extracted dataset or where it is stored.
    with_nulls
        Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.
    with_filter
        Whether to use a folktables filter for this task. Default: True.
    optimize
        Whether to optimize the dataset size by downcasting categorical columns. Default: True.
    subsample_size
        Subsample size to create based on the input dataset.
    subsample_seed
        Seed for sampling using the sample() method from pandas.

    """
    def __init__(self, state, year, root_dir=None, with_nulls=False, with_filter=True,
                 optimize=True, subsample_size: int = None, subsample_seed: int = None):
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if with_filter:
            acs_data = public_coverage_filter(acs_data)
        if subsample_size:
            acs_data = acs_data.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else acs_data.sample(subsample_size)
            acs_data = acs_data.reset_index(drop=True)

        features = ACSPublicCoverage.features
        target = ACSPublicCoverage.target
        categorical_columns = ['SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','ESR','ST','FER','RAC1P']
        numerical_columns = ['AGEP', 'PINCP']

        if with_nulls is True:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize:
            X_data = optimize_data_loading(X_data, categorical_columns)

        optimized_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            optimized_X_data[col] = X_data[col]
        y_data = acs_data[target].apply(lambda x: int(x == 1))

        columns_with_nulls = optimized_X_data.columns[optimized_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=optimized_X_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=optimized_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSTravelTimeDataset(BaseDataLoader):
    """
    Dataset class for the travel time task from the folktables dataset.
    Target: binary classification, predict whether a working adult has a travel time to work of greater than 20 minutes.
    Source of the dataset: https://github.com/socialfoundations/folktables

    Parameters
    ----------
    state
        State in the US for which to get the data. All states in the US are available.
    year
        Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.
    root_dir
        Path to the root directory where to store the extracted dataset or where it is stored.
    with_nulls
        Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.

    """
    def __init__(self, state, year, root_dir=None, with_nulls=False):
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        features = ACSTravelTime.features
        target = ACSTravelTime.target
        categorical_columns = ['MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','POWPUMA','POVPIP']
        numerical_columns = ['AGEP', 'SCHL']

        if with_nulls:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        filtered_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            filtered_X_data[col] = X_data[col]
            
        y_data = acs_data[target].apply(lambda x: int(x > 20))
        columns_with_nulls = filtered_X_data.columns[filtered_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=acs_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=filtered_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSDataset_from_demodq(BaseDataLoader):
    """ Following https://github.com/schelterlabs/demographic-data-quality """
    def __init__(self, state, year, root_dir=None, with_nulls=False, optimize=True):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        features =  ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        target = 'PINCP'
        categorical_columns = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
        numerical_columns = ['AGEP', 'WKHP']

        if with_nulls:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize:
            X_data = optimize_data_loading(X_data, categorical_columns)

        filtered_X_data = X_data
        y_data = acs_data[target].apply(lambda x: x >= 50000).astype(int)
        columns_with_nulls = filtered_X_data.columns[filtered_X_data.isna().any().to_list()].to_list()

        super().__init__(
            full_df=acs_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=filtered_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


def optimize_data_loading(data, categorical):
    """
    Optimizing the dataset size by downcasting categorical columns
    """
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data
