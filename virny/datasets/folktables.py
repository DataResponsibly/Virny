import pathlib
import pandas as pd
import numpy as np

from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility, \
    employment_filter, adult_filter, public_coverage_filter

from virny.datasets.base import BaseDataLoader


def optimize_data_loading(data, categorical):
    """
    Optimizing the dataset size by downcasting categorical columns
    """
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data


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
        full_df = pd.concat([optimized_X_data, y_data], axis=1)

        super().__init__(
            full_df=full_df,
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
        full_df = pd.concat([optimized_X_data, y_data], axis=1)

        super().__init__(
            full_df=full_df,
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
        full_df = pd.concat([filtered_X_data, y_data], axis=1)

        super().__init__(
            full_df=full_df,
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
        full_df = pd.concat([optimized_X_data, y_data], axis=1)

        super().__init__(
            full_df=full_df,
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
        full_df = pd.concat([filtered_X_data, y_data], axis=1)

        super().__init__(
            full_df=full_df,
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
