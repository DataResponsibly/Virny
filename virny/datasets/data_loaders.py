import os
import pandas as pd
import numpy as np

from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility

from virny.custom_classes.base_dataset import BaseDataset


class CompasDataset(BaseDataset):
    """
    Dataset class for COMPAS dataset that contains sensitive attributes among feature columns.

    Parameters
    ----------
    dataset_path
        Path to a dataset file

    """
    def __init__(self, dataset_path: str = os.path.join('virny', 'data', 'COMPAS.csv')):
        df = pd.read_csv(dataset_path)

        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        target = 'recidivism'
        numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                               'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )


class CompasWithoutSensitiveAttrsDataset(BaseDataset):
    """
    Dataset class for COMPAS dataset that does not contain sensitive attributes among feature columns
     to test blind classifiers

    Parameters
    ----------
    dataset_path
        Path to a dataset file

    """
    def __init__(self, dataset_path = os.path.join('virny', 'data', 'COMPAS.csv')):
        # Read a dataset
        df = pd.read_csv(dataset_path)

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
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )


class ACSEmploymentDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'),
                 with_nulls=False, optimize=True, subsample=None):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if subsample is not None:
            acs_data = acs_data.sample(subsample)

        dataset = acs_data
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
            pandas_df=dataset,
            features=features,
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


class ACSMobilityDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'), with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
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
            pandas_df=acs_data,
            features=features,
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


class ACSPublicCoverageDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'), with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        features = ACSPublicCoverage.features
        target = ACSPublicCoverage.target
        categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','ESR','ST','FER','RAC1P']
        numerical_columns = ['AGEP', 'SCHL', 'PINCP']

        if with_nulls is True:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        filtered_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            filtered_X_data[col] = X_data[col]
            
        y_data = acs_data[target].apply(lambda x: int(x == 1))
        columns_with_nulls = filtered_X_data.columns[filtered_X_data.isna().any().to_list()].to_list()

        super().__init__(
            pandas_df=acs_data,
            features=features,
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


class ACSTravelTimeDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'), with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
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
            pandas_df=acs_data,
            features=features,
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
    

class ACSIncomeDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'), with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        features = ACSIncome.features
        target = ACSIncome.target
        categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
        numerical_columns = ['AGEP', 'SCHL']

        if with_nulls:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        filtered_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            filtered_X_data[col] = X_data[col]
            
        y_data = acs_data[target].apply(lambda x: int(x > 50000))
        columns_with_nulls = filtered_X_data.columns[filtered_X_data.isna().any().to_list()].to_list()

        super().__init__(
            pandas_df=acs_data,
            features=features,
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


class ACSDataset_from_demodq(BaseDataset):
    """ Following https://github.com/schelterlabs/demographic-data-quality """
    def __init__(self, state, year, root_dir=os.path.join('virny', 'data'), with_nulls=False, optimize=True):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
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
            pandas_df=acs_data,
            features=features,
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
