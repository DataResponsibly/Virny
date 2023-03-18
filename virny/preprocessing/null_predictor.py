import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score


class NullPredictor:
    def __init__(self, base_classifier, input_columns, target_columns, categorical_columns, numerical_columns):
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.categorical_columns = [x for x in categorical_columns if x in self.input_columns]
        self.numerical_columns = [x for x in numerical_columns if x in self.input_columns]

        self.target_transformer = dict()
        self.target_column_types = dict()
        self.base_model = dict()
        self.fitted_model = dict()

        for col in self.target_columns:
            column_type = 'categorical' if col in categorical_columns else 'numerical'

            # We will need to binarize categorical target columns
            self.target_column_types[col] = column_type
            self.target_transformer[col] = LabelEncoder() if column_type == 'categorical' else None
            self.base_model[col] = base_classifier(column_type)
            self.fitted_model[col] = None

    def fit(self, data_with_nulls, y=None):
        # Fit only on rows without nulls
        data = data_with_nulls.dropna(inplace=False)
        for col in self.target_columns:
            X = data[self.input_columns]

            # Binarizing categorical targets before fitting
            if self.target_transformer[col]:
                y = self.target_transformer[col].fit_transform(data[col])
            else:
                y = data[col]

            encoder = ColumnTransformer(transformers=[
                ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_columns),
                ('numerical_features', StandardScaler(), self.numerical_columns)
            ])
            pipeline = Pipeline([('features', encoder), ('learner', self.base_model[col])])

            print('Train set shape: ', X.shape)
            model = pipeline.fit(X, y)
            print("Fit score: ", pipeline.score(X, y))
            self.fitted_model[col] = model

    def evaluate_prediction(self, target_column, X_test, actual, predicted):
        print("Model prediction score: ", self.fitted_model[target_column].score(X_test, actual))
        if self.target_column_types[target_column] == 'categorical':
            print("Prediction accuracy score: ", accuracy_score(actual, predicted.round()))
            print("Prediction f1 score: ", f1_score(actual, predicted.round(), average='macro'))
        else:
            print("Prediction RMSE score: ", mean_squared_error(actual, predicted.round(), squared=False))
            print("Prediction MAE score: ", mean_absolute_error(actual, predicted.round()))

    def transform(self, X, y=None):
        # Transform only those rows with nulls
        data = X.copy(deep=True)

        for col in self.target_columns:
            if self.fitted_model[col] is None:
                raise ValueError("Call fit before calling transform!")

            null_idx = np.where(data[col].isnull())[0]
            X_test = data[self.input_columns].iloc[null_idx]
            print('Test set shape: ', X_test.shape)
            predicted = self.fitted_model[col].predict(X_test)

            # Inverse transforming binary targets back into categories
            if self.target_transformer[col]:
                predicted = self.target_transformer[col].inverse_transform(predicted)
                print('Predicted values were inversely transformed')

            # Evaluate model prediction
            if isinstance(y, pd.Series):
                try:
                    self.evaluate_prediction(col, X_test, y, predicted)
                except Exception as err:
                    print("Error during prediction evaluation: ", err)

            data[col].iloc[null_idx] = predicted.round()
            data[col] = data[col].astype(int)

        return data

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)
