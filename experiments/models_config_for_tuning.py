from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


MODELS_TUNING_SEED = 42
MODELS_TUNING_TEST_SET_FRACTION = 0.2

# Redefine this function if your models have another method to redefine their seed parameter
def reset_model_seed(model, new_seed):
    if 'random_state' in model.get_params():
        model.set_params(random_state=new_seed)
    return model

'''
MODELS_CONFIG = [
    {
        'model_name': 'DecisionTreeClassifier',
        'model': DecisionTreeClassifier(random_state=MODELS_TUNING_SEED),
        'params': {
            "max_depth": [2, 5, 10, 20, 30],
            "min_samples_split" : [0.01, 0.02, 0.05, 0.1],
            "max_features": [0.6, 'sqrt'],
            "criterion": ["gini", "entropy"]
        }
    },
    {
        'model_name': 'LogisticRegression',
        'model': LogisticRegression(random_state=MODELS_TUNING_SEED),
        'params': {
            'penalty': ['l1', 'l2'],
            'C' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': range(50, 251, 50),
        }
    },
    {
        'model_name': 'RandomForestClassifier',
        'model': RandomForestClassifier(random_state=MODELS_TUNING_SEED),
        'params': {
            "max_depth": [3, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_estimators": [50, 100, 500, 700],
            "max_features": [0.6, 'auto', 'sqrt']
        }
    },
    {
        'model_name': 'XGBClassifier',
        'model': XGBClassifier(random_state=MODELS_TUNING_SEED, verbosity=0),
        'params': {
            'learning_rate': [0.1],
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3,5,7,10],
            'lambda':  [1,10,100]
        }
    },
    {
        'model_name': 'KNeighborsClassifier',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors' : [5, 7, 9, 11, 13, 15, 25],
            'weights' : ['uniform', 'distance'],
            'metric' : ['minkowski', 'euclidean', 'manhattan']
        }
    },
    
     {
         'model_name': 'MLPClassifier',
         'model': MLPClassifier(random_state=MODELS_TUNING_SEED, max_iter=1000),
         'params': {
             'hidden_layer_sizes':[(100,), (100,100,), (100,50,100,)],
             'activation': ['logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
         }
     }
]
'''

MODELS_CONFIG = [
     {
         'model_name': 'MLPClassifier',
         'model': MLPClassifier(random_state=MODELS_TUNING_SEED, max_iter=1000),
         'params': {
             'hidden_layer_sizes':[(100,), (100,100,), (100,50,100,)],
             'activation': ['logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
         }
     }
]
