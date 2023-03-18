import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from virny.configs.constants import NULL_PREDICTOR_SEED


def get_sample_rows(data, target_col, fraction):
    """
    Description: create a list of random indexes for rows, which will be used to place nulls in the dataset
    """
    n_values_to_discard = int(len(data) * min(fraction, 1.0))
    perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
    perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

    depends_on_col = np.random.choice(list(set(data.columns) - {target_col}))
    # Pick a random percentile of values in other column
    rows = data[depends_on_col].sort_values().iloc[perc_idx].index
    return rows


def decide_special_category(data):
    """
    Description: Decides which value to designate as a special value, based on the values already in the data (array)
    """
    data_type = data.dtype
    # If not numerical, simply set the special value to "special"
    try:
        # If data is numerical
        if 0 not in data:
            return 0
        else:
            return max(data) + 1
    except Exception as err:
        print("Data is not numerical, assigning string category")
        return data_type("Special")


def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()


def base_regressor(column_type):
    if column_type == 'numerical':
        model = LinearRegression()
    elif column_type == 'categorical':
        model = LogisticRegression()
    else:
        raise ValueError(
            "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_knn(column_type, n_neighbors):
    if column_type == 'numerical':
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif column_type == 'categorical':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError(
            "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_random_forest(column_type, n_estimators, max_depth, class_weight=None, min_samples_leaf=None, oob_score=None):
    if column_type == 'numerical':
        model = RandomForestRegressor(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      random_state = NULL_PREDICTOR_SEED)
    elif column_type == 'categorical':
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state = NULL_PREDICTOR_SEED,
                                       class_weight=class_weight,
                                       min_samples_leaf=min_samples_leaf,
                                       oob_score=oob_score)
    else:
        raise ValueError("Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model
