import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pprint import pprint
from copy import deepcopy
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from virny.custom_classes.base_dataset import BaseFlowDataset


def folds_iterator(n_folds, samples_per_fold, size):
    """
    Iterator for GridSearch based on Cross-Validation

    :param n_folds: number of folds for Cross-Validation
    :param samples_per_fold: number of samples per fold
    """
    for i in range(n_folds):
        yield np.arange(0, size - samples_per_fold * (i + 1)), \
              np.arange(size - samples_per_fold * (i + 1), size - samples_per_fold * i)


def validate_model(model, x, y, params, n_folds, samples_per_fold):
    """
    Use GridSearchCV for a special model to find the best hyperparameters based on validation set
    """
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               scoring={
                                   "F1_Score": make_scorer(f1_score, average='macro'),
                                   "Accuracy_Score": make_scorer(accuracy_score),
                               },
                               refit="F1_Score",
                               n_jobs=-1,
                               cv=folds_iterator(n_folds, samples_per_fold, x.shape[0]),
                               verbose=10)
    grid_search.fit(x, y.values.ravel())
    best_index = grid_search.best_index_

    return grid_search.best_estimator_, \
           grid_search.cv_results_["mean_test_F1_Score"][best_index], \
           grid_search.cv_results_["mean_test_Accuracy_Score"][best_index], \
           grid_search.best_params_


def test_evaluation(cur_best_model, model_name, cur_best_params,
                    cur_x_train, cur_y_train, cur_x_test, cur_y_test,
                    dataset_title, show_plots, debug_mode):
    """
    Evaluate model on test set.

    :return: F1 score, accuracy and predicted values, which we use to visualisations for model comparison later.
    """
    cur_best_model.fit(cur_x_train, cur_y_train.values.ravel()) # refit model on the whole train set
    cur_model_pred = cur_best_model.predict(cur_x_test)
    test_f1_score = f1_score(cur_y_test, cur_model_pred, average='macro')
    test_accuracy = accuracy_score(cur_y_test, cur_model_pred)

    if debug_mode:
        print("#" * 20, f' {dataset_title} ', "#" * 20)
        print('Test model: ', model_name)
        print('Test model parameters:')
        pprint(cur_best_params)

        # print the scores
        print()
        print(classification_report(cur_y_test, cur_model_pred, digits=3))

    if show_plots:
        # plot the confusion matrix
        sns.set_style("white")
        cm = confusion_matrix(cur_y_test, cur_model_pred, labels=cur_best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Employed", "Not Employed"])
        disp.plot()
        plt.show()

    return test_f1_score, test_accuracy, cur_model_pred


def tune_ML_models(models_params_for_tuning: dict, base_flow_dataset: BaseFlowDataset,
                   dataset_name: str, n_folds: int = 3, samples_per_fold: int = None):
    """
    Tune each model on a validation set with GridSearchCV.

    Return each model with its best hyperparameters that have the highest F1 score and Accuracy.
     results_df is a dataframe with metrics and tuned parameters;
     models_config is a dict with model tuned params for the metrics computation stage
    """
    if samples_per_fold is None:
        samples_per_fold = len(base_flow_dataset.y_test)

    models_config = dict()
    tuned_params_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score', 'Accuracy_Score', 'Model_Best_Params'))
    # Find the most optimal hyperparameters based on accuracy and F1-score for each model in models_config
    for model_idx, (model_name, model_params) in enumerate(models_params_for_tuning.items()):
        try:
            print(f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_name}...")
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_params['model']),
                                                                               base_flow_dataset.X_train_val,
                                                                               base_flow_dataset.y_train_val,
                                                                               model_params['params'],
                                                                               n_folds, samples_per_fold)
            print(f'{datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_name} is finished '
                  f'[F1 score = {cur_f1_score}, Accuracy = {cur_accuracy}]\n')

        except Exception as err:
            print(f"ERROR with {model_name}: ", err)
            continue

        # Save test results of each model in dataframe
        tuned_params_df.loc[model_idx] = [dataset_name, model_name, cur_f1_score, cur_accuracy, cur_params]
        models_config[model_name] = model_params['model'].set_params(**cur_params)

    return tuned_params_df, models_config


def test_ML_models(best_results_df, models_config, n_folds, X_train, y_train, X_test, y_test,
                   dataset_title, show_plots, debug_mode):
    """
    Find the best model from defined list.
    Tune each model on a validation set with GridSearchCV and
    return best_model with its hyperparameters, which has the highest F1 score
    """
    results_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score',
                                       'Accuracy_Score',
                                       'Model_Best_Params'))
    samples_per_fold = len(y_test)
    best_f1_score = -np.Inf
    best_accuracy = -np.Inf
    best_model_pred = []
    best_model_name = 'No model'
    best_params = None
    idx = 0
    # find the best model among defined in models_config
    for model_config in models_config:
        try:
            print(f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_config['model_name']}...")
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_config['model']),
                                                                               X_train, y_train, model_config['params'],
                                                                               n_folds, samples_per_fold)
            print(f'{datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_config["model_name"]} is finished')

            test_f1_score, test_accuracy, cur_model_pred = test_evaluation(cur_model, model_config['model_name'], cur_params,
                                                                           X_train, y_train, X_test, y_test, dataset_title, show_plots, debug_mode)
        except Exception as err:
            print(f"ERROR with {model_config['model_name']}: ", err)
            continue

        # save test results of each model in dataframe
        results_df.loc[idx] = [dataset_title,
                               model_config['model_name'],
                               test_f1_score,
                               test_accuracy,
                               cur_params]
        idx += 1

        if test_f1_score > best_f1_score:
            best_f1_score = test_f1_score
            best_accuracy = test_accuracy
            best_model_name = model_config['model_name']
            best_params = cur_params
            best_model_pred = cur_model_pred

    # append results of best model in best_results_df
    best_results_df.loc[best_results_df.shape[0]] = [dataset_title,
                                                     best_model_name,
                                                     best_f1_score,
                                                     best_accuracy,
                                                     best_params,
                                                     best_model_pred]

    return results_df, best_results_df
