# tune_ML_models

Tune each model on a validation set with GridSearchCV.

Return each model with its best hyperparameters that have the highest F1 score and Accuracy.  results_df is a dataframe with metrics and tuned parameters;  models_config is a dict with model tuned params for the metrics computation stage

## Parameters

- **models_params_for_tuning** (*dict*)

    A dictionary, where keys are model names and values are a dictionary of hyperparameters and value ranges to tune.

- **base_flow_dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    An instance of BaseFlowDataset object. Its train and test sets are used for training and tuning.

- **dataset_name** (*str*)

    A name of the dataset. Used to save tuned hyperparameters to a csv file with an appropriate filename.

- **n_folds** (*int*) – defaults to `3`

    The number of folds for k-fold cross validation.




