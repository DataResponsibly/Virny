# AbstractOverallVarianceAnalyzer

Abstract class for an analyzer that computes overall variance metrics for subgroups.



## Parameters

- **base_model**

    Base model for stability measuring

- **base_model_name** (*str*)

    Model name like 'HoeffdingTreeClassifier' or 'LogisticRegression'

- **bootstrap_fraction** (*float*)

    [0-1], fraction from train_pd_dataset for fitting an ensemble of base models

- **X_train** (*pandas.core.frame.DataFrame*)

    Processed features train set

- **y_train** (*pandas.core.frame.DataFrame*)

    Targets train set

- **X_test** (*pandas.core.frame.DataFrame*)

    Processed features test set

- **y_test** (*pandas.core.frame.DataFrame*)

    Targets test set

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming

- **n_estimators** (*int*)

    Number of estimators in ensemble to measure base_model stability

- **with_predict_proba** (*bool*) – defaults to `True`

