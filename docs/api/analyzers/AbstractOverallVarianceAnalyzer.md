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

- **random_state** (*int*) – defaults to `None`

    [Optional] Controls the randomness of the bootstrap approach for model arbitrariness evaluation

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] A flag if model can return probabilities for its predictions.  If no, only metrics based on labels (not labels and probabilities) will be computed.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.  As for now, 0, 1, 2 levels are supported.




## Methods

???- note "UQ_by_boostrap"

    Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples.

    Return a dictionary where keys are models indexes, and values are lists of  correspondent model predictions for X_test set.

    **Parameters**

    - **boostrap_size**     (*int*)    
    - **with_replacement**     (*bool*)    
    - **with_fit**     (*bool*)     – defaults to `True`    
    
???- note "compute_metrics"

    Measure metrics for the base model. Save results to a .csv file.

    **Parameters**

    - **save_results**     (*bool*)     – defaults to `True`    
    - **with_fit**     (*bool*)     – defaults to `True`    
    
???- note "save_metrics_to_file"

