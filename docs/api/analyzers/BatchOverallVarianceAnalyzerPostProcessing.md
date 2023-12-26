# BatchOverallVarianceAnalyzerPostProcessing

Analyzer to compute subgroup variance metrics using the defined post-processor.



## Parameters

- **postprocessor**

    One of postprocessors from aif360 (https://aif360.readthedocs.io/en/stable/modules/algorithms.html#module-aif360.algorithms.postprocessing)

- **sensitive_attribute** (*str*)

    A sensitive attribute to use for post-processing

- **base_model**

    Base model for stability measuring

- **base_model_name** (*str*)

    Model name like 'DecisionTreeClassifier' or 'LogisticRegression'

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

- **target_column** (*str*)

    Name of the target column

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming

- **n_estimators** (*int*)

    Number of estimators in ensemble to measure base_model stability

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] A flag if model can return probabilities for its predictions.  If no, only metrics based on labels (not labels and probabilities) will be computed.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.  As for now, 0, 1, 2 levels are supported.




## Methods

???- note "UQ_by_boostrap"

    Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples and applying postprocessing intervention.

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

