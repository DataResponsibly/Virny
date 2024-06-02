# SubgroupVarianceAnalyzer

Analyzer to compute variance metrics for subgroups.



## Parameters

- **model_setting** (*[metrics.ModelSetting](../../metrics/ModelSetting)*)

    Model learning type; a constant from virny.configs.constants.ModelSetting

- **n_estimators** (*int*)

    Number of estimators for bootstrap

- **base_model**

    Initialized base model to analyze

- **base_model_name** (*str*)

    Model name

- **bootstrap_fraction** (*float*)

    [0-1], fraction from train_pd_dataset for fitting an ensemble of base models

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    Initialized object of GenericPipeline class

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **test_protected_groups** (*dict*)

    A dictionary of protected groups where keys are subgroup names,  and values are X_test row indexes correspondent to this subgroup.

- **postprocessor** – defaults to `None`

    One of postprocessors from aif360 (https://aif360.readthedocs.io/en/stable/modules/algorithms.html#module-aif360.algorithms.postprocessing)

- **postprocessing_sensitive_attribute** (*str*) – defaults to `None`

    A sensitive attribute to use for post-processing

- **random_state** (*int*) – defaults to `None`

    [Optional] Controls the randomness of the bootstrap approach for model arbitrariness evaluation

- **computation_mode** (*str*) – defaults to `None`

    [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] True, if models in models_config have a predict_proba method and can return probabilities for predictions,  False, otherwise. Note that if it is set to False, only metrics based on labels (not labels and probabilities) will be computed.  Ignored when a postprocessor is not None, and set to False in this case.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.  As for now, 0, 1, 2 levels are supported.




## Methods

???- note "compute_metrics"

    Measure variance metrics for subgroups for the base model. Display variance plots for analysis if needed.  Save results to a .csv file if needed.

    Return averaged bootstrap predictions and a pandas dataframe of variance metrics for subgroups.

    **Parameters**

    - **save_results**     (*bool*)    
    - **result_filename**     (*str*)     – defaults to `None`    
    - **save_dir_path**     (*str*)     – defaults to `None`    
    - **with_fit**     (*bool*)     – defaults to `True`    
    
???- note "set_test_protected_groups"

???- note "set_test_sets"

