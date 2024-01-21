# SubgroupVarianceCalculator

Calculator that calculates variance metrics for subgroups.



## Parameters

- **X_test** (*pandas.core.frame.DataFrame*)

    Processed features test set

- **y_test** (*pandas.core.frame.DataFrame*)

    Targets test set

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these subgroups

- **test_protected_groups** – defaults to `None`

    A dictionary where keys are sensitive attributes, and values input dataset rows  that are correspondent to these sensitive attributes.

- **computation_mode** (*str*) – defaults to `None`

    [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] A flag if model can return probabilities for its predictions.  If no, only metrics based on labels (not labels and probabilities) will be computed.




## Methods

???- note "compute_subgroup_metrics"

    Compute variance metrics for subgroups.

    Return a dict of dicts where key is 'overall' or a subgroup name, and value is a dict of metrics for this subgroup.

    **Parameters**

    - **y_preds**    
    - **models_predictions**     (*dict*)    
    - **save_results**     (*bool*)    
    - **result_filename**     (*str*)     – defaults to `None`    
    - **save_dir_path**     (*str*)     – defaults to `None`    
    
???- note "save_metrics_to_file"

    

    **Parameters**

    - **result_filename**     (*str*)    
    - **save_dir_path**     (*str*)    
    
???- note "set_overall_variance_metrics"

