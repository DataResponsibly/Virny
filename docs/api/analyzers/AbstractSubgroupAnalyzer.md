# AbstractSubgroupAnalyzer

Abstract class for a subgroup analyzer to compute metrics for subgroups.



## Parameters

- **X_test** (*pandas.core.frame.DataFrame*)

    Processed features test set

- **y_test** (*pandas.core.frame.DataFrame*)

    Targets test set

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these attributes

- **test_protected_groups** (*dict*)

    A dictionary where keys are sensitive attributes, and values input dataset rows  that are correspondent to these sensitive attributes




## Methods

???- note "compute_subgroup_metrics"

    Compute metrics for each subgroup in self.test_protected_groups using _compute_metrics method.

    Return a dictionary where keys are subgroup names, and values are subgroup metrics.

    **Parameters**

    - **y_preds**    
    - **save_results**     (*bool*)    
    - **result_filename**     (*str*)     – defaults to `None`    
    - **save_dir_path**     (*str*)     – defaults to `None`    
    
???- note "save_metrics_to_file"

    

    **Parameters**

    - **result_filename**     (*str*)    
    - **save_dir_path**     (*str*)    
    
