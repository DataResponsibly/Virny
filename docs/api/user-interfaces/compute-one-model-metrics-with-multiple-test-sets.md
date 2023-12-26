# compute_one_model_metrics_with_multiple_test_sets

Compute subgroup metrics for the base model based on dataset.X_test and each extra test set in extra_test_sets_lst. Save results in `save_results_dir_path` folder.

Return a dataframe of model metrics.

## Parameters

- **base_model**

    Base model for metrics computation

- **n_estimators** (*int*)

    Number of estimators for bootstrap to compute subgroup stability metrics

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **extra_test_sets_lst** (*list*)

    List of extra test sets like [(X_test1, y_test1, init_features_df1), (X_test2, y_test2, init_features_df2), ...]  to compute metrics.

- **bootstrap_fraction** (*float*)

    Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **dataset_name** (*str*)

    Dataset name to name a result file with metrics

- **base_model_name** (*str*)

    Model name to name a result file with metrics

- **model_setting** (*str*) – defaults to `batch`

    Currently, only batch models are supported. Default: 'batch'.

- **computation_mode** (*str*) – defaults to `None`

    [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported.




