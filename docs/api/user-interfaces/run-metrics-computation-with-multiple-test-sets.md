# run_metrics_computation_with_multiple_test_sets

Compute stability and accuracy metrics for each model in models_config based on dataset.X_test and each extra test set  in extra_test_sets_lst. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    Dataset object that contains all needed attributes like target, features, numerical_columns etc.

- **bootstrap_fraction** (*float*)

    Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap

- **dataset_name** (*str*)

    Dataset name to name a result file with metrics

- **extra_test_sets_lst** (*list*)

    List of extra test sets like [(X_test1, y_test1), (X_test2, y_test2), ...] to compute metrics

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **n_estimators** (*int*)

    Number of estimators for bootstrap to compute subgroup stability metrics

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **model_setting** (*str*) – defaults to `batch`

    Currently, only batch models are supported. Default: 'batch'.

- **computation_mode** (*str*) – defaults to `None`

    [Optional] A non-default mode for metrics computation. Should be included in the ComputationMode enum.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported.




