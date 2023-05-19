# compute_metrics_multiple_runs_with_multiple_test_sets

Compute stability and accuracy metrics for each model in models_config based on dataset.X_test and each extra test set  in extra_test_sets_lst. Arguments are defined as an input config object. Save results to a database after each run   appending fields and value from custom_tbl_fields_dct and using db_writer_func.   Index of each test set is also added as a separate column in out final records in the database   (0 index -- for dataset.X_test, 1 and greater -- for each extra test set in extra_test_sets_lst, keeping the original sequence).



## Parameters

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **extra_test_sets_lst**

    List of extra test sets like [(X_test1, y_test1), (X_test2, y_test2), ...] to compute metrics that are not equal to original dataset.X_test and dataset.y_test

- **config**

    Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **custom_tbl_fields_dct** (*dict*)

    Dictionary where keys are column names and values to add to inserted metrics during saving results to a database

- **db_writer_func**

    Python function object has one argument (run_models_metrics_df) and save this metrics df to a target database

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported.




