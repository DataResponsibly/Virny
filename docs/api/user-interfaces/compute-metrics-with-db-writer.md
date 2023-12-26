# compute_metrics_with_db_writer

Compute stability and accuracy metrics for each model in models_config. Arguments are defined as an input config object. Save results to a database after each run appending fields and value from custom_tbl_fields_dct and using db_writer_func.

Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **custom_tbl_fields_dct** (*dict*)

    Dictionary where keys are column names and values to add to inserted metrics during saving results to a database

- **db_writer_func**

    Python function object has one argument (run_models_metrics_df) and save this metrics df to a target database

- **postprocessor** – defaults to `None`

    [Optional] Postprocessor object to apply to model predictions before metrics computation

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported. Currently, verbose works only with notebook_logs_stdout = False.




