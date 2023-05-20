# compute_metrics_multiple_runs

Compute stability and accuracy metrics for each model in models_config. Arguments are defined as an input config object. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for multiple runs and sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported.




