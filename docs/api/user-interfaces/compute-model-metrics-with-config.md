# compute_model_metrics_with_config

Compute subgroup metrics for the base model. Arguments are defined as an input config object. Save results in `save_results_dir_path` folder.

Return a dataframe of model metrics.

## Parameters

- **base_model**

    Base model for metrics computation

- **model_name** (*str*)

    Model name to name a result file with metrics

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **save_results** (*bool*) – defaults to `True`

    [Optional] If to save result metrics in a file

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported.




