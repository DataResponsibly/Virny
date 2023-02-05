# compute_model_metrics_with_config

Compute subgroup metrics for the base model. Arguments are defined as an input config object. Save results in `save_results_dir_path` folder.

Return a dataframe of model metrics.

## Parameters

- **base_model**

    Base model for metrics computation

- **model_name** (*str*)

    Model name to name a result file with metrics

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains test_set_fraction, bootstrap_fraction, dataset_name,  n_estimators, sensitive_attributes_dct attributes

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **model_seed** (*int*) – defaults to `None`

    [Optional] Model seed

- **save_results** (*bool*) – defaults to `True`

    [Optional] If to save result metrics in a file

- **debug_mode** (*bool*) – defaults to `False`

    [Optional] Enable or disable extra logs




