# compute_metrics_multiple_runs

Find variance and statistical bias metrics for each model in models_config. Arguments are defined as an input config object. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for multiple runs and sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains test_set_fraction, bootstrap_fraction, dataset_name,  n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **debug_mode** – defaults to `False`

    [Optional] Enable or disable extra logs




