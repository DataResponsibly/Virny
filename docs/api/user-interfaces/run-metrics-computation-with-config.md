# run_metrics_computation_with_config

Find variance and statistical bias metrics for each model in models_config. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    Dataset object that contains all needed attributes like target, features, numerical_columns etc

- **config**

    Object that contains test_set_fraction, bootstrap_fraction, dataset_name,  n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **run_seed** (*int*) – defaults to `None`

    [Optional] Base seed for this run

- **debug_mode** (*bool*) – defaults to `False`

    [Optional] Enable or disable extra logs




