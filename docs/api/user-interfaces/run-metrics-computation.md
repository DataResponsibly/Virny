# run_metrics_computation

Find variance and statistical bias metrics for each model in models_config. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    Dataset object that contains all needed attributes like target, features, numerical_columns etc.

- **test_set_fraction** (*float*)

    Fraction of the whole dataset in range [0.0 - 1.0] to create a test set

- **bootstrap_fraction** (*float*)

    Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap

- **dataset_name** (*str*)

    Dataset name to name a result file with metrics

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **n_estimators** (*int*)

    Number of estimators for bootstrap to compute subgroup variance metrics

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **model_seed** (*int*) – defaults to `None`

    [Optional] Model seed

- **save_results** (*bool*) – defaults to `True`

    [Optional] If to save result metrics in a file

- **save_results_dir_path** (*str*) – defaults to `None`

    [Optional] Location where to save result files with metrics

- **debug_mode** (*bool*) – defaults to `False`

    [Optional] Enable or disable extra logs




