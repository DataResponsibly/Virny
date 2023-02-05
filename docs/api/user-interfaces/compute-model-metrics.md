# compute_model_metrics

Compute subgroup metrics for the base model. Save results in `save_results_dir_path` folder.

Return a dataframe of model metrics.

## Parameters

- **base_model**

    Base model for metrics computation

- **n_estimators** (*int*)

    Number of estimators for bootstrap to compute subgroup variance metrics

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    BaseDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **test_set_fraction** (*float*)

    Fraction of the whole dataset in range [0.0 - 1.0] to create a test set

- **bootstrap_fraction** (*float*)

    Fraction of a train set in range [0.0 - 1.0] to fit models in bootstrap

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **model_seed** (*int*)

    Model seed

- **dataset_name** (*str*)

    Dataset name to name a result file with metrics

- **base_model_name** (*str*)

    Model name to name a result file with metrics

- **save_results** (*bool*) – defaults to `True`

    [Optional] If to save result metrics in a file

- **save_results_dir_path** (*str*) – defaults to `None`

    [Optional] Location where to save result files with metrics

- **debug_mode** (*bool*) – defaults to `False`

    [Optional] Enable or disable extra logs




