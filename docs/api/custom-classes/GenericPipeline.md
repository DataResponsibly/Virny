# GenericPipeline

Custom class that is used in many internal functions for convenience. It contains general attributes for different metrics computation pipelines and useful custom methods.



## Parameters

- **dataset** (*[custom_classes.BaseDataset](../../custom_classes/BaseDataset)*)

    Instance of the dataset class inherited from BaseDataset

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes

- **base_model** – defaults to `None`

    Instance of a base model for analyzes

- **metric_names** (*list*) – defaults to `None`

    Names of metrics to compute for the base model




## Methods

???- note "construct_pipeline"

???- note "create_preprocessed_train_test_split"

???- note "create_train_test_val_split"

???- note "create_train_test_val_split_balanced"

???- note "fit_model_batch"

???- note "fit_model_incremental"

???- note "set_pipeline"

???- note "set_train_test_val_data_by_index"

