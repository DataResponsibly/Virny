# preprocess_dataset

Preprocess an input dataset using sklearn ColumnTransformer. Split the dataset on train and test using test_set_fraction.  Create an instance of BaseFlowDataset.



## Parameters

- **data_loader** (*virny.datasets.base.BaseDataLoader*)

    Instance of BaseDataLoader that contains a target, numerical, and categorical columns.

- **column_transformer** (*sklearn.compose._column_transformer.ColumnTransformer*)

    Instance of sklearn ColumnTransformer to preprocess categorical and numerical columns.

- **test_set_fraction** (*float*)

    Fraction from 0 to 1. Used to split the input dataset on the train and test sets.

- **dataset_split_seed** (*int*)

    Seed for dataset splitting.




