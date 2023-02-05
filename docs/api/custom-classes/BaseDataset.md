# BaseDataset

Base dataset class that is used as input for user interfaces. Inherit from it to create a dataset class for your dataset file.



## Parameters

- **pandas_df** (*pandas.core.frame.DataFrame*)

    Full dataset in a pandas dataframe format

- **features** (*list*)

    List of feature column names

- **target** (*str*)

    Name of the target column name

- **numerical_columns** (*list*)

    List of numerical column names

- **categorical_columns** (*list*)

    List of categorical column names

- **X_data** (*pandas.core.frame.DataFrame*) – defaults to `None`

    [Optional] Dataframe of features

- **y_data** (*pandas.core.frame.DataFrame*) – defaults to `None`

    [Optional] Dataframe with a target column

- **columns_with_nulls** (*list*) – defaults to `None`

    [Optional] List of column names that contains nulls




