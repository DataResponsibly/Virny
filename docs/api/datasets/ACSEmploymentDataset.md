# ACSEmploymentDataset

Dataset class for the employment task from the folktables dataset. Target: binary classification, predict if a person is employed. Source of the dataset: https://github.com/socialfoundations/folktables



## Parameters

- **state**

    State in the US for which to get the data. All states in the US are available.

- **year**

    Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.

- **root_dir** – defaults to `None`

    Path to the root directory where to store the extracted dataset or where it is stored.

- **with_nulls** – defaults to `False`

    Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.

- **with_filter** – defaults to `True`

    Whether to use a folktables filter for this task. Default: True.

- **optimize** – defaults to `True`

    Whether to optimize the dataset size by downcasting categorical columns. Default: True.

- **subsample_size** (*int*) – defaults to `None`

    Subsample size to create based on the input dataset.

- **subsample_seed** (*int*) – defaults to `None`

    Seed for sampling using the sample() method from pandas.




## Methods

???- note "update_X_data"

    To save simulated nulls

    **Parameters**

    - **X_data**    
    
