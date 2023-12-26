# ACSTravelTimeDataset

Dataset class for the travel time task from the folktables dataset. Target: binary classification, predict whether a working adult has a travel time to work of greater than 20 minutes. Source of the dataset: https://github.com/socialfoundations/folktables



## Parameters

- **state**

    State in the US for which to get the data. All states in the US are available.

- **year**

    Year for which to get the data. Five different years of data collection are available: 2014–2018 inclusive.

- **root_dir** – defaults to `None`

    Path to the root directory where to store the extracted dataset or where it is stored.

- **with_nulls** – defaults to `False`

    Whether to keep nulls in the dataset or replace them on the new categorical class. Default: False.




## Methods

???- note "update_X_data"

    To save simulated nulls

    **Parameters**

    - **X_data**    
    
