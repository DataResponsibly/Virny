# BaseFlowDataset

Dataset class with custom train and test splits that is used as input for metrics computation interfaces. Create your dataset class based on this one to use it for metrics computation interfaces.



## Parameters

- **init_sensitive_attrs_df** (*pandas.core.frame.DataFrame*)

    Full train + test non-preprocessed dataset of sensitive attributes with initial indexes.  It is used for creating test groups.

- **X_train_val** (*pandas.core.frame.DataFrame*)

    Train dataframe of features

- **X_test** (*pandas.core.frame.DataFrame*)

    Test dataframe of features

- **y_train_val** (*pandas.core.frame.DataFrame*)

    Train dataframe with a target column

- **y_test** (*pandas.core.frame.DataFrame*)

    Test dataframe with a target column

- **target** (*str*)

    Name of the target column name

- **numerical_columns** (*list*)

    List of numerical column names

- **categorical_columns** (*list*)

    List of categorical column names




