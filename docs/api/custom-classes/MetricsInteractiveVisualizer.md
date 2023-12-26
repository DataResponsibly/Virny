# MetricsInteractiveVisualizer

Class to create an interactive web app based on models metrics.



## Parameters

- **X_data** (*pandas.core.frame.DataFrame*)

    An original features dataframe

- **y_data** (*pandas.core.frame.DataFrame*)

    An original target column pandas series

- **model_metrics**

    A dictionary or a dataframe where keys are model names and values are dataframes of subgroup metrics for each model

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these attributes




## Methods

???- note "create_web_app"

    Build an interactive web application.

    **Parameters**

    - **start_app**     – defaults to `True`    
    
