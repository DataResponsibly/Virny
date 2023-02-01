# MetricsVisualizer

Class to create useful visualizations of models metrics.



## Parameters

- **models_metrics_dct** (*dict*)

    Dictionary where keys are model names and values are dataframes of subgroups metrics for each model

- **models_composed_metrics_df** (*pandas.core.frame.DataFrame*)

    Dataframe of all models composed metrics

- **dataset_name** (*str*)

    Name of a dataset that was included in metrics filenames and was used for the metrics computation

- **model_names** (*list*)

    Metrics for what model names to visualize

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these attributes




## Methods

???- note "create_bias_variance_interactive_bar_chart"

???- note "create_boxes_and_whiskers_for_models_multiple_runs"

    Create a boxes-and-whiskers plot for subgroup metrics after multiple runs

    **Parameters**

    - **metrics_lst**     (*list*)    
    
???- note "create_html_report"

???- note "create_models_metrics_bar_chart"

???- note "visualize_overall_metrics"

