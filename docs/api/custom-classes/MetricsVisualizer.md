# MetricsVisualizer

Class to create useful visualizations of models metrics.



## Parameters

- **models_metrics_dct** (*dict*)

    Dictionary where keys are model names and values are dataframes of subgroup metrics for each model

- **models_composed_metrics_df** (*pandas.core.frame.DataFrame*)

    Dataframe of all model composed metrics

- **dataset_name** (*str*)

    Name of a dataset that was included in metric filenames and was used for the metrics computation

- **model_names** (*list*)

    Metrics for what model names to visualize

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these attributes




## Methods

???- note "create_boxes_and_whiskers_for_models_multiple_runs"

    This boxes and whiskers plot is based on overall subgroup error and stability metrics for all defined models and results after all runs. Using it, you can see combined information on one plot that includes different models,  subgroup metrics, and results after multiple runs.

    **Parameters**

    - **metrics_lst**     (*list*)    
    
???- note "create_disparity_metric_heatmap"

    Create a heatmap for disparity metrics.

    **Parameters**

    - **model_names**     (*list*)    
        Metrics for what model names to visualize
    - **metrics_lst**     (*list*)    
    - **groups_lst**     (*list*)    
    - **tolerance**     (*float*)     – defaults to `0.001`    
    - **figsize_scale**     (*tuple*)     – defaults to `(0.7, 0.5)`    
    - **font_increase**     (*int*)     – defaults to `-3`    
    
???- note "create_overall_metric_heatmap"

    Create a heatmap for overall metrics.

    **Parameters**

    - **model_names**     (*list*)    
        Metrics for what model names to visualize
    - **metrics_lst**     (*list*)    
    - **tolerance**     (*float*)     – defaults to `0.001`    
    - **figsize_scale**     (*tuple*)     – defaults to `(0.7, 0.5)`    
    - **font_increase**     (*int*)     – defaults to `-3`    
    
???- note "create_overall_metrics_bar_char"

    This bar chart includes all defined models and all overall subgroup error and stability metrics, which are averaged across multiple runs. Using it, you can compare all models for each subgroup error or stability metric. This comparison also includes reversed metrics, in which values closer to zero are better since straight and reversed metrics in this plot are converted to the same format -- values closer to one are better.

    **Parameters**

    - **metric_names**     (*list*)    
    - **plot_title**     (*str*)     – defaults to `Overall Metrics`    
    
