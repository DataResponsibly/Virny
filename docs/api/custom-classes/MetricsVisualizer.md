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

???- note "create_bias_variance_interactive_bar_chart"

    This interactive bar chart includes all groups, all composed group bias and variance metrics,  and all defined models. Using it, you can select any pair of group bias and variance metrics and   compare them across all groups and models. Since this plot is interactive, it saves a lot of space for other plots.    Also, it could be more convenient to compare individual group bias and variance metrics using the interactive mode.

    
???- note "create_boxes_and_whiskers_for_models_multiple_runs"

    This boxes and whiskers plot is based on overall subgroup bias and variance metrics for all defined models and results after all runs. Using it, you can see combined information on one plot that includes different models,  subgroup metrics, and results after multiple runs.

    **Parameters**

    - **metrics_lst**     (*list*)    
    
???- note "create_html_report"

    Create Statistical Bias and Variance Report depending on report type. It includes visualizations and helpful details to them.

    **Parameters**

    - **report_type**     (*virny.configs.constants.ReportType*)    
    - **report_save_path**     (*str*)    
    
???- note "create_model_rank_heatmap"

    This heatmap includes all group bias and variance metrics and all defined models. Using it, you can visually compare all models across all group metrics. On this plot, colors display ranks where 1 is the best model for the metric. These ranks are conditioned on difference or ratio operations used to create these group metrics:

    1) if the metric is created based on the difference operation, closer values to zero have ranks that are closer to the first rank  2) if the metric is created based on the ratio operation, closer values to one have ranks that are closer to the first rank

    **Parameters**

    - **model_metrics_matrix**    
    - **sorted_matrix_by_rank**    
    - **num_models**     (*int*)    
    
???- note "create_model_rank_heatmaps"

    Create model rank and total model rank heatmaps.

    **Parameters**

    - **metrics_lst**     (*list*)    
    - **groups_lst**    
    
???- note "create_models_metrics_bar_chart"

???- note "create_overall_metrics_bar_char"

    This bar chart includes all defined models and all overall subgroup bias and variance metrics, which are averaged across multiple runs. Using it, you can compare all models for each subgroup bias or variance metric. This comparison also includes reversed metrics, in which values closer to zero are better since straight and reversed metrics in this plot are converted to the same format -- values closer to one are better.

    **Parameters**

    - **metrics_names**     (*list*)    
    - **reversed_metrics_names**     (*list*)     – defaults to `None`    
    - **metrics_title**     (*str*)     – defaults to `Overall Metrics`    
    
???- note "create_total_model_rank_heatmap"

    This heatmap includes all defined models and sums of their bias and variance ranks. On this plot, colors display sums of ranks for one model. If the sum is smaller, the model has better bias or variance characteristics than other models. Using this plot, you can visually compare all models for bias and variance characteristics.

    **Parameters**

    - **sorted_matrix_by_rank**    
    - **num_models**    
    
