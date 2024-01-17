# MetricsComposer

Metric Composer class that combines different subgroup metrics to create disparity metrics  such as 'Disparate_Impact' or 'Accuracy_Parity'.

Definitions of the disparity metrics could be observed in the __init__ method of the Metric Composer:  https://github.com/DataResponsibly/Virny/blob/main/virny/custom_classes/metrics_composer.py

## Parameters

- **models_metrics_dct** (*dict*)

    Dictionary where keys are model names and values are dataframes of subgroups metrics for each model

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes




## Methods

???- note "compose_metrics"

    Compose subgroup metrics from self.model_metrics_df.

    Return a dictionary of composed metrics.

    
