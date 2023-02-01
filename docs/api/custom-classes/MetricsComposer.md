# MetricsComposer

Composer class that combines different metrics to create new ones such as 'Disparate_Impact' or 'Accuracy_Parity'



## Parameters

- **models_metrics_dct** (*dict*)

    Dictionary where keys are model names and values are dataframes of subgroups metrics for each model

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes




## Methods

???- note "compose_metrics"

    Compose subgroup metrics from self.model_metrics_df.

    Return a dictionary of composed metrics.

    
