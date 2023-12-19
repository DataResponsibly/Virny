# validate_config

Validate parameters types and values in config yaml file.

Extra details: * config_obj.model_setting is an optional argument that defines a type of models to use to compute fairness and stability metrics. Currently, only batch models are supported. Default: 'batch'. 

* config_obj.computation_mode is an optional argument that defines a non-default mode for metrics computation.   Currently, only 'error_analysis' mode is supported.

## Parameters

- **config_obj**

    Object with parameters defined in a yaml file




