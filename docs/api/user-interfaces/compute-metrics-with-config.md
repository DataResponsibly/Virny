# compute_metrics_with_config

Compute stability and accuracy metrics for each model in models_config. Arguments are defined as an input config object. Save results in `save_results_dir_path` folder.

Return a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

## Parameters

- **dataset** (*[custom_classes.BaseFlowDataset](../../custom_classes/BaseFlowDataset)*)

    BaseFlowDataset object that contains all needed attributes like target, features, numerical_columns etc.

- **config**

    Object that contains bootstrap_fraction, dataset_name, n_estimators, sensitive_attributes_dct attributes

- **models_config** (*dict*)

    Dictionary where keys are model names, and values are initialized models

- **save_results_dir_path** (*str*)

    Location where to save result files with metrics

- **postprocessor** – defaults to `None`

    [Optional] Postprocessor object to apply to model predictions before metrics computation

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] True, if models in models_config have a predict_proba method and can return probabilities for predictions,  False, otherwise. Note that if it is set to False, only metrics based on labels (not labels and probabilities) will be computed.  Ignored when a postprocessor is not None, and set to False in this case.

- **notebook_logs_stdout** (*bool*) – defaults to `False`

    [Optional] True, if this interface was execute in a Jupyter notebook,  False, otherwise.

- **verbose** (*int*) – defaults to `0`

    [Optional] Level of logs printing. The greater level provides more logs.     As for now, 0, 1, 2 levels are supported. Currently, verbose works only with notebook_logs_stdout = False.




