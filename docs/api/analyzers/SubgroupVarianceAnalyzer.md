# SubgroupVarianceAnalyzer

Analyzer to compute variance metrics for subgroups.



## Parameters

- **model_setting** (*configs.constants.ModelSetting*)

    Model learning type; a constant from configs.constants.ModelSetting

- **n_estimators** (*int*)

    Number of estimators for bootstrap

- **base_model**

    Initialized base model to analyze

- **base_model_name** (*str*)

    Model name

- **bootstrap_fraction** (*float*)

    [0-1], fraction from train_pd_dataset for fitting an ensemble of base models

- **base_pipeline** (*[custom_classes.GenericPipeline](../../custom_classes/GenericPipeline)*)

    Initialized object of GenericPipeline class

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming




## Methods

???- note "compute_metrics"

    Measure variance metrics for subgroups for the base model. Display variance plots for analysis if needed.  Save results to a .csv file if needed.

    Return averaged bootstrap predictions and a pandas dataframe of variance metrics for subgroups.

    **Parameters**

    - **save_results**     (*bool*)    
    - **result_filename**     (*str*)     – defaults to `None`    
    - **save_dir_path**     (*str*)     – defaults to `None`    
    - **make_plots**     (*bool*)     – defaults to `True`    
    
