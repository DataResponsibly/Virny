import os
import pandas as pd


def create_average_metrics_df(dataset_name, model_names, metrics_path):
    results_filenames = [filename for filename in os.listdir(metrics_path)]
    models_average_results_dct = dict()
    for model_name in model_names:
        model_results_filenames = [filename for filename in results_filenames if 'Average_Metrics' not in filename
                                   and dataset_name in filename
                                   and model_name in filename]

        if len(model_results_filenames) == 0:
            continue

        model_results_dfs = []
        for model_results_filename in model_results_filenames:
            model_results_df = pd.read_csv(f'{metrics_path}/{model_results_filename}')
            model_results_df.set_index('index', inplace = True)
            model_results_dfs.append(model_results_df)

        model_average_results_df = None
        for model_results_df in model_results_dfs:
            if model_average_results_df is None:
                model_average_results_df = model_results_df
            else:
                model_average_results_df += model_results_df

        model_average_results_df = model_average_results_df / len(model_results_dfs)
        models_average_results_dct[model_name] = model_average_results_df

        filename = f'Average_Metrics_{dataset_name}_{model_name}.csv'
        model_average_results_df.reset_index().to_csv(f'{metrics_path}/{filename}', index=False)
        print(f'File with average metrics for {model_name} is created')

    return models_average_results_dct
