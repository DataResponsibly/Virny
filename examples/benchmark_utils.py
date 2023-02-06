import os
import shutil
import pandas as pd


def convert_transposed_df_to_df(delta_model_metrics_df, subgroup_names, metric_names):
    converted_df_dct = {'Subgroup': []}
    for metric_name in metric_names:
        converted_df_dct[metric_name] = []

    for subgroup_name in subgroup_names:
        for metric_name in metric_names:
            subgroup_metric = delta_model_metrics_df.loc[delta_model_metrics_df['Metric'] == metric_name][subgroup_name].values[0]
            converted_df_dct[metric_name].append(subgroup_metric)

        converted_df_dct['Subgroup'].append(subgroup_name)

    return pd.DataFrame(converted_df_dct)


def populate_benchmark_report(report_df, models_metrics_dct, standard_models_metrics_dct, dataset_name, sensitive_attributes_dct):
    for model_name in models_metrics_dct.keys():
        benchmark_model_metrics_df = models_metrics_dct[model_name]
        standard_model_metrics_df = standard_models_metrics_dct[model_name]

        subgroup_names = ['overall']
        for group_name in sensitive_attributes_dct.keys():
            subgroup_names.append(group_name + '_priv')
            subgroup_names.append(group_name + '_dis')

        delta_model_metrics_df = pd.DataFrame()
        delta_model_metrics_df['Metric'] = standard_model_metrics_df['Metric']
        delta_model_metrics_df[subgroup_names] = \
            standard_model_metrics_df[subgroup_names].sub(benchmark_model_metrics_df[subgroup_names], fill_value=0) * 100
        # Round for better readability
        delta_model_metrics_df[subgroup_names] = delta_model_metrics_df[subgroup_names].round(3)

        metric_names = list(delta_model_metrics_df['Metric'].unique())
        converted_df = convert_transposed_df_to_df(delta_model_metrics_df, subgroup_names, metric_names)
        converted_df['Dataset'] = dataset_name
        converted_df['Model'] = model_name
        converted_df = converted_df.rename(columns={metric_name: 'Delta%_' + metric_name for metric_name in metric_names})
        columns_positions = ['Dataset', 'Model'] + [col for col in converted_df.columns if col not in ['Dataset', 'Model']]
        converted_df = converted_df[columns_positions]

        report_df = pd.concat([report_df, converted_df])

    return report_df.reset_index(drop=True)


def create_averaged_dfs_dict(models_metrics_dct):
    models_average_metrics_dct = dict()
    for model_name in models_metrics_dct.keys():
        columns_to_group = [col for col in models_metrics_dct[model_name].columns
                            if col not in ('Model_Seed', 'Run_Number')]
        models_average_metrics_dct[model_name] = models_metrics_dct[model_name][columns_to_group].groupby(['Metric', 'Model_Name']).mean().reset_index()

    return models_average_metrics_dct


def clear_directory(dir_path):
    if not os.path.exists(dir_path):
        print('Directory does not exist --> skip deletion')
        return

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print('Directory is cleared')
