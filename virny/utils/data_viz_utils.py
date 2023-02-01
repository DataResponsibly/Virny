import os
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def plot_generic(x, y, xlabel, ylabel, x_lim, y_lim, plot_title):
    sns.set_style("darkgrid")
    plt.figure(figsize=(20,10))
    plt.scatter(x, y)
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(plot_title, fontsize=20)
    plt.show()


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


def visualize_fairness_metrics_for_prediction_metric(models_average_results_dct, x_metric, y_metrics: list):
    sns.set_style("darkgrid")
    x_lim = 0.5
    y_lim = 0.22
    priv_dis_pairs = [('SEX_RAC1P_priv', 'SEX_RAC1P_dis'),
                      ('SEX_priv', 'SEX_dis'),
                      ('RAC1P_priv', 'RAC1P_dis')]
    for y_metric in y_metrics:
        for fairness_metric_priv, fairness_metric_dis in priv_dis_pairs:
            display_fairness_plot(models_average_results_dct, x_metric, y_metric,
                                  fairness_metric_priv, fairness_metric_dis, x_lim, y_lim)


def display_fairness_plot(models_average_results_dct, x_metric, y_metric,
                          fairness_metric_priv, fairness_metric_dis, x_lim, y_lim):
    fig, ax = plt.subplots()
    set_size(15, 8, ax)

    # List of all markers -- https://matplotlib.org/stable/api/markers_api.html
    markers = ['o', '*', '|', '<', '>', '^', 'v', '1', 's', 'x', 'D', 'P', 'H']
    model_names = models_average_results_dct.keys()
    shapes = []
    for idx, model_name in enumerate(model_names):
        x_val = abs(models_average_results_dct[model_name][fairness_metric_priv].loc[x_metric] - \
                    models_average_results_dct[model_name][fairness_metric_dis].loc[x_metric])
        y_val = abs(models_average_results_dct[model_name][fairness_metric_priv].loc[y_metric] - \
                    models_average_results_dct[model_name][fairness_metric_dis].loc[y_metric])
        a = ax.scatter(x_val, y_val, marker=markers[idx], s=100)
        shapes.append(a)

    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel(f'{x_metric} Difference')
    plt.ylabel(f'{y_metric} Difference')
    plt.xlim(-0.01, x_lim)
    plt.ylim(-0.01, y_lim)
    plt.title(f'{fairness_metric_priv}-{fairness_metric_dis} difference for {x_metric} and {y_metric}', fontsize=20)
    ax.legend(shapes, model_names, fontsize=12, title='Markers')

    plt.show()
