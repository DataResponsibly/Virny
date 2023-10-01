import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt

from virny.utils.common_helpers import check_substring_in_list


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


def create_sorted_matrix_by_rank(model_metrics_matrix) -> np.array:
    models_distances_matrix = model_metrics_matrix.copy(deep=True).T
    metric_names = models_distances_matrix.columns
    for metric_name in metric_names:
        if 'impact' in metric_name.lower() or 'ratio' in metric_name.lower():
            models_distances_matrix[metric_name] = models_distances_matrix[metric_name] - 1
        models_distances_matrix[metric_name] = models_distances_matrix[metric_name].abs()

    models_distances_matrix = models_distances_matrix.T
    sorted_matrix_by_rank = np.argsort(np.argsort(models_distances_matrix, axis=1), axis=1)
    return sorted_matrix_by_rank


def create_subgroup_sorted_matrix_by_rank(model_metrics_matrix) -> np.array:
    models_distances_matrix = model_metrics_matrix.copy(deep=True).T
    metric_names = models_distances_matrix.columns
    for metric_name in metric_names:
        if check_substring_in_list(metric_name, ['TPR', 'TNR', 'PPV', 'Accuracy', 'F1', 'Label_Stability']):
            # Cast a metric to a case when the closer value to zero is the better
            models_distances_matrix[metric_name] = 1 - models_distances_matrix[metric_name]
        models_distances_matrix[metric_name] = models_distances_matrix[metric_name].abs()

    models_distances_matrix = models_distances_matrix.T
    sorted_matrix_by_rank = np.argsort(np.argsort(models_distances_matrix, axis=1), axis=1)
    return sorted_matrix_by_rank


def create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank, num_models: int):
    """
    This heatmap includes group fairness and stability metrics and defined models.
    Using it, you can visually compare the models across defined group metrics. On this plot,
    colors display ranks where 1 is the best model for the metric. These ranks are conditioned
    on difference or ratio operations used to create these group metrics:

    1) if the metric is created based on the difference operation, closer values to zero have ranks that are closer to the first rank

    2) if the metric is created based on the ratio operation, closer values to one have ranks that are closer to the first rank

    Parameters
    ----------
    model_metrics_matrix
        Matrix of model metrics values where indexes are group metric names and columns are model names
    sorted_matrix_by_rank
        Matrix of model ranks per metric where indexes are group metric names and columns are model names
    num_models
        Number of models to visualize

    """
    font_increase = 2
    matrix_width = 20
    matrix_height = model_metrics_matrix.shape[0] // 2
    fig = plt.figure(figsize=(matrix_width, matrix_height))
    rank_colors = sns.color_palette("coolwarm", n_colors=num_models).as_hex()[::-1]
    ax = sns.heatmap(sorted_matrix_by_rank, annot=model_metrics_matrix, cmap=rank_colors,
                     fmt='', annot_kws={'color': 'black', 'alpha': 0.7, 'fontsize': 16 + font_increase})
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=16 + font_increase)
    fig.subplots_adjust(left=0.25, right=1., top=0.9)

    cbar = ax.collections[0].colorbar
    model_ranks = [idx for idx in range(num_models)]
    cbar.set_ticks([float(idx) for idx in model_ranks])
    tick_labels = [str(idx + 1) for idx in model_ranks]
    tick_labels[0] = tick_labels[0] + ', best'
    tick_labels[-1] = tick_labels[-1] + ', worst'
    cbar.set_ticklabels(tick_labels, fontsize=16 + font_increase)
    cbar.set_label('Model Ranks', fontsize=18 + font_increase)

    return fig, ax
