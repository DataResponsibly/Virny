import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns

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


def create_bar_plot_for_model_selection(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                                        metrics_value_range_dct: dict, group: str):
    # Compute the number of models that satisfy the conditions
    models_in_range_df = create_models_in_range_dct(all_subgroup_metrics_per_model_dct, all_group_metrics_per_model_dct,
                                                    metrics_value_range_dct, group)
    # Replace metric groups on their aliases
    metric_name_to_alias_dct = {
        # C1
        'TPR': 'C1',
        'TNR': 'C1',
        'FNR': 'C1',
        'FPR': 'C1',
        'PPV': 'C1',
        'Accuracy': 'C1',
        'F1': 'C1',
        # C2
        'Equalized_Odds_TPR': 'C2',
        'Equalized_Odds_FPR': 'C2',
        'Equalized_Odds_FNR': 'C2',
        'Disparate_Impact': 'C2',
        'Statistical_Parity_Difference': 'C2',
        # C3
        'Std': 'C3',
        'IQR': 'C3',
        'Jitter': 'C3',
        'Label_Stability': 'C3',
        # C4
        'IQR_Parity': 'C4',
        'Label_Stability_Ratio': 'C4',
        'Std_Parity': 'C4',
        'Std_Ratio': 'C4',
        'Jitter_Parity': 'C4',
    }

    def get_column_alias(metric_group):
        if '&' not in metric_group:
            alias = metric_name_to_alias_dct[metric_group]
        else:
            metrics = metric_group.split('&')
            alias = None
            for idx, metric in enumerate(metrics):
                if idx == 0:
                    alias = metric_name_to_alias_dct[metric]
                else:
                    alias += ' & ' + metric_name_to_alias_dct[metric]

        return alias

    models_in_range_df['Alias'] = models_in_range_df['Metric_Group'].apply(get_column_alias)
    models_in_range_df['Title'] = models_in_range_df['Alias']

    base_font_size = 25
    bar_plot = alt.Chart(models_in_range_df).mark_bar().encode(
        x=alt.X("Title", type="nominal", title='Metric Group', axis=alt.Axis(labelAngle=-30),
                sort=alt.Sort(order='ascending')),
        y=alt.Y("Number_of_Models", title="Number of Models", type="quantitative"),
        color=alt.Color('Model_Name', legend=alt.Legend(title='Model Name'))
    ).configure_axis(
        labelFontSize=base_font_size + 2,
        titleFontSize=base_font_size + 4,
        labelFontWeight='normal',
        titleFontWeight='normal',
        labelLimit=300,
    ).configure_title(
        fontSize=base_font_size + 2
    ).configure_legend(
        titleFontSize=base_font_size + 2,
        labelFontSize=base_font_size,
        symbolStrokeWidth=4,
        labelLimit=300,
        titleLimit=220,
        orient='none',
        legendX=345, legendY=10,
    ).properties(width=650, height=450)

    return bar_plot


def create_models_in_range_dct(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                               metrics_value_range_dct: dict, group: str):
    # Merge subgroup and group metrics for each model and align their columns
    all_metrics_for_all_models_df = pd.DataFrame()
    for model_name in all_subgroup_metrics_per_model_dct.keys():
        group_metrics_per_model_df = all_group_metrics_per_model_dct[model_name][
            (all_group_metrics_per_model_dct[model_name]['Subgroup'] == group)
            ]
        subgroup_metrics_per_model_df = all_subgroup_metrics_per_model_dct[model_name][
            (all_subgroup_metrics_per_model_dct[model_name]['Subgroup'] == 'overall')
            ]
        subgroup_metrics_per_model_df['Subgroup'] = subgroup_metrics_per_model_df['Subgroup']
        aligned_subgroup_metrics_per_model_df = subgroup_metrics_per_model_df[group_metrics_per_model_df.columns]

        combined_metrics_per_model_df = pd.concat([group_metrics_per_model_df, aligned_subgroup_metrics_per_model_df]).reset_index(drop=True)
        all_metrics_for_all_models_df = pd.concat([all_metrics_for_all_models_df, combined_metrics_per_model_df])

    all_metrics_for_all_models_df = all_metrics_for_all_models_df.reset_index(drop=True)
    all_metrics_for_all_models_df = all_metrics_for_all_models_df.drop(['Subgroup'], axis=1)

    # Create new columns based on values in Metric and Value columns
    pivoted_model_metrics_df = all_metrics_for_all_models_df.pivot(columns='Metric', values='Value',
                                                                   index=[col for col in all_metrics_for_all_models_df.columns
                                                                          if col not in ('Metric', 'Value')]).reset_index()

    # Create a pandas condition for filtering based on the input value ranges
    models_in_range_df = pd.DataFrame()
    for idx, (metric_group, value_range) in enumerate(metrics_value_range_dct.items()):
        pd_condition = None
        if '&' not in metric_group:
            min_range_val, max_range_val = value_range
            if max_range_val < min_range_val:
                raise ValueError('The second element in the input range must be greater than the first element, '
                                 'so to be in the following format -- (min_range_val, max_range_val)')
            metric = metric_group
            pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
        else:
            metrics = metric_group.split('&')
            for idx, metric in enumerate(metrics):
                min_range_val, max_range_val = metrics_value_range_dct[metric]
                if max_range_val < min_range_val:
                    raise ValueError('The second element in the input range must be greater than the first element, '
                                     'so to be in the following format -- (min_range_val, max_range_val)')
                if idx == 0:
                    pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
                else:
                    pd_condition &= (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)

        num_satisfied_models_df = pivoted_model_metrics_df[pd_condition]['Model_Name'].value_counts().reset_index()
        num_satisfied_models_df.rename(columns = {'Model_Name': 'Number_of_Models'}, inplace = True)
        num_satisfied_models_df.rename(columns = {'index': 'Model_Name'}, inplace = True)
        num_satisfied_models_df['Metric_Group'] = metric_group
        if idx == 0:
            models_in_range_df = num_satisfied_models_df
        else:
            # Concatenate based on rows
            models_in_range_df = pd.concat([models_in_range_df, num_satisfied_models_df], ignore_index=True, sort=False)

    return models_in_range_df
