import math
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns

from matplotlib import pyplot as plt
from altair.utils.schemapi import Undefined

from virny.configs.constants import *
from virny.utils.common_helpers import check_substring_in_list


def rank_with_tolerance(pd_series: pd.Series, tolerance: float = 0.001):
    """
    Rank a pandas series with defined tolerance.
    Ref: https://stackoverflow.com/questions/72956450/pandas-ranking-with-tolerance

    Parameters
    ----------
    pd_series
        A pandas series to rank
    tolerance
        A float value for ranking

    Returns
    -------
    A pandas series with dense ranks for the input pd series.

    """
    sorted_vals = sorted(pd_series.tolist())

    # Create a dictionary with bin constraints
    bin_constraints_dct = dict()
    for i in range(len(sorted_vals)):
        val = sorted_vals[i]
        rank = i + 1
        bin_constraints_dct[rank] = [round(val - tolerance, 3), round(val + tolerance, 3)]

    # Assign ranks for each pandas series value
    assigned_ranks_dct = dict()
    for i in range(len(sorted_vals)):
        val = sorted_vals[i]
        max_rank = i + 1
        actual_rank = None
        for rank in bin_constraints_dct.keys():
            min_limit, max_limit = bin_constraints_dct[rank]
            if min_limit <= val <= max_limit:
                actual_rank = rank
                break

        assigned_ranks_dct[str(round(val, 3))] = actual_rank
        # Dynamically delete constraints from bin_constraints_dct to keep values in the same bin with tolerance
        if actual_rank != max_rank:
            del bin_constraints_dct[max_rank]

    def get_rank_with_tolerance(val):
        return assigned_ranks_dct[str(round(val, 3))]

    return pd_series.apply(get_rank_with_tolerance).rank(method='dense')


def compute_proportions(protected_groups, X_data):
    subgroup_proportions_dct = {'overall': {'percentage': 1.0, 'num_rows': X_data.shape[0]}}
    for col_name in protected_groups.keys():
        proportion = protected_groups[col_name].shape[0] / X_data.shape[0]
        subgroup_proportions_dct[col_name] = {'percentage': proportion, 'num_rows': protected_groups[col_name].shape[0]}

    return subgroup_proportions_dct


def compute_base_rates(protected_groups, y_data):
    overall_base_rate = y_data[y_data == 1].shape[0] / y_data.shape[0]
    subgroup_base_rates_dct = {'overall': {'percentage': overall_base_rate, 'num_rows': y_data[y_data == 1].shape[0]}}
    for col_name in protected_groups.keys():
        filtered_df = y_data.iloc[protected_groups[col_name].index].copy(deep=True)
        base_rate = filtered_df[filtered_df == 1].shape[0] / filtered_df.shape[0]
        subgroup_base_rates_dct[col_name] = {'percentage': base_rate, 'num_rows': filtered_df[filtered_df == 1].shape[0]}

    return subgroup_base_rates_dct


def create_sorted_matrix_by_rank(model_metrics_matrix, tolerance) -> np.array:
    models_distances_matrix = model_metrics_matrix.copy(deep=True).T
    metric_names = models_distances_matrix.columns
    for metric_name in metric_names:
        if 'impact' in metric_name.lower() or 'ratio' in metric_name.lower():
            models_distances_matrix[metric_name] = models_distances_matrix[metric_name] - 1
        models_distances_matrix[metric_name] = models_distances_matrix[metric_name].abs()

    models_distances_matrix = models_distances_matrix.T
    models_distances_df = pd.DataFrame(models_distances_matrix)
    sorted_matrix_by_rank = models_distances_df.apply(
        lambda row : rank_with_tolerance(row, tolerance), axis = 1
    )

    return sorted_matrix_by_rank


def create_subgroup_sorted_matrix_by_rank(model_metrics_matrix, tolerance) -> np.array:
    models_distances_matrix = model_metrics_matrix.copy(deep=True).T
    metric_names = models_distances_matrix.columns
    for metric_name in metric_names:
        if check_substring_in_list(metric_name, ['TPR', 'TNR', 'PPV', 'Accuracy', 'F1', 'Label_Stability', 'Positive-Rate']):
            # Cast a metric to a case when the closer value to one is the better
            models_distances_matrix[metric_name] = 1 - models_distances_matrix[metric_name]
        models_distances_matrix[metric_name] = models_distances_matrix[metric_name].abs()

    models_distances_matrix = models_distances_matrix.T
    models_distances_df = pd.DataFrame(models_distances_matrix)
    sorted_matrix_by_rank = models_distances_df.apply(
        lambda row : rank_with_tolerance(row, tolerance), axis = 1
    )

    return sorted_matrix_by_rank


def create_col_facet_bar_chart(df, x_col, y_col, facet_column_name, text_labels_column, x_sort_by_lst=Undefined,
                               facet_sort_by_lst=Undefined, color_legend_title=Undefined, facet_title=Undefined):
    num_facets = len(df[facet_column_name].unique())
    max_y_axis_limit = df[y_col].max()
    base_font_size = 16

    # Set dynamic variables that adapt to the number of defined groups
    dynamic_facet_width = 100
    dynamic_label_angle = -20
    dynamic_font_size = base_font_size
    dynamic_top_padding = 40
    dynamic_legend_y_padding = -140
    if num_facets > 4 * 2 + 1 and num_facets <= 6 * 2 + 1:
        dynamic_facet_width = 75
        dynamic_label_angle = -25
        dynamic_font_size -= 2
        dynamic_top_padding = 40
        dynamic_legend_y_padding = -160
    elif num_facets > 6 * 2 + 1:
        dynamic_facet_width = 50
        dynamic_label_angle = -45
        dynamic_font_size -= 4
        dynamic_top_padding = 50
        dynamic_legend_y_padding = -200

    bar_chart = (
        alt.Chart().mark_bar().encode(
            alt.X(f'{x_col}:N', axis=None, sort=x_sort_by_lst),
            alt.Y(f'{y_col}:Q', axis=alt.Axis(grid=True), title='', scale=alt.Scale(domain=[0, max_y_axis_limit])),
            alt.Color(f'{x_col}:N',
                      scale=alt.Scale(scheme="tableau20"),
                      sort=x_sort_by_lst,
                      legend=alt.Legend(title=color_legend_title,
                                        labelFontSize=base_font_size,
                                        titleFontSize=base_font_size + 2,
                                        orient='none',
                                        legendX=0, legendY=dynamic_legend_y_padding,
                                        direction='horizontal'))
        )
    )

    text_labels = (
        bar_chart.mark_text(
            baseline='middle',
            fontSize=dynamic_font_size,
            dy=-10
        ).encode(
            text=alt.Text(f'{text_labels_column}:Q', format=",.2f"),
            color=alt.value("black")
        )
    )

    final_chart = (
        alt.layer(
            bar_chart, text_labels, data=df
        ).properties(
            width=dynamic_facet_width,
            height=500
        ).facet(
            column=alt.Column(f'{facet_column_name}:N', title=facet_title,
                              sort=facet_sort_by_lst, header=alt.Header(labelAngle=dynamic_label_angle,
                                                                        labelAnchor='middle',
                                                                        labelAlign='center',
                                                                        labelPadding=-15))
        ).configure(
            padding={'top':  dynamic_top_padding},
        ).configure_headerColumn(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size, titleFontSize=base_font_size + 2
        )
    )

    return final_chart


def create_row_facet_bar_chart(df, x_col, y_col, facet_column_name, y_sort_by_lst=Undefined,
                               facet_sort_by_lst=Undefined, color_legend_title=Undefined, facet_title=Undefined):
    base_font_size = 16
    bar_chart = (
        alt.Chart().mark_bar().encode(
            alt.Y(f'{y_col}:N', axis=None, sort=y_sort_by_lst),
            alt.X(f'{x_col}:Q', axis=alt.Axis(grid=True), title=''),
            alt.Color(f'{y_col}:N',
                      scale=alt.Scale(scheme="tableau20"),
                      sort=y_sort_by_lst,
                      legend=alt.Legend(title=color_legend_title,
                                        labelFontSize=base_font_size,
                                        titleFontSize=base_font_size + 2,
                                        orient='top'))
        )
    )

    text_labels = (
        bar_chart.mark_text(
            baseline='middle',
            fontSize=base_font_size,
            dx=10
        ).encode(
            text=alt.Text('Value:Q', format=",.3f"),
            color=alt.value("black")
        )
    )

    final_chart = (
        alt.layer(
            bar_chart, text_labels, data=df
        ).properties(
            width=500,
            height=100
        ).facet(
            row=alt.Row(f'{facet_column_name}:N', title=facet_title, sort=facet_sort_by_lst)
        ).configure(
            padding={'top':  33},
        ).configure_headerRow(
            labelFontSize=base_font_size,
            titleFontSize=base_font_size + 2
        ).configure_axis(
            labelFontSize=base_font_size, titleFontSize=base_font_size + 2
        )
    )

    return final_chart


def create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank,
                                            figsize_scale: tuple = (1.0, 1.0), font_increase: int = 4):
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
    figsize_scale
        [Optional] Scale factors for a heatmap size. The first element is a scale factor for a plot width, the second one is for height.
    font_increase
        [Optional] An integer to increase or decrease the plot font.

    """
    matrix_width = 20
    matrix_height = model_metrics_matrix.shape[0] if model_metrics_matrix.shape[0] >= 3 \
        else model_metrics_matrix.shape[0] * 2.5
    num_ranks = int(sorted_matrix_by_rank.values.max())

    fig = plt.figure(figsize=(matrix_width * figsize_scale[0], matrix_height * figsize_scale[1]))
    # Set a green color when there is only one rank
    if num_ranks == 1:
        rank_colors = sns.diverging_palette(145, 13, s=75, l=70, n=num_ranks).as_hex()
    else:
        rank_colors = sns.diverging_palette(13, 145, s=75, l=70, n=num_ranks).as_hex()
    # Convert ranks to minus ranks (1 --> -1; 4 --> -4) to align rank positions with a color scheme
    reversed_sorted_matrix_by_rank = sorted_matrix_by_rank * -1
    ax = sns.heatmap(reversed_sorted_matrix_by_rank, annot=model_metrics_matrix.round(3), cmap=rank_colors,
                     fmt='', annot_kws={'color': 'black', 'alpha': 0.7, 'fontsize': 16 + font_increase})
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=10)
    ax.tick_params(labelsize=16 + font_increase)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    fig.tight_layout()

    cbar = ax.collections[0].colorbar
    model_ranks = [idx + 1 for idx in range(num_ranks)]
    cbar.set_ticks([float(idx) * -1 for idx in model_ranks])
    tick_labels = ['' for _ in model_ranks]
    if len(tick_labels) > 1:
        tick_labels[0] = 'Best'
        tick_labels[-1] = 'Worst'
    cbar.set_ticklabels(tick_labels, fontsize=16 + font_increase)

    return fig, ax


def create_model_performance_summary_visualization(main_matrix, matrix_for_colors):
    font_increase = 6
    matrix_width = 20
    matrix_height = main_matrix.shape[0] if main_matrix.shape[0] >= 3 else main_matrix.shape[0] * 2.5

    fig = plt.figure(figsize=(matrix_width, matrix_height))
    ax = sns.heatmap(matrix_for_colors, annot=main_matrix.round(3),
                     cmap=["#EE8367", "#58D68D"], # [red, green]
                     fmt='', linewidths=1.0,
                     vmin=0, vmax=1,
                     cbar_kws={"ticks":[0, 1]},
                     annot_kws={'color': 'black', 'alpha': 0.7, 'fontsize': 10 + font_increase})
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(labelsize=10 + font_increase)
    fig.subplots_adjust(left=0.2, top=0.7)

    cbar = ax.collections[0].colorbar
    tick_labels = ['Failed', 'Passed']
    cbar.set_ticks([0.25,0.75])
    cbar.set_ticklabels(tick_labels, fontsize=10 + font_increase)

    return fig, ax


def create_flexible_bar_plot_for_model_selection(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                                                 metrics_value_range_dct: dict, group: str, metric_name_to_alias_dct: dict):
    # Compute the number of models that satisfy the conditions
    models_in_range_df, df_with_models_satisfied_all_constraints = (
        create_models_in_range_dct(all_subgroup_metrics_per_model_dct, all_group_metrics_per_model_dct,
                                   metrics_value_range_dct, group))

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

    # Replace metric groups on their aliases
    models_in_range_df['Alias'] = models_in_range_df['Metric_Group'].apply(get_column_alias)
    models_in_range_df['Title'] = models_in_range_df['Alias']

    base_font_size = 14
    bar_plot = alt.Chart(models_in_range_df).mark_bar().encode(
        x=alt.X("Title", type="nominal", title='Metric Group', axis=alt.Axis(labelAngle=-30),
                sort=alt.Sort(order='ascending')),
        y=alt.Y("Number_of_Models", title="Number of Models", type="quantitative"),
        color=alt.Color('Model_Type', legend=alt.Legend(title='Model Type'))
    ).configure(padding={'top':  33}
                ).configure_axis(
        labelFontSize=base_font_size + 2,
        titleFontSize=base_font_size + 4,
        labelFontWeight='normal',
        titleFontWeight='normal',
        labelLimit=300,
        tickMinStep=1,
    ).configure_title(
        fontSize=base_font_size + 2
    ).configure_legend(
        titleFontSize=base_font_size + 4,
        labelFontSize=base_font_size + 2,
        symbolStrokeWidth=4,
        labelLimit=300,
        titleLimit=220,
    ).properties(width=650, height=450)

    return bar_plot, df_with_models_satisfied_all_constraints


def create_bar_plot_for_model_selection(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                                        metrics_value_range_dct: dict, group: str):
    # Compute the number of models that satisfy the conditions
    models_in_range_df, df_with_models_satisfied_all_constraints = (
        create_models_in_range_dct(all_subgroup_metrics_per_model_dct, all_group_metrics_per_model_dct,
                                   metrics_value_range_dct, group))
    # Replace metric groups on their aliases
    metric_name_to_alias_dct = {
        # C1
        TPR: 'C1',
        TNR: 'C1',
        FNR: 'C1',
        FPR: 'C1',
        PPV: 'C1',
        ACCURACY: 'C1',
        F1: 'C1',
        POSITIVE_RATE: 'C1',
        # C2
        EQUALIZED_ODDS_TPR: 'C2',
        EQUALIZED_ODDS_FPR: 'C2',
        EQUALIZED_ODDS_FNR: 'C2',
        DISPARATE_IMPACT: 'C2',
        STATISTICAL_PARITY_DIFFERENCE: 'C2',
        # C3
        STD: 'C3',
        IQR: 'C3',
        JITTER: 'C3',
        LABEL_STABILITY: 'C3',
        # C4
        IQR_DIFFERENCE: 'C4',
        LABEL_STABILITY_RATIO: 'C4',
        LABEL_STABILITY_DIFFERENCE: 'C4',
        STD_DIFFERENCE: 'C4',
        STD_RATIO: 'C4',
        JITTER_DIFFERENCE: 'C4',
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

    base_font_size = 14
    bar_plot = alt.Chart(models_in_range_df).mark_bar().encode(
        x=alt.X("Title", type="nominal", title='Metric Group', axis=alt.Axis(labelAngle=-30),
                sort=alt.Sort(order='ascending')),
        y=alt.Y("Number_of_Models", title="Number of Models", type="quantitative"),
        color=alt.Color('Model_Type', legend=alt.Legend(title='Model Type'))
    ).configure(padding={'top':  33}
    ).configure_axis(
        labelFontSize=base_font_size + 2,
        titleFontSize=base_font_size + 4,
        labelFontWeight='normal',
        titleFontWeight='normal',
        labelLimit=300,
        tickMinStep=1,
    ).configure_title(
        fontSize=base_font_size + 2
    ).configure_legend(
        titleFontSize=base_font_size + 4,
        labelFontSize=base_font_size + 2,
        symbolStrokeWidth=4,
        labelLimit=300,
        titleLimit=220,
    ).properties(width=650, height=450)

    return bar_plot, df_with_models_satisfied_all_constraints


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
    # Create a Model_Type column to count the number of models that satisfied the constraints based on their model types
    pivoted_model_metrics_df['Model_Type'] = pivoted_model_metrics_df['Model_Name'].str.split('__', expand=True)[0]
    model_types = pivoted_model_metrics_df['Model_Type'].unique()

    # Create a pandas condition for filtering based on the input value ranges
    models_in_range_df = pd.DataFrame()
    df_with_models_satisfied_all_constraints = pd.DataFrame()
    for idx, (metric_group, value_range) in enumerate(metrics_value_range_dct.items()):
        pd_condition = None
        if '&' not in metric_group:
            min_range_val, max_range_val = value_range
            if max_range_val < min_range_val:
                raise ValueError('The second value in the input range must be greater than the first value, '
                                 'so to be in the following format -- (min_range_val, max_range_val)')
            metric = metric_group
            pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
        else:
            metrics = metric_group.split('&')
            for idx, metric in enumerate(metrics):
                min_range_val, max_range_val = metrics_value_range_dct[metric]
                if max_range_val < min_range_val:
                    raise ValueError('The second value in the input range must be greater than the first value, '
                                     'so to be in the following format -- (min_range_val, max_range_val)')
                if idx == 0:
                    pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
                else:
                    pd_condition &= (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)

        num_satisfied_models_df = pivoted_model_metrics_df[pd_condition]['Model_Type'].value_counts().reset_index()
        num_satisfied_models_df.rename(columns = {'Model_Type': 'Number_of_Models'}, inplace = True)
        num_satisfied_models_df.rename(columns = {'index': 'Model_Type'}, inplace = True)
        # If a constraint for a metric group is not satisfied, add zeros for all model names
        if num_satisfied_models_df.shape[0] == 0:
            num_satisfied_models_df = pd.DataFrame({'Model_Type': model_types,
                                                    'Number_of_Models': [0] * len(model_types)})

        num_satisfied_models_df['Metric_Group'] = metric_group
        if idx == 0:
            models_in_range_df = num_satisfied_models_df
        else:
            # Concatenate based on rows
            models_in_range_df = pd.concat([models_in_range_df, num_satisfied_models_df], ignore_index=True, sort=False)

        if metric_group.count('&') == 3:
            df_with_models_satisfied_all_constraints = pivoted_model_metrics_df[pd_condition][['Model_Type', 'Model_Name']]

    return models_in_range_df, df_with_models_satisfied_all_constraints
