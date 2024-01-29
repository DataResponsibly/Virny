import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from virny.configs.constants import *
from virny.utils.data_viz_utils import (create_sorted_matrix_by_rank, create_subgroup_sorted_matrix_by_rank,
                                        create_model_rank_heatmap_visualization)


class MetricsVisualizer:
    """
    Class to create useful visualizations of models metrics.

    Parameters
    ----------
    models_metrics_dct
        Dictionary where keys are model names and values are dataframes of subgroup metrics for each model
    models_composed_metrics_df
        Dataframe of all model composed metrics
    dataset_name
        Name of a dataset that was included in metric filenames and was used for the metrics computation
    model_names
        Metrics for what model names to visualize
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these attributes

    """
    def __init__(self, models_metrics_dct: dict, models_composed_metrics_df: pd.DataFrame,
                 dataset_name: str, model_names: list, sensitive_attributes_dct: dict):
        sns.set_theme(style="whitegrid")

        self.dataset_name = dataset_name
        self.model_names = model_names
        self.sensitive_attributes_dct = sensitive_attributes_dct

        # Metric names
        self.all_accuracy_metrics = [STATISTICAL_BIAS, TPR, TNR, PPV, FNR, FPR, F1, ACCURACY, POSITIVE_RATE]
        self.all_stability_metrics = [STD, IQR, JITTER, LABEL_STABILITY]
        self.all_uncertainty_metrics = [ALEATORIC_UNCERTAINTY, OVERALL_UNCERTAINTY, EPISTEMIC_UNCERTAINTY]
        self.all_error_disparity_metrics = [EQUALIZED_ODDS_TPR, EQUALIZED_ODDS_TNR, EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_FNR, DISPARATE_IMPACT, STATISTICAL_PARITY_DIFFERENCE, ACCURACY_DIFFERENCE]
        self.all_stability_disparity_metrics = [LABEL_STABILITY_RATIO, LABEL_STABILITY_DIFFERENCE, IQR_DIFFERENCE, STD_DIFFERENCE, STD_RATIO, JITTER_DIFFERENCE]
        self.all_uncertainty_disparity_metrics = [OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO, ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO,
                                                  EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO]

        self.all_overall_metrics = self.all_accuracy_metrics + self.all_stability_metrics + self.all_uncertainty_metrics
        self.all_disparity_metrics = self.all_error_disparity_metrics + self.all_stability_disparity_metrics + self.all_uncertainty_disparity_metrics

        # Create models_average_metrics_dct
        models_average_metrics_dct = dict()
        for model_name in model_names:
            columns_to_group = [col for col in models_metrics_dct[model_name].columns
                                if col not in ('Model_Seed', 'Run_Number')]
            models_average_metrics_dct[model_name] = models_metrics_dct[model_name][columns_to_group].groupby(['Metric', 'Model_Name']).mean().reset_index()

        # Create one average metrics df with all model_dfs
        models_average_metrics_df = pd.DataFrame()
        for model_name in models_average_metrics_dct.keys():
            model_average_metrics_df = models_average_metrics_dct[model_name]
            models_average_metrics_df = pd.concat([models_average_metrics_df, model_average_metrics_df])

        models_average_metrics_df = models_average_metrics_df.reset_index(drop=True)

        # Create one metrics df with all model_dfs
        all_models_metrics_df = pd.DataFrame()
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

        all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)

        self.models_metrics_dct = models_metrics_dct
        self.models_average_metrics_dct = models_average_metrics_dct
        self.all_models_metrics_df = all_models_metrics_df
        self.models_average_metrics_df = models_average_metrics_df
        self.models_composed_metrics_df = models_composed_metrics_df

        self.models_metrics_df = self._align_input_metric_df(all_models_metrics_df, allowed_cols=["Metric", "Model_Name", "overall"],
                                                             sensitive_attrs=list(self.sensitive_attributes_dct.keys()))
        self.melted_model_metrics_df = self.models_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                       var_name="Subgroup",
                                                                       value_name="Value")
        self.sorted_model_metrics_df = self.melted_model_metrics_df.sort_values(by=['Value'])
        self.melted_models_composed_metrics_df = self.models_composed_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                                      var_name="Subgroup",
                                                                                      value_name="Value")
        self.sorted_models_composed_metrics_df = self.melted_models_composed_metrics_df.sort_values(by=['Value'])

    def _align_input_metric_df(self, model_metrics_df: pd.DataFrame, allowed_cols: list, sensitive_attrs: list):
        """
        Filter columns in the input dataframe based on allowed_cols and sensitive_attrs.
        """
        filtered_cols = allowed_cols
        for col in model_metrics_df.columns:
            for sensitive_attr in sensitive_attrs:
                if sensitive_attr in col:
                    filtered_cols.append(col)
                    break

        return model_metrics_df[filtered_cols]

    def __filter_subgroup_metrics_df(self, results: dict, subgroup_metric: str,
                                     selected_metric: str, selected_subgroup: str, defined_model_names: list):
        """
        Find metric values for each model based on metric, subgroup, and model names.
        Add the values to a results dict.
        """
        results[subgroup_metric] = dict()

        # Get distinct sorted model names
        sorted_model_names_arr = self.sorted_model_metrics_df[
            (self.sorted_model_metrics_df.Metric == selected_metric) &
            (self.sorted_model_metrics_df.Subgroup == selected_subgroup)
            ]['Model_Name'].values
        sorted_model_names_arr = [model for model in sorted_model_names_arr if model in defined_model_names]

        # Add values to a results dict
        for idx, model_name in enumerate(sorted_model_names_arr):
            metric_value = self.sorted_model_metrics_df[
                (self.sorted_model_metrics_df.Metric == selected_metric) &
                (self.sorted_model_metrics_df.Subgroup == selected_subgroup) &
                (self.sorted_model_metrics_df.Model_Name == model_name)
                ]['Value'].values[0]
            metric_value = metric_value
            results[subgroup_metric][model_name] = metric_value

        return results

    def create_overall_metrics_bar_char(self, metric_names: list, plot_title: str = "Overall Metrics"):
        """
        This bar chart includes all defined models and all overall subgroup error and stability metrics,
        which are averaged across multiple runs. Using it, you can compare all models for each subgroup error or stability metric.
        This comparison also includes reversed metrics, in which values closer to zero are better
        since straight and reversed metrics in this plot are converted to the same format -- values closer to one are better.

        Parameters
        ----------
        metric_names
            List of subgroup metric names to visualize that have a scale from 0 to 1 where closer to 1 is better
        plot_title
            Title for input metrics to display on the plot

        """
        overall_metrics_df = pd.DataFrame()
        for model_name in self.models_average_metrics_dct.keys():
            model_average_results_df = self.models_average_metrics_dct[model_name].copy(deep=True)
            model_average_results_df = model_average_results_df.loc[model_average_results_df['Metric'].isin(metric_names)]

            overall_model_metrics_df = pd.DataFrame()
            overall_model_metrics_df['overall'] = model_average_results_df['overall']
            overall_model_metrics_df['metric'] = model_average_results_df['Metric']
            overall_model_metrics_df['model_name'] = model_name
            overall_metrics_df = pd.concat([overall_metrics_df, overall_model_metrics_df])

        font_increase = 2
        models_metrics_chart = (
            alt.Chart(overall_metrics_df).mark_bar().encode(
                alt.Row('metric:N', title=plot_title, sort=metric_names),
                alt.Y('model_name:N', axis=None),
                alt.X('overall:Q', axis=alt.Axis(grid=True), title=''),
                alt.Color('model_name:N',
                          scale=alt.Scale(scheme="tableau20"),
                          legend=alt.Legend(title='Model Name',
                                            labelFontSize=13 + font_increase,
                                            titleFontSize=13 + font_increase,
                                            labelLimit=300,
                                            titleLimit=300)
                          )
            )
        ).properties(
            width=500, height=50
        ).configure_headerRow(
            labelAngle=0,
            labelPadding=10,
            labelAlign='left',
            labelFontSize=14 + font_increase,
            titleFontSize=18 + font_increase,
        ).configure_axis(
            labelFontSize=14 + font_increase,
            titleFontSize=18 + font_increase,
        )

        return models_metrics_chart

    def create_boxes_and_whiskers_for_models_multiple_runs(self, metrics_lst: list):
        """
        This boxes and whiskers plot is based on overall subgroup error and stability metrics for all defined models
        and results after all runs. Using it, you can see combined information on one plot that includes different models,
         subgroup metrics, and results after multiple runs.

        Parameters
        ----------
        metrics_lst
            List of subgroup metric names to visualize

        """
        to_plot = self.all_models_metrics_df[self.all_models_metrics_df['Metric'].isin(metrics_lst)]

        plt.figure(figsize=(15, 10))
        ax = sns.boxplot(x=to_plot['Metric'],
                         y=to_plot['overall'],
                         hue=to_plot['Model_Name'])

        plt.legend(loc='upper left',
                   ncol=2,
                   fancybox=True,
                   shadow=True)
        plt.xlabel("Metric name")
        plt.ylabel("Metric value")
        fig = ax.get_figure()
        fig.tight_layout()

    def create_overall_metric_heatmap(self, model_names: list, metrics_lst: list,
                                      tolerance: float = 0.001, figsize_scale: tuple = (0.7, 0.5), font_increase: int = -3):
        """
        Create a heatmap for overall metrics.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        metrics_lst
            List of group metric names to visualize
        tolerance
            [Optional] An acceptable value difference for metrics dense ranking
        figsize_scale
            [Optional] Scale factors for a heatmap size. The first element is a scale factor for a plot width, the second one is for height.
        font_increase
            [Optional] An integer to increase or decrease the plot font.

        """
        if tolerance < 0.001 or tolerance > 0.2:
            raise ValueError('Tolerance should be in the [0.001, 0.2] range')

        # Find metric values for each model based on metric, subgroup, and model names.
        # Add the values to a results dict.
        results = {}
        for metric in metrics_lst:
            # Add an overall metric
            subgroup_metric = metric
            results = self.__filter_subgroup_metrics_df(results, subgroup_metric, metric,
                                                        selected_subgroup='overall', defined_model_names=model_names)

        model_metrics_matrix = pd.DataFrame(results).T
        model_metrics_matrix = model_metrics_matrix[sorted(model_metrics_matrix.columns)]
        model_metrics_matrix = model_metrics_matrix.round(3)  # round to make tolerance more precise
        sorted_matrix_by_rank = create_subgroup_sorted_matrix_by_rank(model_metrics_matrix, tolerance)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank,
                                                                        figsize_scale=figsize_scale,
                                                                        font_increase=font_increase)

    def create_disparity_metric_heatmap(self, model_names: list, metrics_lst: list, groups_lst: list,
                                        tolerance: float = 0.001, figsize_scale: tuple = (0.7, 0.5), font_increase: int = -3):
        """
        Create a heatmap for disparity metrics.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        metrics_lst
            List of group metric names to visualize
        groups_lst
            List of sensitive attributes
        tolerance
            [Optional] An acceptable value difference for metrics dense ranking
        figsize_scale
            [Optional] Scale factors for a heatmap size. The first element is a scale factor for a plot width, the second one is for height.
        font_increase
            [Optional] An integer to increase or decrease the plot font.

        """
        if tolerance < 0.001 or tolerance > 0.2:
            raise ValueError('Tolerance should be in the [0.001, 0.2] range')

        results = {}
        for metric in metrics_lst:
            for group in groups_lst:
                group_metric = metric + '_' + group
                results[group_metric] = dict()
                sorted_model_names_arr = self.sorted_models_composed_metrics_df[
                    (self.sorted_models_composed_metrics_df.Metric == metric) &
                    (self.sorted_models_composed_metrics_df.Subgroup == group)
                    ]['Model_Name'].values
                sorted_model_names_arr = [model for model in sorted_model_names_arr if model in model_names]

                # Add values to results dict
                for idx, model_name in enumerate(sorted_model_names_arr):
                    metric_value = self.sorted_models_composed_metrics_df[
                        (self.sorted_models_composed_metrics_df.Metric == metric) &
                        (self.sorted_models_composed_metrics_df.Subgroup == group) &
                        (self.sorted_models_composed_metrics_df.Model_Name == model_name)
                        ]['Value'].values[0]
                    metric_value = round(metric_value, 3)
                    results[group_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        model_metrics_matrix = model_metrics_matrix[sorted(model_metrics_matrix.columns)]
        model_metrics_matrix = model_metrics_matrix.round(3)  # round to make tolerance more precise
        sorted_matrix_by_rank = create_sorted_matrix_by_rank(model_metrics_matrix, tolerance)
        model_rank_heatmap = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank,
                                                                     figsize_scale=figsize_scale,
                                                                     font_increase=font_increase)
