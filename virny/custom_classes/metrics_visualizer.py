import os
import altair as alt
import numpy as np
import pandas as pd
import datapane as dp
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from virny.configs.constants import ReportType


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
        self.__create_report = False
        self.bias_metrics_lst = [
            'Accuracy_Parity',
            'Equalized_Odds_TPR',
            'Equalized_Odds_FPR',
            'Disparate_Impact',
            'Statistical_Parity_Difference',
        ]
        self.variance_metrics_lst = [
            'IQR_Parity',
            'Label_Stability_Ratio',
            'Std_Parity',
            'Std_Ratio',
            'Jitter_Parity',
        ]

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
        self.melted_models_composed_metrics_df = self.models_composed_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                                      var_name="Subgroup",
                                                                                      value_name="Value")
        self.sorted_models_composed_metrics_df = self.melted_models_composed_metrics_df.sort_values(by=['Value'])

    def create_overall_metrics_bar_char(self, metrics_names: list, reversed_metrics_names: list = None,
                                        metrics_title: str = "Overall Metrics"):
        """
        This bar chart includes all defined models and all overall subgroup bias and variance metrics,
        which are averaged across multiple runs. Using it, you can compare all models for each subgroup bias or variance metric.
        This comparison also includes reversed metrics, in which values closer to zero are better
        since straight and reversed metrics in this plot are converted to the same format -- values closer to one are better.

        Parameters
        ----------
        metrics_names
            List of subgroup metric names to visualize that have a scale from 0 to 1 where closer to 1 is better
        reversed_metrics_names
            List of subgroup metric names to visualize that have a scale from 0 to 1 where closer to 0 is better
        metrics_title
            Title to input metrics (both metrics_names and reversed_metrics_names) to display on the plot

        """
        if reversed_metrics_names is None:
            reversed_metrics_names = []
        metrics_names = set(metrics_names + reversed_metrics_names)

        overall_metrics_df = pd.DataFrame()
        for model_name in self.models_average_metrics_dct.keys():
            model_average_results_df = self.models_average_metrics_dct[model_name].copy(deep=True)
            model_average_results_df = model_average_results_df.loc[model_average_results_df['Metric'].isin(metrics_names)]

            overall_model_metrics_df = pd.DataFrame()
            overall_model_metrics_df['overall'] = model_average_results_df['overall']
            overall_model_metrics_df['metric'] = model_average_results_df['Metric']
            overall_model_metrics_df['model_name'] = model_name
            overall_metrics_df = pd.concat([overall_metrics_df, overall_model_metrics_df])

        overall_metrics_df.loc[overall_metrics_df['metric'].isin(reversed_metrics_names), 'overall'] = \
            1 - overall_metrics_df.loc[overall_metrics_df['metric'].isin(reversed_metrics_names), 'overall']

        models_metrics_chart = (
            alt.Chart(overall_metrics_df).mark_bar().encode(
                alt.Row('metric:N', title=metrics_title),
                alt.Y('model_name:N', axis=None),
                alt.X('overall:Q', axis=alt.Axis(grid=True), title=''),
                alt.Color('model_name:N',
                          scale=alt.Scale(scheme="tableau20"),
                          legend=alt.Legend(title='Model Name',
                                            labelFontSize=13,
                                            titleFontSize=13)
                          )
            )
        ).properties(
            width=500, height=50
        ).configure_headerRow(
            labelAngle=0,
            labelPadding=10,
            labelAlign='left',
            labelFontSize=14,
            titleFontSize=18
        ).configure_axis(
            labelFontSize=14, titleFontSize=18
        )

        return models_metrics_chart

    def create_boxes_and_whiskers_for_models_multiple_runs(self, metrics_lst: list):
        """
        This boxes and whiskers plot is based on overall subgroup bias and variance metrics for all defined models
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

        if self.__create_report:
            plt.close()
            return fig

    def create_models_metrics_bar_chart(self, metrics_lst: list, metrics_group_name: str, default_plot_metric: str = None):
        if default_plot_metric is None:
            default_plot_metric = metrics_lst[0]

        df_for_model_metrics_chart = self.melted_models_composed_metrics_df.loc[self.melted_models_composed_metrics_df['Metric'].isin(metrics_lst)]
        df_for_model_metrics_chart = df_for_model_metrics_chart.reset_index(drop=True)

        radio_select = alt.selection_single(fields=['Metric'], init={'Metric': default_plot_metric}, empty="none")
        color_condition = alt.condition(radio_select,
                                        alt.Color('Metric:N', legend=None, scale=alt.Scale(scheme="tableau20")),
                                        alt.value('lightgray'))

        models_metrics_chart = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_bar()
            .transform_filter(radio_select)
            .encode(
                x='Value:Q',
                y=alt.Y('Model_Name:N', axis=None),
                color=alt.Color(
                    'Model_Name:N',
                    scale=alt.Scale(scheme="tableau20")
                ),
                row=alt.Row('Subgroup:N', title='Group'),
            )
        )

        select_metric_legend = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_circle(size=200)
            .encode(
                y=alt.Y("Metric:N", axis=alt.Axis(title=f"Select {metrics_group_name} Metric", titleFontSize=15)),
                color=color_condition,
            )
            .add_selection(radio_select)
        )

        color_legend = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_circle(size=200)
            .encode(
                y=alt.Y("Model_Name:N", axis=alt.Axis(title="Model Name", titleFontSize=15)),
                color=alt.Color("Model_Name:N", scale=alt.Scale(scheme="tableau20")),
            )
        )

        return models_metrics_chart, select_metric_legend, color_legend

    def create_bias_variance_interactive_bar_chart(self):
        """
        This interactive bar chart includes all groups, all composed group bias and variance metrics,
         and all defined models. Using it, you can select any pair of group bias and variance metrics and
          compare them across all groups and models. Since this plot is interactive, it saves a lot of space for other plots.
           Also, it could be more convenient to compare individual group bias and variance metrics using the interactive mode.
        """
        models_bias_metrics_chart, select_bias_metric_legend, bias_color_legend = \
            self.create_models_metrics_bar_chart(self.bias_metrics_lst, metrics_group_name="Bias")

        models_variance_metrics_chart, select_variance_metric_legend, variance_color_legend = \
            self.create_models_metrics_bar_chart(self.variance_metrics_lst, metrics_group_name="Variance")

        return (
                alt.hconcat(
                    alt.vconcat(
                        select_bias_metric_legend.properties(height=200, width=50),
                        select_variance_metric_legend.properties(height=200, width=50),
                        bias_color_legend.properties(height=200, width=50),
                    ),
                    models_bias_metrics_chart.properties(height=200, width=300, title="Bias Metric Plot"),
                    models_variance_metrics_chart.properties(height=200, width=300, title="Variance Metric Plot"),
                )
        )

    @staticmethod
    def _create_sorted_matrix_by_rank(model_metrics_matrix) -> np.array:
        models_distances_matrix = model_metrics_matrix.copy(deep=True).T
        metric_names = models_distances_matrix.columns
        for metric_name in metric_names:
            if 'impact' in metric_name.lower() or 'ratio' in metric_name.lower():
                models_distances_matrix[metric_name] = models_distances_matrix[metric_name] - 1
            models_distances_matrix[metric_name] = models_distances_matrix[metric_name].abs()

        models_distances_matrix = models_distances_matrix.T
        sorted_matrix_by_rank = np.argsort(np.argsort(models_distances_matrix, axis=1), axis=1)
        return sorted_matrix_by_rank

    def create_model_rank_heatmap(self, model_metrics_matrix, sorted_matrix_by_rank, num_models: int):
        """
        This heatmap includes all group bias and variance metrics and all defined models.
        Using it, you can visually compare all models across all group metrics. On this plot,
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
        matrix_width = num_models * 3
        matrix_height = model_metrics_matrix.shape[0] // 3
        plt.figure(figsize=(matrix_width, matrix_height))
        rank_colors = sns.color_palette("coolwarm", n_colors=num_models).as_hex()[::-1]
        ax = sns.heatmap(sorted_matrix_by_rank, annot=model_metrics_matrix, cmap=rank_colors,
                         fmt = '', annot_kws={'color': 'black', 'alpha': 0.7})
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top()

        cbar = ax.collections[0].colorbar
        model_ranks = [idx for idx in range(num_models)]
        cbar.set_ticks([float(idx) for idx in model_ranks])
        tick_labels = [str(idx + 1) for idx in model_ranks]
        tick_labels[0] = tick_labels[0] + ', best'
        tick_labels[-1] = tick_labels[-1] + ', worst'
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Model Ranks')

        if self.__create_report:
            plt.close()
            return ax

        ax.set_title('Model Ranks Based On Group Statistical Bias and Variance Metrics', fontsize=20)

    def create_total_model_rank_heatmap(self, sorted_matrix_by_rank, num_models):
        """
        This heatmap includes all defined models and sums of their bias and variance ranks.
        On this plot, colors display sums of ranks for one model. If the sum is smaller,
        the model has better bias or variance characteristics than other models.
        Using this plot, you can visually compare all models for bias and variance characteristics.

        Parameters
        ----------
        sorted_matrix_by_rank
            Matrix of model ranks per metric where indexes are group metric names and columns are model names
        num_models
            Number of models to visualize

        """
        total_model_ranks = dict()
        matrix_bias_metrics = [metric_name for metric_name in sorted_matrix_by_rank[self.model_names[0]].index
                               if metric_name[:metric_name.rfind('_')] in self.bias_metrics_lst]
        matrix_variance_metrics = [metric_name for metric_name in sorted_matrix_by_rank[self.model_names[0]].index
                                   if metric_name[:metric_name.rfind('_')] in self.variance_metrics_lst]

        for model_name in self.model_names:
            model_ranks = dict()
            model_ranks['Bias_Ranks_Sum'] = np.sum(sorted_matrix_by_rank[model_name][matrix_bias_metrics] + 1)
            model_ranks['Variance_Ranks_Sum'] = np.sum(sorted_matrix_by_rank[model_name][matrix_variance_metrics] + 1)
            total_model_ranks[model_name] = model_ranks

        total_model_ranks_df = pd.DataFrame(total_model_ranks).T

        matrix_width = 6
        matrix_height = num_models // 2
        plt.figure(figsize=(matrix_width, matrix_height))
        ax = sns.heatmap(total_model_ranks_df, annot=True, cmap="coolwarm_r", fmt = '')
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top()

        if self.__create_report:
            plt.close()
            return ax

        ax.set_title('Total Ranks Sum For Group Statistical Bias and Variance Metrics', fontsize=15)

    def create_model_rank_heatmaps(self, metrics_lst: list, groups_lst):
        """
        Create model rank and total model rank heatmaps.

        Parameters
        ----------
        metrics_lst
            List of group metric names to visualize
        groups_lst
            List of sensitive attributes

        """
        results = {}
        num_models = len(self.model_names)
        for metric in metrics_lst:
            for group in groups_lst:
                group_metric = metric + '_' + group
                results[group_metric] = dict()
                sorted_model_names_arr = self.sorted_models_composed_metrics_df[
                    (self.sorted_models_composed_metrics_df.Metric == metric) &
                    (self.sorted_models_composed_metrics_df.Subgroup == group)
                    ]['Model_Name'].values
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
        sorted_matrix_by_rank = MetricsVisualizer._create_sorted_matrix_by_rank(model_metrics_matrix)
        model_rank_heatmap = self.create_model_rank_heatmap(model_metrics_matrix, sorted_matrix_by_rank, num_models)
        total_model_rank_heatmap = self.create_total_model_rank_heatmap(sorted_matrix_by_rank, num_models)
        if self.__create_report:
            return model_rank_heatmap, total_model_rank_heatmap

    def create_html_report(self, report_type: ReportType, report_save_path: str):
        """
        Create Statistical Bias and Variance Report depending on report type.
        It includes visualizations and helpful details to them.
        """
        # Create a directory if it does not exist
        if not os.path.exists(report_save_path):
            os.makedirs(report_save_path, exist_ok=True)

        self.__create_report = True

        # Create plots
        bias_overall_metrics_bar_chart = self.create_overall_metrics_bar_char(
            metrics_names=['TPR', 'PPV', 'Accuracy', 'F1', 'Selection-Rate', 'Positive-Rate'],
            metrics_title="Bias Metrics"
        )
        variance_overall_metrics_bar_chart = self.create_overall_metrics_bar_char(
            metrics_names=['Label_Stability'],
            reversed_metrics_names=['Std', 'IQR', 'Jitter'],
            metrics_title="Variance Metrics"
        )
        interactive_bar_chart = self.create_bias_variance_interactive_bar_chart()
        model_rank_heatmap, total_model_rank_heatmap = \
            self.create_model_rank_heatmaps(metrics_lst=self.bias_metrics_lst + self.variance_metrics_lst,
                                            groups_lst=self.sensitive_attributes_dct.keys())

        # Set descriptions for the report
        general_desc = dp.Text(
            f"**Date of creation**: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n\n"
            "This report was created based on the following input arguments:\n"
            f"* __Dataset name__: {self.dataset_name}\n"
            f"* __Model names__: {self.model_names}\n"
            f"* __Sensitive attributes__: {list(self.sensitive_attributes_dct.keys())}\n"
        )
        composed_metrics_desc = dp.Text(
            "Below you can find a dataframe of composed group metrics for all defined models and sensitive attributes.\n"
        )
        boxes_and_whiskers_plot_desc = dp.Text(
            "The below boxes and whiskers plot is based on _overall_ subgroup bias and variance metrics for all defined models and results after all runs.\n"
            "This plot can give you the following benefits:\n"
            "* You can see combined information on one plot that includes different models, subgroup metrics, and results after multiple runs\n"
            "* You can see all quartiles for each model metric based on multiple runs\n"
            "* You can compare different models for each metric\n"
            "* You can see the variance of each model metric\n"
        )
        overall_metrics_desc = dp.Text(
            "The below bar chart includes all defined models and all _overall_ subgroup bias and variance metrics, which are averaged across multiple runs.\n"
            "This plot can give you the following benefits:\n"
            "* You can compare all models for each subgroup bias or variance metric\n"
            "* This comparison also includes reversed metrics, in which values closer to zero are better "
            "since straight and reversed metrics in this plot are converted to the same format -- values closer to one are better\n"
        )
        individual_metrics_interactive_bar_chart_desc = dp.Text(
            "The below interactive bar chart includes all groups, all composed group bias and variance metrics, "
            "and all defined models.\n"
            "This plot can give you the following benefits:\n"
            "* You can select any pair of group bias and variance metrics and compare them across all groups and models\n"
            "* Since this plot is interactive, it saves a lot of space for other plots. "
            "Also, it could be more convenient to compare individual group bias and variance metrics using the interactive mode\n"
        )
        model_ranked_heatmap_desc = dp.Text(
            "The below heatmap includes all group bias and variance metrics and all defined models.\n"
            "On this plot, colors display ranks where 1 is the best model for the metric. "
            "These ranks are conditioned on difference or ratio operations used to create these group metrics:\n"
            "* If the metric is created based on the difference operation, **closer values to zero** have ranks that are closer to the first rank\n"
            "* If the metric is created based on the ratio operation, **closer values to one** have ranks that are closer to the first rank\n\n"
            "This plot can give you the following benefits:\n"
            "* You can visually compare all models across all group metrics\n"
            "* You can visually understand where one model is better or worse than other models\n"
            "* You can find the best and worst models for each group metric\n"
        )
        overall_model_ranked_heatmap_desc = dp.Text(
            "The below heatmap includes all defined models and sums of their bias and variance ranks.\n"
            "On this plot, colors display sums of ranks for one model. If the sum is smaller, the model has better bias or variance characteristics than other models.\n"
            "This plot can give you the following benefits:\n"
            "* You can visually compare all models for bias and variance characteristics\n"
            "* You can visually understand where one model is better or worse than other models\n"
            "* You can find the best or most balanced model based on bias or variance metrics\n"
        )

        report_filename = f'{self.dataset_name}_Metrics_Report_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}.html'
        if report_type == ReportType.MULTIPLE_RUNS_MULTIPLE_MODELS:
            boxes_and_whiskers_plot = self.create_boxes_and_whiskers_for_models_multiple_runs(
                metrics_lst=['Std', 'IQR', 'Jitter', 'Label_Stability', 'Accuracy', 'TPR', 'TNR', 'FPR', 'FNR']
            )

            dp.Report("# Statistical Bias and Variance Report",
                      general_desc,

                      "## Model Composed Metrics",
                      composed_metrics_desc,
                      dp.DataTable(self.models_composed_metrics_df),

                      "## Boxes and Whiskers Plot Based On Multiple Models Runs",
                      boxes_and_whiskers_plot_desc,
                      dp.Plot(boxes_and_whiskers_plot),

                      "## Overall Bias and Variance Model Metrics Comparison",
                      overall_metrics_desc,
                      dp.Plot(bias_overall_metrics_bar_chart, responsive=False),
                      dp.Plot(variance_overall_metrics_bar_chart, responsive=False),

                      "## Bias and Variance Interactive Bar Chart",
                      individual_metrics_interactive_bar_chart_desc,
                      dp.Plot(interactive_bar_chart),

                      "## Model Ranks Based On Group Statistical Bias and Variance Metrics",
                      model_ranked_heatmap_desc,
                      dp.Plot(model_rank_heatmap, responsive=False),

                      "## Total Ranks Sum For Group Statistical Bias and Variance Metrics",
                      overall_model_ranked_heatmap_desc,
                      dp.Plot(total_model_rank_heatmap, responsive=False),
                      ).save(path=os.path.join(report_save_path, report_filename))
        else:
            dp.Report("# Statistical Bias and Variance Report",
                      general_desc,

                      "## Model Composed Metrics",
                      composed_metrics_desc,
                      dp.DataTable(self.models_composed_metrics_df),

                      "## Overall Bias and Variance Model Metrics Comparison",
                      overall_metrics_desc,
                      dp.Plot(bias_overall_metrics_bar_chart, responsive=False),
                      dp.Plot(variance_overall_metrics_bar_chart, responsive=False),

                      "## Bias and Variance Interactive Bar Chart",
                      individual_metrics_interactive_bar_chart_desc,
                      dp.Plot(interactive_bar_chart),

                      "## Model Ranks Based On Group Statistical Bias and Variance Metrics",
                      model_ranked_heatmap_desc,
                      dp.Plot(model_rank_heatmap, responsive=False),

                      "## Total Ranks Sum For Group Statistical Bias and Variance Metrics",
                      overall_model_ranked_heatmap_desc,
                      dp.Plot(total_model_rank_heatmap, responsive=False),
                      ).save(path=os.path.join(report_save_path, report_filename))

        self.__create_report = False
