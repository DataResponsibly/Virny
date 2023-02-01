import os
import altair as alt
import pandas as pd
import datapane as dp
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# TODO: complete documentation when finish this class

class MetricsVisualizer:
    """
    Class to create useful visualizations of models metrics.

    Parameters
    ----------
    models_metrics_dct
        Dictionary where keys are model names and values are dataframes of subgroups metrics for each model
    models_composed_metrics_df
        Dataframe of all models composed metrics
    dataset_name
        Name of a dataset that was included in metrics filenames and was used for the metrics computation
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

    def visualize_overall_metrics(self, metrics_names: list, reversed_metrics_names: list = None,
                                  x_label: str = "Prediction Metrics"):
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

        # Draw a nested barplot
        height = 9 if len(metrics_names) >= 7 else 6
        g = sns.catplot(
            data=overall_metrics_df, kind="bar",
            x="overall", y="metric", hue="model_name",
            palette="tab20",
            alpha=.8,
            height=height
        )
        g.despine(left=True)
        g.set_axis_labels("", x_label)
        g.legend.set_title("")

    def create_boxes_and_whiskers_for_models_multiple_runs(self, metrics_lst: list):
        """
        Create a boxes-and-whiskers plot for subgroup metrics after multiple runs
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
                row='Subgroup:N',
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
        bias_metrics_lst = [
            'Accuracy_Parity',
            'Equalized_Odds_TPR',
            'Equalized_Odds_FPR',
            'Disparate_Impact',
            'Statistical_Parity_Impact',
        ]
        models_bias_metrics_chart, select_bias_metric_legend, bias_color_legend = \
            self.create_models_metrics_bar_chart(bias_metrics_lst, metrics_group_name="Bias")

        variance_metrics_lst = [
            'IQR_Parity',
            'Label_Stability_Impact',
            'Std_Parity',
            'Std_Ratio',
            'Jitter_Parity',
        ]
        models_variance_metrics_chart, select_variance_metric_legend, variance_color_legend = \
            self.create_models_metrics_bar_chart(variance_metrics_lst, metrics_group_name="Variance")

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

    def create_html_report(self, report_save_path: str):
        # Create a directory if it does not exist
        if not os.path.exists(report_save_path):
            os.makedirs(report_save_path, exist_ok=True)

        self.__create_report = True
        boxes_and_whiskers_plot = self.create_boxes_and_whiskers_for_models_multiple_runs(metrics_lst=['Std', 'IQR', 'Jitter', 'FNR','FPR'])
        interactive_bar_chart = self.create_bias_variance_interactive_bar_chart()

        report_filename = f'Statistical_Bias_and_Variance_Report_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}.html'
        dp.Report("# Statistical Bias and Variance Report",
               "## Models Composed Metrics",
               dp.DataTable(self.models_composed_metrics_df, caption="Models Composed Metrics"),
               "## Boxes and Whiskers Plot for Multiple Models Runs",
               dp.Plot(boxes_and_whiskers_plot, caption="Boxes and Whiskers Plot for Multiple Models Runs"),
               "## Bias and Variance Interactive Bar Chart",
               dp.Plot(interactive_bar_chart, caption="Bias and Variance Interactive Bar Chart"),
               ).save(path=os.path.join(report_save_path, report_filename))

        self.__create_report = False
