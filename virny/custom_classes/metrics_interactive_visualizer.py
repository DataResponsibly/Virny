import pandas as pd
import gradio as gr
import altair as alt

from virny.utils.data_viz_utils import (create_model_rank_heatmap_visualization, create_sorted_matrix_by_rank,
                                        create_subgroup_sorted_matrix_by_rank)


class MetricsInteractiveVisualizer:
    """
    Class to create useful visualizations of models metrics.

    Parameters
    ----------
    model_metrics_dct
        Dictionary where keys are model names and values are dataframes of subgroup metrics for each model
    model_composed_metrics_df
        Dataframe of all model composed metrics
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these attributes

    """
    def __init__(self, model_metrics_dct: dict, model_composed_metrics_df: pd.DataFrame,
                 sensitive_attributes_dct: dict):
        self.demo = None
        self.model_names = list(model_metrics_dct.keys())
        self.sensitive_attributes_dct = sensitive_attributes_dct

        # Create one metrics df with all model_dfs
        models_metrics_df = pd.DataFrame()
        for model_name in model_metrics_dct.keys():
            model_metrics_df = model_metrics_dct[model_name]
            models_metrics_df = pd.concat([models_metrics_df, model_metrics_df])

        models_metrics_df = models_metrics_df.reset_index(drop=True)

        self.models_metrics_dct = model_metrics_dct
        self.models_metrics_df = self._align_input_metric_df(models_metrics_df, allowed_cols=["Metric", "Model_Name", "overall"],
                                                             sensitive_attrs=list(self.sensitive_attributes_dct.keys()))
        self.model_composed_metrics_df =  self._align_input_metric_df(model_composed_metrics_df, allowed_cols=["Metric", "Model_Name"],
                                                                      sensitive_attrs=list(self.sensitive_attributes_dct.keys()))

        self.melted_model_metrics_df = self.models_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                   var_name="Subgroup",
                                                                   value_name="Value")
        self.sorted_model_metrics_df = self.melted_model_metrics_df.sort_values(by=['Value'])
        self.melted_model_composed_metrics_df = self.model_composed_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                                    var_name="Subgroup",
                                                                                    value_name="Value")
        self.sorted_model_composed_metrics_df = self.melted_model_composed_metrics_df.sort_values(by=['Value'])

    def _align_input_metric_df(self, model_metrics_df: pd.DataFrame, allowed_cols: list, sensitive_attrs: list):
        # Filter columns in the input dataframe based on allowed_cols and sensitive_attrs
        filtered_cols = allowed_cols
        for col in model_metrics_df.columns:
            for sensitive_attr in sensitive_attrs:
                if sensitive_attr in col:
                    filtered_cols.append(col)
                    break

        return model_metrics_df[filtered_cols]

    def start_web_app(self):
        # css = """
        # .plot_output1 {position: right !important}
        # """
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            # ======================================= Subgroup Metrics Heatmap =======================================
            gr.Markdown(
                """
                ## Subgroup Metrics Heatmap
                Select input arguments to create a subgroup metrics heatmap.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    model_names = gr.Dropdown(
                        self.model_names, value=self.model_names[:4], max_choices=5, multiselect=True,
                        label="Model Names", info="Select model names to display on the heatmap:",
                    )
                    accuracy_metrics = gr.Dropdown(
                        ['Statistical_Bias', 'TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1'],
                        value=['Accuracy', 'F1'], multiselect=True, label="Accuracy Metrics", info="Select accuracy metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        ['Aleatoric_Uncertainty', 'Overall_Uncertainty'],
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        ['Std', 'IQR', 'Jitter', 'Label_Stability'],
                        value=['Jitter', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    subgroup_btn_view1 = gr.Button("Submit")
                with gr.Column(scale=2):
                    subgroup_model_ranking_heatmap = gr.Plot(label="Plot")

            subgroup_btn_view1.click(self._create_subgroup_model_rank_heatmap,
                                     inputs=[model_names, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics],
                                     outputs=[subgroup_model_ranking_heatmap])
            # ======================================== Group Metrics Heatmap ========================================
            gr.Markdown(
                """
                ## Group Metrics Heatmap
                Select input arguments to create a group metrics heatmap.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    model_names = gr.Dropdown(
                        self.model_names, value=self.model_names[:4], max_choices=5, multiselect=True,
                        label="Model Names", info="Select model names to display on the heatmap:",
                    )
                    fairness_metrics = gr.Dropdown(
                        ['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity'],
                        value=['Equalized_Odds_TPR', 'Equalized_Odds_FPR'], multiselect=True, label="Error Parity Metrics", info="Select error parity metrics to display on the heatmap:",
                    )
                    group_stability_metrics = gr.Dropdown(
                        ['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity'],
                        value=['Label_Stability_Ratio', 'Std_Parity'], multiselect=True, label="Stability Parity Metrics", info="Select stability parity metrics to display on the heatmap:",
                    )
                    group_btn_view1 = gr.Button("Submit")
                with gr.Column(scale=2):
                    group_model_ranking_heatmap = gr.Plot(label="Plot")

            group_btn_view1.click(self._create_group_model_rank_heatmap,
                                  inputs=[model_names, fairness_metrics, group_stability_metrics],
                                  outputs=[group_model_ranking_heatmap])
            # =============================== Subgroup and Group Metrics Bar Chart ===============================
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Subgroup Metrics Bar Chart
                        """)
                    subgroup_model_names = gr.Dropdown(
                        self.model_names, value=self.model_names[0], multiselect=False,
                        label="Model Names", info="Select one model to display on the bar chart:",
                    )
                    accuracy_metrics = gr.Dropdown(
                        ['Statistical_Bias', 'TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1'],
                        value=['Accuracy', 'F1'], multiselect=True, label="Accuracy Metrics", info="Select accuracy metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        ['Aleatoric_Uncertainty', 'Overall_Uncertainty'],
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        ['Std', 'IQR', 'Jitter', 'Label_Stability'],
                        value=['Jitter', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    subgroup_btn_view2 = gr.Button("Submit")
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Group Metrics Bar Chart
                        """)
                    group_model_names = gr.Dropdown(
                        self.model_names, value=self.model_names[0], multiselect=False,
                        label="Model Names", info="Select one model to display on the bar chart:",
                    )
                    fairness_metrics = gr.Dropdown(
                        ['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity'],
                        value=['Equalized_Odds_TPR', 'Equalized_Odds_FPR'], multiselect=True, label="Error Parity Metrics", info="Select error parity metrics to display on the heatmap:",
                    )
                    group_stability_metrics = gr.Dropdown(
                        ['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity'],
                        value=['Label_Stability_Ratio', 'Std_Parity'], multiselect=True, label="Stability Parity Metrics", info="Select stability parity metrics to display on the heatmap:",
                    )
                    group_btn_view2 = gr.Button("Submit")
            with gr.Row():
                with gr.Column():
                    subgroup_metrics_bar_chart = gr.Plot(label="Plot")
                with gr.Column():
                    group_metrics_bar_chart = gr.Plot(label="Plot")

            subgroup_btn_view2.click(self._create_subgroup_metrics_bar_chart_per_one_model,
                                     inputs=[subgroup_model_names, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics],
                                     outputs=[subgroup_metrics_bar_chart])
            group_btn_view2.click(self._create_group_metrics_bar_chart_per_one_model,
                                  inputs=[group_model_names, fairness_metrics, group_stability_metrics],
                                  outputs=[group_metrics_bar_chart])

        self.demo = demo
        self.demo.launch(inline=False, debug=True, show_error=True)

    def stop_web_app(self):
        self.demo.close()

    def _create_subgroup_model_rank_heatmap(self, model_names: list, subgroup_accuracy_metrics_lst: list,
                                            subgroup_uncertainty_metrics: list, subgroup_stability_metrics_lst: list):
        """
        Create a group model rank heatmap.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        subgroup_accuracy_metrics_lst
            A list of subgroup accuracy metrics to visualize
        subgroup_uncertainty_metrics
            A list of subgroup uncertainty metrics to visualize
        subgroup_stability_metrics_lst
            A list of subgroup stability metrics to visualize

        """
        groups_lst = self.sensitive_attributes_dct.keys()
        metrics_lst = subgroup_accuracy_metrics_lst + subgroup_uncertainty_metrics + subgroup_stability_metrics_lst

        # Find metric values for each model based on metric, subgroup, and model names.
        # Add the values to a results dict.
        results = {}
        num_models = len(model_names)
        for metric in metrics_lst:
            for group in groups_lst:
                for prefix in ['priv', 'dis']:
                    subgroup = group + '_' + prefix
                    subgroup_metric = metric + '_' + subgroup
                    results[subgroup_metric] = dict()

                    # Get distinct sorted model names
                    sorted_model_names_arr = self.sorted_model_metrics_df[
                        (self.sorted_model_metrics_df.Metric == metric) &
                        (self.sorted_model_metrics_df.Subgroup == subgroup)
                        ]['Model_Name'].values
                    sorted_model_names_arr = [model for model in sorted_model_names_arr if model in model_names]

                    # Add values to a results dict
                    for idx, model_name in enumerate(sorted_model_names_arr):
                        metric_value = self.sorted_model_metrics_df[
                            (self.sorted_model_metrics_df.Metric == metric) &
                            (self.sorted_model_metrics_df.Subgroup == subgroup) &
                            (self.sorted_model_metrics_df.Model_Name == model_name)
                            ]['Value'].values[0]
                        metric_value = round(metric_value, 3)
                        results[subgroup_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        sorted_matrix_by_rank = create_subgroup_sorted_matrix_by_rank(model_metrics_matrix)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank, num_models)

        return model_rank_heatmap

    def _create_group_model_rank_heatmap(self, model_names: list, group_fairness_metrics_lst: list,
                                         group_stability_metrics_lst: list):
        """
        Create a group model rank heatmap.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        group_fairness_metrics_lst
            A list of group fairness metrics to visualize
        group_stability_metrics_lst
            A list of group stability metrics to visualize

        """
        groups_lst = self.sensitive_attributes_dct.keys()
        metrics_lst = group_fairness_metrics_lst + group_stability_metrics_lst

        # Find metric values for each model based on metric, group, and model names.
        # Add the values to a results dict.
        results = {}
        num_models = len(model_names)
        for metric in metrics_lst:
            for group in groups_lst:
                group_metric = metric + '_' + group
                results[group_metric] = dict()

                # Get distinct sorted model names
                sorted_model_names_arr = self.sorted_model_composed_metrics_df[
                    (self.sorted_model_composed_metrics_df.Metric == metric) &
                    (self.sorted_model_composed_metrics_df.Subgroup == group)
                    ]['Model_Name'].values
                sorted_model_names_arr = [model for model in sorted_model_names_arr if model in model_names]

                # Add values to a results dict
                for idx, model_name in enumerate(sorted_model_names_arr):
                    metric_value = self.sorted_model_composed_metrics_df[
                        (self.sorted_model_composed_metrics_df.Metric == metric) &
                        (self.sorted_model_composed_metrics_df.Subgroup == group) &
                        (self.sorted_model_composed_metrics_df.Model_Name == model_name)
                        ]['Value'].values[0]
                    metric_value = round(metric_value, 3)
                    results[group_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        sorted_matrix_by_rank = create_sorted_matrix_by_rank(model_metrics_matrix)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank, num_models)

        return model_rank_heatmap

    def _create_subgroup_metrics_bar_chart_per_one_model(self, model_name: str, subgroup_accuracy_metrics_lst: list,
                                                         subgroup_uncertainty_metrics: list, subgroup_stability_metrics_lst: list):
        metrics_names = subgroup_accuracy_metrics_lst + subgroup_uncertainty_metrics + subgroup_stability_metrics_lst
        return self._create_metrics_bar_chart_per_one_model(model_name, metrics_names, metrics_type='subgroup')

    def _create_group_metrics_bar_chart_per_one_model(self, model_name: str, group_fairness_metrics_lst: list,
                                                      group_stability_metrics_lst: list):
        metrics_names = group_fairness_metrics_lst + group_stability_metrics_lst
        return self._create_metrics_bar_chart_per_one_model(model_name, metrics_names, metrics_type='group')

    def _create_metrics_bar_chart_per_one_model(self, model_name: str, metrics_names: list, metrics_type: str):
        """
        This bar chart displays metrics for different groups and one specific model.

        Parameters
        ----------
        model_name
            A model name to display metrics
        metrics_names
            A list of metric names to visualize
        metrics_type
            A metrics type ('subgroup' or 'group') to visualize

        """
        metrics_title = f'{metrics_type.capitalize()} Metrics'
        metrics_df = self.melted_model_composed_metrics_df if metrics_type == "group" else self.melted_model_metrics_df
        filtered_groups = [grp for grp in metrics_df.Subgroup.unique() if '_correct' not in grp and '_incorrect' not in grp]
        filtered_metrics_df = metrics_df[(metrics_df['Metric'].isin(metrics_names)) &
                                         (metrics_df['Model_Name'] == model_name) &
                                         (metrics_df['Subgroup'].isin(filtered_groups))]

        models_metrics_chart = (
            alt.Chart(filtered_metrics_df).mark_bar().encode(
                alt.Row('Metric:N', title=metrics_title),
                alt.Y('Subgroup:N', axis=None),
                alt.X('Value:Q', axis=alt.Axis(grid=True), title=''),
                alt.Color('Subgroup:N',
                          scale=alt.Scale(scheme="tableau20"),
                          legend=alt.Legend(title=metrics_type.capitalize(),
                                            labelFontSize=14,
                                            titleFontSize=14)
                          )
            )
        ).properties(
            width=500, height=80
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
