import pandas as pd
import gradio as gr
import altair as alt

from virny.configs.constants import *
from virny.utils.common_helpers import str_to_float
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.utils.protected_groups_partitioning import create_test_protected_groups
from virny.utils.data_viz_utils import (create_model_rank_heatmap_visualization, create_sorted_matrix_by_rank,
                                        create_subgroup_sorted_matrix_by_rank, create_flexible_bar_plot_for_model_selection,
                                        compute_proportions, compute_base_rates, create_col_facet_bar_chart,
                                        create_model_performance_summary_visualization)


class MetricsInteractiveVisualizer:
    """
    Class to create an interactive web app based on models metrics.

    Parameters
    ----------
    X_data
        An original features dataframe
    y_data
        An original target column pandas series
    model_metrics
        A dictionary or a dataframe where keys are model names and values are dataframes of subgroup metrics for each model
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these attributes

    """
    def __init__(self, X_data: pd.DataFrame, y_data: pd.DataFrame, model_metrics, sensitive_attributes_dct: dict):
        # Preprocessed variables
        if isinstance(model_metrics, dict):
            model_metrics_dct = model_metrics
        elif isinstance(model_metrics, pd.DataFrame):
            model_names = model_metrics['Model_Name'].unique()
            model_metrics_dct = dict()
            for model_name in model_names:
                model_metrics_dct[model_name] = model_metrics[model_metrics['Model_Name'] == model_name]
        else:
            raise ValueError('model_metrics argument must be a dictionary or a pandas dataframe of metrics.')

        model_composed_metrics_df = MetricsComposer(model_metrics_dct, sensitive_attributes_dct).compose_metrics()

        # Attributes from input arguments
        self.X_data = X_data
        self.y_data = y_data
        self.model_names = list(model_metrics_dct.keys())
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.group_names = list(self.sensitive_attributes_dct.keys())

        # Technical attributes
        self.demo = None
        self.max_groups = 8

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

    def __variable_inputs(self, k):
        k = int(k)
        return [gr.Textbox(visible=True)] * k + [gr.Textbox(value='', visible=False)] * (self.max_groups - k)

    def create_web_app(self, start_app=True):
        """
        Build an interactive web application.

        Parameters
        ----------
        start_app
            [Optional] If True, the web app will be started when calling this method.
             Otherwise, the gradio demo object will be returned, and users can start the web app themselves.

        """
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            # ==================================== Dataset Statistics ====================================
            gr.Markdown(
                """
                ## Dataset Statistics
                """)
            with gr.Row():
                with gr.Column(scale=2):
                    default_val = 5
                    s = gr.Slider(1, self.max_groups, value=default_val, step=1, label="How many groups to show:")
                    grp_names = []
                    grp_dis_values = []
                    sensitive_attr_items = list(self.sensitive_attributes_dct.items())
                    for i in range(self.max_groups):
                        visibility = True if i + 1 <= default_val else False
                        with gr.Row():
                            if visibility and i + 1 <= len(sensitive_attr_items):
                                grp, dis_value = sensitive_attr_items[i]
                                if dis_value is None:
                                    dis_value = '-'
                                elif isinstance(dis_value, str):
                                    dis_value = f"'{dis_value}'"
                                grp_name = gr.Text(label=f"Group {i + 1}", value=grp, interactive=True, visible=visibility)
                                grp_dis_value = gr.Text(label="Disadvantage value", value=dis_value, interactive=True, visible=visibility)
                            else:
                                grp_name = gr.Text(label=f"Group {i + 1}", interactive=True, visible=visibility)
                                grp_dis_value = gr.Text(label="Disadvantage value", interactive=True, visible=visibility)
                        grp_names.append(grp_name)
                        grp_dis_values.append(grp_dis_value)

                    s.change(self.__variable_inputs, s, grp_names)
                    s.change(self.__variable_inputs, s, grp_dis_values)
                    btn_view0 = gr.Button("Submit")
                with gr.Column(scale=4):
                    dataset_proportions_bar_chart = gr.Plot(label="Group Proportions and Base Rates")

            btn_view0.click(self._create_dataset_proportions_bar_chart,
                            inputs=[grp_names[0], grp_names[1], grp_names[2], grp_names[3], grp_names[4], grp_names[5], grp_names[6], grp_names[7],
                                    grp_dis_values[0], grp_dis_values[1], grp_dis_values[2], grp_dis_values[3], grp_dis_values[4], grp_dis_values[5], grp_dis_values[6], grp_dis_values[7]],
                            outputs=[dataset_proportions_bar_chart])

            # ==================================== Bar Chart for Model Selection ====================================
            gr.Markdown(
                """
                ## Bar Chart for Model Selection
                Select input arguments to create a bar chart for model selection. Default values display the lowest and greatest limits of constraints.
                """)
            with gr.Row():
                with gr.Column(scale=2):
                    group_name = gr.Dropdown(
                        self.group_names,
                        value=self.group_names[0], multiselect=False, label="Group Name for Disparity Metrics",
                    )
                    with gr.Row():
                        overall_metric1 = gr.Dropdown(
                            sorted(self.all_overall_metrics),
                            value='Accuracy', multiselect=False, label="Overall Constraint (C1)",
                            scale=2
                        )
                        overall_metric_min_val1 = gr.Text(value="0.0", label="Min value", scale=1)
                        overall_metric_max_val1 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        disparity_metric1 = gr.Dropdown(
                            sorted(self.all_disparity_metrics),
                            value='Equalized_Odds_FPR', multiselect=False, label="Disparity Constraint (C2)",
                            scale=2
                        )
                        disparity_metric_min_val1 = gr.Text(value="-1.0", label="Min value", scale=1)
                        disparity_metric_max_val1 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        overall_metric2 = gr.Dropdown(
                            sorted(self.all_overall_metrics),
                            value='Label_Stability', multiselect=False, label="Overall Constraint (C3)",
                            scale=2
                        )
                        overall_metric_min_val2 = gr.Text(value="0.0", label="Min value", scale=1)
                        overall_metric_max_val2 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        disparity_metric2 = gr.Dropdown(
                            sorted(self.all_disparity_metrics),
                            value='Label_Stability_Ratio', multiselect=False, label="Disparity Constraint (C4)",
                            scale=2
                        )
                        disparity_metric_min_val2 = gr.Text(value="0.7", label="Min value", scale=1)
                        disparity_metric_max_val2 = gr.Text(value="1.5", label="Max value", scale=1)

                    btn_view1 = gr.Button("Submit")
                with gr.Column(scale=3):
                    bar_plot_for_model_selection = gr.Plot(label="Bar Chart")
                    df_with_models_satisfied_all_constraints = gr.DataFrame(label='Models that satisfy all 4 constraints')

            btn_view1.click(self._create_bar_plot_for_model_selection,
                            inputs=[group_name,
                                    overall_metric1, overall_metric_min_val1, overall_metric_max_val1,
                                    disparity_metric1, disparity_metric_min_val1, disparity_metric_max_val1,
                                    overall_metric2, overall_metric_min_val2, overall_metric_max_val2,
                                    disparity_metric2, disparity_metric_min_val2, disparity_metric_max_val2],
                            outputs=[bar_plot_for_model_selection, df_with_models_satisfied_all_constraints])

            # ======================================= Overall Metrics Heatmap =======================================
            gr.Markdown(
                """
                ## Overall Metrics Heatmap
                Select input arguments to create an overall metrics heatmap.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    model_names = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[:4], max_choices=5, multiselect=True,
                        label="Model Names", info="Select model names to display on the heatmap:",
                    )
                    subgroup_tolerance = gr.Text(value="0.005", label="Tolerance", info="Define an acceptable tolerance for metric dense ranking.")
                    accuracy_metrics = gr.Dropdown(
                        sorted(self.all_accuracy_metrics),
                        value=['Accuracy', 'F1'], multiselect=True, label="Correctness Metrics", info="Select correctness metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        sorted(self.all_uncertainty_metrics),
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        sorted(self.all_stability_metrics),
                        value=['Std', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    subgroup_btn_view2 = gr.Button("Submit")
                with gr.Column(scale=2):
                    subgroup_model_ranking_heatmap = gr.Plot(label="Heatmap")

            subgroup_btn_view2.click(self._create_subgroup_model_rank_heatmap,
                                     inputs=[model_names, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics, subgroup_tolerance],
                                     outputs=[subgroup_model_ranking_heatmap])

            # ======================================== Disparity Metrics Heatmap ========================================
            gr.Markdown(
                """
                ## Disparity Metrics Heatmap
                Select input arguments to create a disparity metrics heatmap.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    model_names = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[:4], max_choices=5, multiselect=True,
                        label="Model Names", info="Select model names to display on the heatmap:",
                    )
                    group_tolerance = gr.Text(value="0.005", label="Tolerance", info="Define an acceptable tolerance for metric dense ranking.")
                    fairness_metrics_vw2 = gr.Dropdown(
                        sorted(self.all_error_disparity_metrics),
                        value=['Equalized_Odds_FPR', 'Equalized_Odds_TPR'], multiselect=True, label="Error Disparity Metrics", info="Select error disparity metrics to display on the heatmap:",
                    )
                    group_uncertainty_metrics_vw2 = gr.Dropdown(
                        sorted(self.all_uncertainty_disparity_metrics),
                        value=[OVERALL_UNCERTAINTY_DIFFERENCE], multiselect=True, label="Uncertainty Disparity Metrics", info="Select uncertainty disparity metrics to display on the heatmap:",
                    )
                    group_stability_metrics_vw2 = gr.Dropdown(
                        sorted(self.all_stability_disparity_metrics),
                        value=[LABEL_STABILITY_RATIO, STD_DIFFERENCE], multiselect=True, label="Stability Disparity Metrics", info="Select stability disparity metrics to display on the heatmap:",
                    )
                    group_btn_view2 = gr.Button("Submit")
                with gr.Column(scale=2):
                    group_model_ranking_heatmap = gr.Plot(label="Heatmap")

            group_btn_view2.click(self._create_group_model_rank_heatmap,
                                  inputs=[model_names, fairness_metrics_vw2, group_uncertainty_metrics_vw2, group_stability_metrics_vw2, group_tolerance],
                                  outputs=[group_model_ranking_heatmap])

            # ============================ Group Specific and Disparity Metrics Bar Charts ============================
            with gr.Row():
                # Scale column 1 to a half of a screen
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Group Specific and Disparity Metrics Bar Charts
                        """)
                    model_name_vw3 = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[0], multiselect=False, scale=1,
                        label="Model Name", info="Select one model to display on the bar charts:",
                    )
                with gr.Column():
                    pass
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Group Specific Metrics
                        """)
                    accuracy_metrics = gr.Dropdown(
                        sorted(self.all_accuracy_metrics),
                        value=['Accuracy', 'F1'], multiselect=True, label="Correctness Metrics", info="Select correctness metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        sorted(self.all_uncertainty_metrics),
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        sorted(self.all_stability_metrics),
                        value=['Std', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    btn_view3 = gr.Button("Submit")
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Disparity Metrics
                        """)
                    fairness_metrics_vw3 = gr.Dropdown(
                        sorted(self.all_error_disparity_metrics),
                        value=[EQUALIZED_ODDS_FPR, EQUALIZED_ODDS_TPR], multiselect=True, label="Error Disparity Metrics", info="Select error disparity metrics to display on the heatmap:",
                    )
                    group_uncertainty_metrics_vw3 = gr.Dropdown(
                        sorted(self.all_uncertainty_disparity_metrics),
                        value=[ALEATORIC_UNCERTAINTY_RATIO, OVERALL_UNCERTAINTY_DIFFERENCE], multiselect=True, label="Uncertainty Disparity Metrics", info="Select uncertainty disparity metrics to display on the heatmap:",
                    )
                    group_stability_metrics_vw3 = gr.Dropdown(
                        sorted(self.all_stability_disparity_metrics),
                        value=[LABEL_STABILITY_RATIO, STD_DIFFERENCE], multiselect=True, label="Stability Disparity Metrics", info="Select stability disparity metrics to display on the heatmap:",
                    )
            with gr.Row():
                with gr.Column():
                    subgroup_metrics_bar_chart = gr.Plot(label="Group Specific Bar Chart")
                with gr.Column():
                    group_metrics_bar_chart = gr.Plot(label="Disparity Bar Chart")

            btn_view3.click(self._create_subgroup_metrics_bar_chart_per_one_model,
                            inputs=[model_name_vw3, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics],
                            outputs=[subgroup_metrics_bar_chart])
            btn_view3.click(self._create_group_metrics_bar_chart_per_one_model,
                            inputs=[model_name_vw3, fairness_metrics_vw3, group_uncertainty_metrics_vw3, group_stability_metrics_vw3],
                            outputs=[group_metrics_bar_chart])

            # ============================ Model Performance Summary ============================
            with gr.Row():
                # Scale column 1 to a half of a screen
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Model Performance Summary
                        """)
                    model_name_vw4 = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[0], multiselect=False, scale=1,
                        label="Model Name", info="Select one model to generate a performance summary:",
                    )
                with gr.Column():
                    pass
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Overall Metric Constraints
                        """)
                    with gr.Row():
                        accuracy_metric_vw4 = gr.Dropdown(
                            sorted([metric for metric in self.all_accuracy_metrics if metric not in (POSITIVE_RATE, SELECTION_RATE)]),
                            value=ACCURACY, multiselect=False, label="Correctness Metric",
                            scale=2
                        )
                        accuracy_min_val_vw4 = gr.Text(value="0.0", label="Min value", scale=1)
                        accuracy_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        subgroup_stability_metric_vw4 = gr.Dropdown(
                            sorted(self.all_stability_metrics),
                            value=LABEL_STABILITY, multiselect=False, label="Stability Metric",
                            scale=2
                        )
                        subgroup_stab_min_val_vw4 = gr.Text(value="0.0", label="Min value", scale=1)
                        subgroup_stab_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        subgroup_uncertainty_metric_vw4 = gr.Dropdown(
                            sorted(self.all_uncertainty_metrics),
                            value=ALEATORIC_UNCERTAINTY, multiselect=False, label="Uncertainty Metric",
                            scale=2
                        )
                        subgroup_uncertainty_min_val_vw4 = gr.Text(value="0.0", label="Min value", scale=1)
                        subgroup_uncertainty_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        positive_rate_metric_vw4 = gr.Dropdown(
                            [POSITIVE_RATE, SELECTION_RATE],
                            value=SELECTION_RATE, multiselect=False, label="Representation Metric",
                            scale=2
                        )
                        positive_rate_min_val_vw4 = gr.Text(value="0.0", label="Min value", scale=1)
                        positive_rate_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)

                    btn_view4 = gr.Button("Submit")
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Disparity Metric Constraints
                        """)
                    with gr.Row():
                        fairness_metric_vw4 = gr.Dropdown(
                            sorted(self.all_error_disparity_metrics),
                            value=EQUALIZED_ODDS_FPR, multiselect=False, label="Error Disparity Metric",
                            scale=2
                        )
                        fairness_min_val_vw4 = gr.Text(value="-1.0", label="Min value", scale=1)
                        fairness_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        group_stability_metrics_vw4 = gr.Dropdown(
                            sorted(self.all_stability_disparity_metrics),
                            value=LABEL_STABILITY_RATIO, multiselect=False, label="Stability Disparity Metric",
                            scale=2
                        )
                        group_stab_min_val_vw4 = gr.Text(value="0.7", label="Min value", scale=1)
                        group_stab_max_val_vw4 = gr.Text(value="1.5", label="Max value", scale=1)
                    with gr.Row():
                        group_uncertainty_metrics_vw4 = gr.Dropdown(
                            sorted(self.all_uncertainty_disparity_metrics),
                            value=ALEATORIC_UNCERTAINTY_DIFFERENCE, multiselect=False, label="Uncertainty Disparity Metric",
                            scale=2
                        )
                        group_uncertainty_min_val_vw4 = gr.Text(value="-1.0", label="Min value", scale=1)
                        group_uncertainty_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        group_positive_rate_metrics_vw4 = gr.Dropdown(
                            sorted([DISPARATE_IMPACT, STATISTICAL_PARITY_DIFFERENCE]),
                            value=STATISTICAL_PARITY_DIFFERENCE, multiselect=False, label="Representation Disparity Metric",
                            scale=2
                        )
                        group_positive_rate_min_val_vw4 = gr.Text(value="-1.0", label="Min value", scale=1)
                        group_positive_rate_max_val_vw4 = gr.Text(value="1.0", label="Max value", scale=1)
            with gr.Row():
                model_performance_summary = gr.Plot(label="Model Performance Summary")

            btn_view4.click(self._create_model_performance_summary,
                            inputs=[model_name_vw4,
                                    accuracy_metric_vw4, accuracy_min_val_vw4, accuracy_max_val_vw4,
                                    subgroup_stability_metric_vw4, subgroup_stab_min_val_vw4, subgroup_stab_max_val_vw4,
                                    subgroup_uncertainty_metric_vw4, subgroup_uncertainty_min_val_vw4, subgroup_uncertainty_max_val_vw4,
                                    positive_rate_metric_vw4, positive_rate_min_val_vw4, positive_rate_max_val_vw4,
                                    fairness_metric_vw4, fairness_min_val_vw4, fairness_max_val_vw4,
                                    group_stability_metrics_vw4, group_stab_min_val_vw4, group_stab_max_val_vw4,
                                    group_uncertainty_metrics_vw4, group_uncertainty_min_val_vw4, group_uncertainty_max_val_vw4,
                                    group_positive_rate_metrics_vw4, group_positive_rate_min_val_vw4, group_positive_rate_max_val_vw4],
                            outputs=[model_performance_summary])

        self.demo = demo
        if start_app:
            self.demo.launch(inline=False, debug=True, show_error=True)
        else:
            return self.demo

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

    def __check_metric_constraints(self, model_performance_dct, input_constraint_dct):
        """
        Create a dictionary of constraint check identifiers for each metric and group.
        """
        model_metrics_constraints_check_dct = dict()
        for metric_dim in model_performance_dct.keys():
            model_metrics_constraints_check_dct[metric_dim] = dict()
            for group in model_performance_dct[metric_dim]:
                constraint_type = 'overall' if group == 'Overall' else 'disparity'
                min_val, max_val = input_constraint_dct[metric_dim][constraint_type][1]
                if min_val > max_val:
                    raise ValueError(f'Max value for the {metric_dim} {constraint_type} dimension should be greater than min value.')

                check = 1 if model_performance_dct[metric_dim][group] >= min_val and model_performance_dct[metric_dim][group] <= max_val else 0
                model_metrics_constraints_check_dct[metric_dim][group] = check

        return model_metrics_constraints_check_dct

    def _create_dataset_proportions_bar_chart(self, grp_name1, grp_name2, grp_name3, grp_name4, grp_name5, grp_name6, grp_name7, grp_name8,
                                              grp_dis_val1, grp_dis_val2, grp_dis_val3, grp_dis_val4, grp_dis_val5, grp_dis_val6, grp_dis_val7, grp_dis_val8):
        grp_names = [grp_name1, grp_name2, grp_name3, grp_name4, grp_name5, grp_name6, grp_name7, grp_name8]
        grp_names = [grp.strip() for grp in grp_names if grp != '' and grp is not None]
        grp_dis_values = [grp_dis_val1, grp_dis_val2, grp_dis_val3, grp_dis_val4, grp_dis_val5, grp_dis_val6, grp_dis_val7, grp_dis_val8]
        grp_dis_values = [grp.strip() for grp in grp_dis_values if grp != '' and grp is not None]

        if len(grp_names) != len(grp_dis_values):
            raise ValueError("Numbers of sensitive attributes and their disadvantaged groups are different."
                             "Please, put '-' as a disadvantaged value for intersectional sensitive attributes.")

        # Create a sensitive attrs dict
        input_sensitive_attrs_dct = dict()
        for grp_name, grp_dis_val in zip(grp_names, grp_dis_values):
            if '&' in grp_name:
                input_sensitive_attrs_dct[grp_name] = None
            else:
                try:
                    converted_grp_dis_val = eval(grp_dis_val)
                    input_sensitive_attrs_dct[grp_name] = converted_grp_dis_val
                except Exception as _:
                    raise ValueError(f"Type casting error with the {grp_dis_val} value. Use quotes for string disavantaged values.")

        # Partition on protected groups
        protected_groups = create_test_protected_groups(self.X_data, self.X_data, input_sensitive_attrs_dct)

        # Create a df with group proportions and group base rates
        subgroup_proportions_dct = compute_proportions(protected_groups, self.X_data)
        subgroup_base_rates_dct = compute_base_rates(protected_groups, self.y_data)
        subgroup_relative_base_rates_dct = dict()
        for subgroup in subgroup_proportions_dct.keys():
            pct = subgroup_base_rates_dct[subgroup]['percentage'] * subgroup_proportions_dct[subgroup]['percentage']
            subgroup_relative_base_rates_dct[subgroup] = {'percentage': pct, 'num_rows': subgroup_base_rates_dct[subgroup]['num_rows']}

        stats_df = pd.DataFrame(columns=['Subgroup', 'Percentage', 'Num_Rows', 'Statistics_Type'])
        for subgroup in subgroup_proportions_dct.keys():
            stats_df.loc[len(stats_df.index)] = [subgroup, subgroup_proportions_dct[subgroup]['percentage'],
                                                 subgroup_proportions_dct[subgroup]['num_rows'], 'Proportion']
            stats_df.loc[len(stats_df.index)] = [subgroup, subgroup_relative_base_rates_dct[subgroup]['percentage'],
                                                 subgroup_relative_base_rates_dct[subgroup]['num_rows'], 'Base_Rate']

        # Create a row facet bar chart
        facet_sort_by_lst = ['overall'] + [grp for grp in stats_df.Subgroup.unique() if grp.lower() != 'overall']
        col_facet_bar_chart = create_col_facet_bar_chart(stats_df,
                                                         x_col='Statistics_Type',
                                                         y_col='Num_Rows',
                                                         facet_column_name='Subgroup',
                                                         text_labels_column='Percentage',
                                                         x_sort_by_lst=['Proportion', 'Base_Rate'],
                                                         facet_sort_by_lst=facet_sort_by_lst,
                                                         color_legend_title='Statistics Type',
                                                         facet_title='')

        return col_facet_bar_chart

    def _create_bar_plot_for_model_selection(self, group_name, overall_metric1, overall_metric_min_val1, overall_metric_max_val1,
                                             disparity_metric1, disparity_metric_min_val1, disparity_metric_max_val1,
                                             overall_metric2, overall_metric_min_val2, overall_metric_max_val2,
                                             disparity_metric2, disparity_metric_min_val2, disparity_metric_max_val2):
        metric_name_to_alias_dct = {
            overall_metric1: 'C1',
            disparity_metric1: 'C2',
            overall_metric2: 'C3',
            disparity_metric2: 'C4',
        }
        overall_constraint1 = (overall_metric1, str_to_float(overall_metric_min_val1, 'Overall Constraint (C1) min value'), str_to_float(overall_metric_max_val1, 'Overall Constraint (C1) max value'))
        disparity_constraint1 = (disparity_metric1, str_to_float(disparity_metric_min_val1, 'Disparity Constraint (C2) min value'), str_to_float(disparity_metric_max_val1, 'Disparity Constraint (C2) max value'))
        overall_constraint2 = (overall_metric2, str_to_float(overall_metric_min_val2, 'Overall Constraint (C3) min value'), str_to_float(overall_metric_max_val2, 'Overall Constraint (C3) max value'))
        disparity_constraint2 = (disparity_metric2, str_to_float(disparity_metric_min_val2, 'Disparity Constraint (C4) min value'), str_to_float(disparity_metric_max_val2, 'Disparity Constraint (C4) max value'))

        # Create individual constraints
        metrics_value_range_dct = dict()
        for constraint in [overall_constraint1, disparity_constraint1, overall_constraint2, disparity_constraint2]:
            metrics_value_range_dct[constraint[0]] = [constraint[1], constraint[2]]
        # Create intersectional constraints
        metrics_value_range_dct[f'{overall_constraint1[0]}&{disparity_constraint1[0]}'] = None
        metrics_value_range_dct[f'{overall_constraint1[0]}&{overall_constraint2[0]}'] = None
        metrics_value_range_dct[f'{overall_constraint1[0]}&{disparity_constraint2[0]}'] = None
        metrics_value_range_dct[(f'{overall_constraint1[0]}&{disparity_constraint1[0]}'
                                 f'&{overall_constraint2[0]}&{disparity_constraint2[0]}')] = None

        melted_all_subgroup_metrics_per_model_dct = dict()
        for model_name in self.melted_model_metrics_df['Model_Name'].unique():
            melted_all_subgroup_metrics_per_model_dct[model_name] = (
                self.melted_model_metrics_df)[self.melted_model_metrics_df.Model_Name == model_name]

        melted_all_group_metrics_per_model_dct = dict()
        for model_name in self.melted_model_composed_metrics_df['Model_Name'].unique():
            melted_all_group_metrics_per_model_dct[model_name] = (
                self.melted_model_composed_metrics_df)[self.melted_model_composed_metrics_df.Model_Name == model_name]

        return create_flexible_bar_plot_for_model_selection(melted_all_subgroup_metrics_per_model_dct,
                                                            melted_all_group_metrics_per_model_dct,
                                                            metrics_value_range_dct,
                                                            group=group_name,
                                                            metric_name_to_alias_dct=metric_name_to_alias_dct)

    def _create_subgroup_model_rank_heatmap(self, model_names: list, subgroup_accuracy_metrics_lst: list,
                                            subgroup_uncertainty_metrics: list, subgroup_stability_metrics_lst: list,
                                            tolerance: str):
        """
        Create a group model rank heatmap.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        subgroup_accuracy_metrics_lst
            A list of subgroup correctness metrics to visualize
        subgroup_uncertainty_metrics
            A list of subgroup uncertainty metrics to visualize
        subgroup_stability_metrics_lst
            A list of subgroup stability metrics to visualize
        tolerance
            An acceptable value difference for metrics dense ranking

        """
        tolerance = str_to_float(tolerance, 'Tolerance')
        if tolerance < 0.001 or tolerance > 0.2:
            raise ValueError('Tolerance should be in the [0.001, 0.2] range')
        metrics_lst = subgroup_accuracy_metrics_lst + subgroup_uncertainty_metrics + subgroup_stability_metrics_lst

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
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank)

        return model_rank_heatmap

    def _create_group_model_rank_heatmap(self, model_names: list, group_fairness_metrics_lst: list,
                                         group_uncertainty_metrics: list, group_stability_metrics_lst: list, tolerance: str):
        """
        Create a group model rank heatmap.

        Parameters
        ----------
        model_names
            A list of selected model names to display on the heatmap
        group_fairness_metrics_lst
            A list of group fairness metrics to visualize
        group_uncertainty_metrics
            A list of group uncertainty metrics to visualize
        group_stability_metrics_lst
            A list of group stability metrics to visualize
        tolerance
            An acceptable value difference for metrics dense ranking

        """
        tolerance = str_to_float(tolerance, 'Tolerance')
        if tolerance < 0.001 or tolerance > 0.2:
            raise ValueError('Tolerance should be in the [0.001, 0.2] range')

        groups_lst = self.sensitive_attributes_dct.keys()
        metrics_lst = group_fairness_metrics_lst + group_uncertainty_metrics + group_stability_metrics_lst

        # Find metric values for each model based on metric, group, and model names.
        # Add the values to a results dict.
        results = {}
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
                    metric_value = metric_value
                    results[group_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        model_metrics_matrix = model_metrics_matrix[sorted(model_metrics_matrix.columns)]
        model_metrics_matrix = model_metrics_matrix.round(3)  # round to make tolerance more precise
        sorted_matrix_by_rank = create_sorted_matrix_by_rank(model_metrics_matrix, tolerance)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank)

        return model_rank_heatmap

    def _create_model_performance_summary(self, model_name: str, accuracy_metric, accuracy_min_val, accuracy_max_val,
                                          stability_metric, stability_min_val, stability_max_val,
                                          uncertainty_metric, uncertainty_min_val, uncertainty_max_val,
                                          positive_rate_metric, positive_rate_min_val, positive_rate_max_val,
                                          fairness_metric, fairness_min_val, fairness_max_val,
                                          group_stability_metric, group_stab_min_val, group_stab_max_val,
                                          group_uncertainty_metric, group_uncertainty_min_val, group_uncertainty_max_val,
                                          group_positive_rate_metric, group_positive_rate_min_val, group_positive_rate_max_val):
        accuracy_constraint = (accuracy_metric, [str_to_float(accuracy_min_val, 'Correctness min value'),
                                                 str_to_float(accuracy_max_val, 'Correctness max value')])
        stability_constraint = (stability_metric, [str_to_float(stability_min_val, 'Stability min value'),
                                                   str_to_float(stability_max_val, 'Stability max value')])
        uncertainty_constraint = (uncertainty_metric, [str_to_float(uncertainty_min_val, 'Uncertainty min value'),
                                                       str_to_float(uncertainty_max_val, 'Uncertainty max value')])
        positive_rate_constraint = (positive_rate_metric, [str_to_float(positive_rate_min_val, 'Representation min value'),
                                                           str_to_float(positive_rate_max_val, 'Representation max value')])

        fairness_constraint = (fairness_metric, [str_to_float(fairness_min_val, 'Error disparity metric min value'),
                                                 str_to_float(fairness_max_val, 'Error disparity metric max value')])
        group_stability_constraint = (group_stability_metric, [str_to_float(group_stab_min_val, 'Stability disparity min value'),
                                                               str_to_float(group_stab_max_val, 'Stability disparity max value')])
        group_uncertainty_constraint = (group_uncertainty_metric, [str_to_float(group_uncertainty_min_val, 'Uncertainty disparity min value'),
                                                                   str_to_float(group_uncertainty_max_val, 'Uncertainty disparity max value')])
        group_positive_rate_constraint = (group_positive_rate_metric, [str_to_float(group_positive_rate_min_val, 'Representation disparity min value'),
                                                                       str_to_float(group_positive_rate_max_val, 'Representation disparity max value')])

        input_constraints_dct = {
            'Correctness': {
                'overall': accuracy_constraint,
                'disparity': fairness_constraint,
            },
            'Stability': {
                'overall': stability_constraint,
                'disparity': group_stability_constraint,
            },
            'Uncertainty': {
                'overall': uncertainty_constraint,
                'disparity': group_uncertainty_constraint,
            },
            'Representation': {
                'overall': positive_rate_constraint,
                'disparity': group_positive_rate_constraint,
            },
        }

        # Extract overall and disparity metrics from metrics dfs.
        # Add the values to a results dict.
        model_performance_dct = {}
        for metric_dim in input_constraints_dct.keys():
            model_performance_dct[metric_dim] = dict()
            subgroup_metric = input_constraints_dct[metric_dim]['overall'][0]
            model_performance_dct[metric_dim]['Overall'] = self.sorted_model_metrics_df[
                (self.sorted_model_metrics_df.Metric == subgroup_metric) &
                (self.sorted_model_metrics_df.Subgroup == 'overall') &
                (self.sorted_model_metrics_df.Model_Name == model_name)
                ]['Value'].values[0]

            group_metric = input_constraints_dct[metric_dim]['disparity'][0]
            for group_name in self.group_names:
                model_performance_dct[metric_dim]['Disparity: ' + group_name] = self.sorted_model_composed_metrics_df[
                    (self.sorted_model_composed_metrics_df.Metric == group_metric) &
                    (self.sorted_model_composed_metrics_df.Subgroup == group_name) &
                    (self.sorted_model_composed_metrics_df.Model_Name == model_name)
                    ]['Value'].values[0]

        metric_constraints_check_dct = self.__check_metric_constraints(model_performance_dct, input_constraints_dct)

        model_metrics_matrix = pd.DataFrame(model_performance_dct).T
        aligned_column_names = ['Overall'] + [col for col in model_metrics_matrix.columns if col != 'Overall']
        model_metrics_matrix = model_metrics_matrix[aligned_column_names]
        metric_constraints_check_matrix = pd.DataFrame(metric_constraints_check_dct).T
        metric_constraints_check_matrix = metric_constraints_check_matrix[aligned_column_names]

        model_performance_summary, _ = create_model_performance_summary_visualization(model_metrics_matrix, metric_constraints_check_matrix)
        return model_performance_summary

    def _create_subgroup_metrics_bar_chart_per_one_model(self, model_name: str, subgroup_accuracy_metrics_lst: list,
                                                         subgroup_uncertainty_metrics: list, subgroup_stability_metrics_lst: list):
        metrics_names = subgroup_accuracy_metrics_lst + subgroup_uncertainty_metrics + subgroup_stability_metrics_lst
        return self._create_metrics_bar_chart_per_one_model(model_name, metrics_names, metrics_type='subgroup')

    def _create_group_metrics_bar_chart_per_one_model(self, model_name: str, group_fairness_metrics_lst: list,
                                                      group_uncertainty_metrics_lst: list,
                                                      group_stability_metrics_lst: list):
        metrics_names = group_fairness_metrics_lst + group_uncertainty_metrics_lst + group_stability_metrics_lst
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
        metrics_title = 'Disparity Metrics' if metrics_type == "group" else 'Group Specific Metrics'
        metrics_df = self.melted_model_composed_metrics_df if metrics_type == "group" else self.melted_model_metrics_df
        filtered_groups = [grp for grp in metrics_df.Subgroup.unique() if '_correct' not in grp and '_incorrect' not in grp]
        filtered_groups = [grp for grp in filtered_groups if grp.lower() != 'overall'] + ['overall']
        filtered_metrics_df = metrics_df[(metrics_df['Metric'].isin(metrics_names)) &
                                         (metrics_df['Model_Name'] == model_name) &
                                         (metrics_df['Subgroup'].isin(filtered_groups))]

        base_font_size = 16
        models_metrics_chart = (
            alt.Chart().mark_bar().encode(
                alt.Y('Subgroup:N', axis=None, sort=filtered_groups),
                alt.X('Value:Q', axis=alt.Axis(grid=True), title=''),
                alt.Color('Subgroup:N',
                          scale=alt.Scale(scheme="tableau20"),
                          sort=filtered_groups,
                          legend=alt.Legend(title='Disparity' if metrics_type == 'group' else 'Group',
                                            labelFontSize=base_font_size,
                                            titleFontSize=base_font_size + 2))
            )
        )

        text = (
            models_metrics_chart.mark_text(
                align='left',
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
                models_metrics_chart, text, data=filtered_metrics_df
            ).properties(
                width=500,
                height=100
            ).facet(
                row=alt.Row('Metric:N', title=metrics_title, sort=metrics_names)
            ).configure(
                padding={'top':  33},
            ).configure_headerRow(
                labelAngle=0,
                labelPadding=10,
                labelAlign='left',
                labelFontSize=base_font_size,
                titleFontSize=base_font_size + 2
            ).configure_axis(
                labelFontSize=base_font_size, titleFontSize=base_font_size + 2
            )
        )

        return final_chart
