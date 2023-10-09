import pandas as pd
import gradio as gr
import altair as alt

from virny.utils.common_helpers import isfloat_regex, str_to_float
from virny.utils.data_viz_utils import (create_model_rank_heatmap_visualization, create_sorted_matrix_by_rank,
                                        create_subgroup_sorted_matrix_by_rank, create_bar_plot_for_model_selection)


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
        self.max_groups = 8
        self.model_names = list(model_metrics_dct.keys())
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.group_names = list(self.sensitive_attributes_dct.keys())

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

    def __variable_inputs(self, k):
        k = int(k)
        return [gr.Textbox(value='', visible=True)] * k + [gr.Textbox(value='', visible=False)] * (self.max_groups - k)

    def _test(self, grp_name1, grp_name2, grp_name3, grp_name4, grp_name5, grp_name6, grp_name7, grp_name8,
              grp_dis_val1, grp_dis_val2, grp_dis_val3, grp_dis_val4, grp_dis_val5, grp_dis_val6, grp_dis_val7, grp_dis_val8):
        grp_names = [grp_name1, grp_name2, grp_name3, grp_name4, grp_name5, grp_name6, grp_name7, grp_name8]
        grp_names = [grp for grp in grp_names if grp != '' and grp is not None]
        grp_dis_values = [grp_dis_val1, grp_dis_val2, grp_dis_val3, grp_dis_val4, grp_dis_val5, grp_dis_val6, grp_dis_val7, grp_dis_val8]
        grp_dis_values = [grp for grp in grp_dis_values if grp != '' and grp is not None]

        inp_str1 = ' '.join(grp_names) + '.'
        inp_str2 = ' '.join(grp_dis_values) + '.'

        return inp_str1 + ' | ' + inp_str2

    def start_web_app(self):
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
                    for i in range(self.max_groups):
                        visibility = True if i + 1 <= default_val else False
                        with gr.Row():
                            grp_name = gr.Text(label=f"Group {i + 1}", interactive=True, visible=visibility)
                            grp_dis_value = gr.Text(label="Disadvantage value", interactive=True, visible=visibility)
                        grp_names.append(grp_name)
                        grp_dis_values.append(grp_dis_value)

                    s.change(self.__variable_inputs, s, grp_names)
                    s.change(self.__variable_inputs, s, grp_dis_values)
                    btn_view0 = gr.Button("Submit")
                with gr.Column(scale=3):
                    test_output = gr.Text(label="Test")

            btn_view0.click(self._test,
                            inputs=[grp_names[0], grp_names[1], grp_names[2], grp_names[3], grp_names[4], grp_names[5], grp_names[6], grp_names[7],
                                    grp_dis_values[0], grp_dis_values[1], grp_dis_values[2], grp_dis_values[3], grp_dis_values[4], grp_dis_values[5], grp_dis_values[6], grp_dis_values[7]],
                            outputs=[test_output])
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
                        value=self.group_names[0], multiselect=False, label="Group Name for Parity Metrics",
                    )
                    with gr.Row():
                        accuracy_metric = gr.Dropdown(
                            sorted(['Statistical_Bias', 'TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1']),
                            value='Accuracy', multiselect=False, label="Constraint 1 (C1)",
                            scale=2
                        )
                        acc_min_val = gr.Text(value="0.0", label="Min value", scale=1)
                        acc_max_val = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        fairness_metric = gr.Dropdown(
                            sorted(['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity']),
                            value='Equalized_Odds_FPR', multiselect=False, label="Constraint 2 (C2)",
                            scale=2
                        )
                        fairness_min_val = gr.Text(value="-1.0", label="Min value", scale=1)
                        fairness_max_val = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        subgroup_stability_metric = gr.Dropdown(
                            sorted(['Std', 'IQR', 'Jitter', 'Label_Stability']),
                            value='Label_Stability', multiselect=False, label="Constraint 3 (C3)",
                            scale=2
                        )
                        subgroup_stab_min_val = gr.Text(value="0.0", label="Min value", scale=1)
                        subgroup_stab_max_val = gr.Text(value="1.0", label="Max value", scale=1)
                    with gr.Row():
                        group_stability_metrics = gr.Dropdown(
                            sorted(['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity']),
                            value='Label_Stability_Ratio', multiselect=False, label="Constraint 4 (C4)",
                            scale=2
                        )
                        group_stab_min_val = gr.Text(value="0.1", label="Min value", scale=1)
                        group_stab_max_val = gr.Text(value="10.0", label="Max value", scale=1)
                    btn_view1 = gr.Button("Submit")
                with gr.Column(scale=3):
                    bar_plot_for_model_selection = gr.Plot(label="Bar Chart")
                    df_with_models_satisfied_all_constraints = gr.DataFrame(label='Models that satisfy all 4 constraints')

            btn_view1.click(self._create_bar_plot_for_model_selection,
                            inputs=[group_name,
                                    accuracy_metric, acc_min_val, acc_max_val,
                                    fairness_metric, fairness_min_val, fairness_max_val,
                                    subgroup_stability_metric, subgroup_stab_min_val, subgroup_stab_max_val,
                                    group_stability_metrics, group_stab_min_val, group_stab_max_val],
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
                        sorted(['Statistical_Bias', 'TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1']),
                        value=['Accuracy', 'F1'], multiselect=True, label="Accuracy Metrics", info="Select accuracy metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        sorted(['Aleatoric_Uncertainty', 'Overall_Uncertainty']),
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        sorted(['Std', 'IQR', 'Jitter', 'Label_Stability']),
                        value=['Jitter', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    subgroup_btn_view2 = gr.Button("Submit")
                with gr.Column(scale=2):
                    subgroup_model_ranking_heatmap = gr.Plot(label="Heatmap")

            subgroup_btn_view2.click(self._create_subgroup_model_rank_heatmap,
                                     inputs=[model_names, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics, subgroup_tolerance],
                                     outputs=[subgroup_model_ranking_heatmap])
            # ======================================== Parity Metrics Heatmap ========================================
            gr.Markdown(
                """
                ## Parity Metrics Heatmap
                Select input arguments to create a parity metrics heatmap.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    model_names = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[:4], max_choices=5, multiselect=True,
                        label="Model Names", info="Select model names to display on the heatmap:",
                    )
                    group_tolerance = gr.Text(value="0.005", label="Tolerance", info="Define an acceptable tolerance for metric dense ranking.")
                    fairness_metrics = gr.Dropdown(
                        sorted(['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity']),
                        value=['Equalized_Odds_FPR', 'Equalized_Odds_TPR'], multiselect=True, label="Error Parity Metrics", info="Select error parity metrics to display on the heatmap:",
                    )
                    group_stability_metrics = gr.Dropdown(
                        sorted(['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity']),
                        value=['Label_Stability_Ratio', 'Std_Parity'], multiselect=True, label="Stability Parity Metrics", info="Select stability parity metrics to display on the heatmap:",
                    )
                    group_btn_view2 = gr.Button("Submit")
                with gr.Column(scale=2):
                    group_model_ranking_heatmap = gr.Plot(label="Heatmap")

            group_btn_view2.click(self._create_group_model_rank_heatmap,
                                  inputs=[model_names, fairness_metrics, group_stability_metrics, group_tolerance],
                                  outputs=[group_model_ranking_heatmap])
            # =============================== Subgroup and Group Metrics Bar Chart ===============================
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Subgroup Metrics Bar Chart
                        """)
                    subgroup_model_names = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[0], multiselect=False,
                        label="Model Name", info="Select one model to display on the bar chart:",
                    )
                    accuracy_metrics = gr.Dropdown(
                        sorted(['Statistical_Bias', 'TPR', 'TNR', 'PPV', 'FNR', 'FPR', 'Accuracy', 'F1']),
                        value=['Accuracy', 'F1'], multiselect=True, label="Accuracy Metrics", info="Select accuracy metrics to display on the heatmap:",
                    )
                    uncertainty_metrics = gr.Dropdown(
                        sorted(['Aleatoric_Uncertainty', 'Overall_Uncertainty']),
                        value=['Aleatoric_Uncertainty', 'Overall_Uncertainty'], multiselect=True, label="Uncertainty Metrics", info="Select uncertainty metrics to display on the heatmap:",
                    )
                    subgroup_stability_metrics = gr.Dropdown(
                        sorted(['Std', 'IQR', 'Jitter', 'Label_Stability']),
                        value=['Jitter', 'Label_Stability'], multiselect=True, label="Stability Metrics", info="Select stability metrics to display on the heatmap:",
                    )
                    subgroup_btn_view3 = gr.Button("Submit")
                with gr.Column():
                    gr.Markdown(
                        """
                        ## Group Metrics Bar Chart
                        """)
                    group_model_names = gr.Dropdown(
                        sorted(self.model_names), value=sorted(self.model_names)[0], multiselect=False,
                        label="Model Name", info="Select one model to display on the bar chart:",
                    )
                    fairness_metrics = gr.Dropdown(
                        sorted(['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity']),
                        value=['Equalized_Odds_FPR', 'Equalized_Odds_TPR'], multiselect=True, label="Error Parity Metrics", info="Select error parity metrics to display on the heatmap:",
                    )
                    group_stability_metrics = gr.Dropdown(
                        sorted(['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity']),
                        value=['Label_Stability_Ratio', 'Std_Parity'], multiselect=True, label="Stability Parity Metrics", info="Select stability parity metrics to display on the heatmap:",
                    )
                    group_btn_view3 = gr.Button("Submit")
            with gr.Row():
                with gr.Column():
                    subgroup_metrics_bar_chart = gr.Plot(label="Subgroup Bar Chart")
                with gr.Column():
                    group_metrics_bar_chart = gr.Plot(label="Group Bar Chart")

            subgroup_btn_view3.click(self._create_subgroup_metrics_bar_chart_per_one_model,
                                     inputs=[subgroup_model_names, accuracy_metrics, uncertainty_metrics, subgroup_stability_metrics],
                                     outputs=[subgroup_metrics_bar_chart])
            group_btn_view3.click(self._create_group_metrics_bar_chart_per_one_model,
                                  inputs=[group_model_names, fairness_metrics, group_stability_metrics],
                                  outputs=[group_metrics_bar_chart])

        self.demo = demo
        self.demo.launch(inline=False, debug=True, show_error=True)

    def stop_web_app(self):
        self.demo.close()

    def __filter_subgroup_metrics_df(self, results: dict, subgroup_metric: str,
                                     selected_metric: str, selected_subgroup: str, defined_model_names: list):
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

    def _create_bar_plot_for_model_selection(self, group_name, accuracy_metric, acc_min_val, acc_max_val,
                                             fairness_metric, fairness_min_val, fairness_max_val,
                                             subgroup_stability_metric, subgroup_stab_min_val, subgroup_stab_max_val,
                                             group_stability_metrics, group_stab_min_val, group_stab_max_val):
        accuracy_constraint = (accuracy_metric, str_to_float(acc_min_val, 'C1 min value'), str_to_float(acc_max_val, 'C2 max value'))
        fairness_constraint = (fairness_metric, str_to_float(fairness_min_val, 'C2 min value'), str_to_float(fairness_max_val, 'C2 max value'))
        subgroup_stability_constraint = (subgroup_stability_metric, str_to_float(subgroup_stab_min_val, 'C3 min value'), str_to_float(subgroup_stab_max_val, 'C3 max value'))
        group_stability_constraint = (group_stability_metrics, str_to_float(group_stab_min_val, 'C4 min value'), str_to_float(group_stab_max_val, 'C4 max value'))

        # Create individual constraints
        metrics_value_range_dct = dict()
        for constraint in [accuracy_constraint, fairness_constraint, subgroup_stability_constraint, group_stability_constraint]:
            metrics_value_range_dct[constraint[0]] = [constraint[1], constraint[2]]
        # Create intersectional constraints
        metrics_value_range_dct[f'{accuracy_constraint[0]}&{fairness_constraint[0]}'] = None
        metrics_value_range_dct[f'{accuracy_constraint[0]}&{subgroup_stability_constraint[0]}'] = None
        metrics_value_range_dct[f'{accuracy_constraint[0]}&{group_stability_constraint[0]}'] = None
        metrics_value_range_dct[(f'{accuracy_constraint[0]}&{fairness_constraint[0]}'
                                 f'&{subgroup_stability_constraint[0]}&{group_stability_constraint[0]}')] = None

        melted_all_subgroup_metrics_per_model_dct = dict()
        for model_name in self.melted_model_metrics_df['Model_Name'].unique():
            melted_all_subgroup_metrics_per_model_dct[model_name] = (
                self.melted_model_metrics_df)[self.melted_model_metrics_df.Model_Name == model_name]

        melted_all_group_metrics_per_model_dct = dict()
        for model_name in self.melted_model_composed_metrics_df['Model_Name'].unique():
            melted_all_group_metrics_per_model_dct[model_name] = (
                self.melted_model_composed_metrics_df)[self.melted_model_composed_metrics_df.Model_Name == model_name]

        return create_bar_plot_for_model_selection(melted_all_subgroup_metrics_per_model_dct,
                                                   melted_all_group_metrics_per_model_dct,
                                                   metrics_value_range_dct,
                                                   group=group_name)

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
            A list of subgroup accuracy metrics to visualize
        subgroup_uncertainty_metrics
            A list of subgroup uncertainty metrics to visualize
        subgroup_stability_metrics_lst
            A list of subgroup stability metrics to visualize
        tolerance
            An acceptable value difference for metrics dense ranking

        """
        tolerance = str_to_float(tolerance, 'Tolerance')
        metrics_lst = subgroup_accuracy_metrics_lst + subgroup_uncertainty_metrics + subgroup_stability_metrics_lst

        # Find metric values for each model based on metric, subgroup, and model names.
        # Add the values to a results dict.
        results = {}
        num_models = len(model_names)
        for metric in metrics_lst:
            # Add an overall metric
            subgroup_metric = metric
            results = self.__filter_subgroup_metrics_df(results, subgroup_metric, metric,
                                                        selected_subgroup='overall', defined_model_names=model_names)

        model_metrics_matrix = pd.DataFrame(results).T
        model_metrics_matrix = model_metrics_matrix[sorted(model_metrics_matrix.columns)]
        sorted_matrix_by_rank = create_subgroup_sorted_matrix_by_rank(model_metrics_matrix, tolerance)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank)

        return model_rank_heatmap

    def _create_group_model_rank_heatmap(self, model_names: list, group_fairness_metrics_lst: list,
                                         group_stability_metrics_lst: list, tolerance: str):
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
        tolerance
            An acceptable value difference for metrics dense ranking

        """
        tolerance = str_to_float(tolerance, 'Tolerance')

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
                    metric_value = metric_value
                    results[group_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        model_metrics_matrix = model_metrics_matrix[sorted(model_metrics_matrix.columns)]
        sorted_matrix_by_rank = create_sorted_matrix_by_rank(model_metrics_matrix, tolerance)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank)

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
                          legend=alt.Legend(title=metrics_type.capitalize(),
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
