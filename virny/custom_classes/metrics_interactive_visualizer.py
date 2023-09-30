import pandas as pd
import gradio as gr

from virny.utils.data_viz_utils import create_model_rank_heatmap_visualization, create_sorted_matrix_by_rank


class MetricsInteractiveVisualizer:
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
        self.demo = None
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.sensitive_attributes_dct = sensitive_attributes_dct

        # Create one metrics df with all model_dfs
        all_models_metrics_df = pd.DataFrame()
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

        all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)

        self.models_metrics_dct = models_metrics_dct
        self.all_models_metrics_df = all_models_metrics_df
        self.models_composed_metrics_df = models_composed_metrics_df
        self.melted_models_composed_metrics_df = self.models_composed_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                                      var_name="Subgroup",
                                                                                      value_name="Value")
        self.sorted_models_composed_metrics_df = self.melted_models_composed_metrics_df.sort_values(by=['Value'])
        
    def start_web_app(self):
        css = """
        .plot_output1 {position: right !important}
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    fairness_metrics = gr.Dropdown(
                        ['Equalized_Odds_TPR', 'Equalized_Odds_FPR', 'Disparate_Impact', 'Statistical_Parity_Difference', 'Accuracy_Parity'],
                        value=['Equalized_Odds_TPR', 'Equalized_Odds_FPR'], multiselect=True, label="Fairness Metrics", info="Select fairness metrics to display on the heatmap:",
                    )
                    group_stability_metrics = gr.Dropdown(
                        ['Label_Stability_Ratio', 'IQR_Parity', 'Std_Parity', 'Std_Ratio', 'Jitter_Parity'],
                        value=['Label_Stability_Ratio', 'Std_Parity'], multiselect=True, label="Group Stability Metrics", info="Select group stability metrics to display on the heatmap:",
                    )
                    btn = gr.Button("Submit")
                with gr.Column(scale=2):
                    model_ranking_heatmap = gr.Plot(label="Plot")

            btn.click(self._create_model_rank_heatmap,
                      inputs=[fairness_metrics, group_stability_metrics],
                      outputs=[model_ranking_heatmap])

        self.demo = demo
        self.demo.launch(inline=False, debug=True, show_error=True)

    def stop_web_app(self):
        self.demo.close()

    def _create_model_rank_heatmap(self, group_fairness_metrics_lst: list, group_stability_metrics_lst: list):
        """
        Create a model rank heatmap.

        Parameters
        ----------
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
        num_models = len(self.model_names)
        for metric in metrics_lst:
            for group in groups_lst:
                group_metric = metric + '_' + group
                results[group_metric] = dict()
                # Get distinct sorted model names
                sorted_model_names_arr = self.sorted_models_composed_metrics_df[
                    (self.sorted_models_composed_metrics_df.Metric == metric) &
                    (self.sorted_models_composed_metrics_df.Subgroup == group)
                    ]['Model_Name'].values
                # Add values to a results dict
                for idx, model_name in enumerate(sorted_model_names_arr):
                    metric_value = self.sorted_models_composed_metrics_df[
                        (self.sorted_models_composed_metrics_df.Metric == metric) &
                        (self.sorted_models_composed_metrics_df.Subgroup == group) &
                        (self.sorted_models_composed_metrics_df.Model_Name == model_name)
                        ]['Value'].values[0]
                    metric_value = round(metric_value, 3)
                    results[group_metric][model_name] = metric_value

        model_metrics_matrix = pd.DataFrame(results).T
        sorted_matrix_by_rank = create_sorted_matrix_by_rank(model_metrics_matrix)
        model_rank_heatmap, _ = create_model_rank_heatmap_visualization(model_metrics_matrix, sorted_matrix_by_rank, num_models)

        return model_rank_heatmap
