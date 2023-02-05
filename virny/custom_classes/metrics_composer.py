import pandas as pd


class MetricsComposer:
    """
    Composer class that combines different metrics to create new ones such as 'Disparate_Impact' or 'Accuracy_Parity'

    Parameters
    ----------
    models_metrics_dct
        Dictionary where keys are model names and values are dataframes of subgroups metrics for each model
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes

    """
    def __init__(self, models_metrics_dct: dict, sensitive_attributes_dct: dict):
        self.models_metrics_dct = models_metrics_dct
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.models_average_metrics_dct = None  # will be created in self.compose_metrics()

    def compose_metrics(self):
        """
        Compose subgroup metrics from self.model_metrics_df.

        Return a dictionary of composed metrics.
        """
        # Create models_average_metrics_dct
        models_average_metrics_dct = dict()
        for model_name in self.models_metrics_dct.keys():
            columns_to_group = [col for col in self.models_metrics_dct[model_name].columns
                                if col not in ('Model_Seed', 'Run_Number')]
            models_average_metrics_dct[model_name] = self.models_metrics_dct[model_name][columns_to_group].groupby(['Metric', 'Model_Name']).mean().reset_index()

        self.models_average_metrics_dct = models_average_metrics_dct

        groups_metrics_dct = dict()
        models_composed_metrics_df = pd.DataFrame()
        for model_name in self.models_average_metrics_dct.keys():
            cfm = self.models_average_metrics_dct[model_name]
            cfm = cfm.set_index('Metric')

            for sensitive_attr in self.sensitive_attributes_dct.keys():
                dis_group = sensitive_attr + '_dis'
                priv_group = sensitive_attr + '_priv'

                groups_metrics_dct[sensitive_attr] = {
                    # Group statistical bias metrics
                    'Equalized_Odds_TPR': cfm[dis_group]['TPR'] - cfm[priv_group]['TPR'],
                    'Equalized_Odds_FPR': cfm[dis_group]['FPR'] - cfm[priv_group]['FPR'],
                    'Disparate_Impact': cfm[dis_group]['Positive-Rate'] / cfm[priv_group]['Positive-Rate'],
                    'Statistical_Parity_Difference': cfm[dis_group]['Positive-Rate'] - cfm[priv_group]['Positive-Rate'],
                    'Accuracy_Parity': cfm[dis_group]['Accuracy'] - cfm[priv_group]['Accuracy'],
                    # Group variance metrics
                    'Label_Stability_Ratio': cfm[dis_group]['Label_Stability'] / cfm[priv_group]['Label_Stability'],
                    'IQR_Parity': cfm[dis_group]['IQR'] - cfm[priv_group]['IQR'],
                    'Std_Parity': cfm[dis_group]['Std'] - cfm[priv_group]['Std'],
                    'Std_Ratio': cfm[dis_group]['Std'] / cfm[priv_group]['Std'],
                    'Jitter_Parity': cfm[dis_group]['Jitter'] - cfm[priv_group]['Jitter'],
                }

            model_composed_metrics_df = pd.DataFrame(groups_metrics_dct).reset_index()
            model_composed_metrics_df = model_composed_metrics_df.rename(columns={"index": "Metric"})
            model_composed_metrics_df['Model_Name'] = model_name
            models_composed_metrics_df = pd.concat([models_composed_metrics_df, model_composed_metrics_df])

        models_composed_metrics_df = models_composed_metrics_df.reset_index(drop=True)
        return models_composed_metrics_df
