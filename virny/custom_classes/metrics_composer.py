import pandas as pd

from virny.configs.constants import *


class MetricsComposer:
    """
    Metric Composer class that combines different subgroup metrics to create disparity metrics
     such as 'Disparate_Impact' or 'Accuracy_Difference'.

    Definitions of the disparity metrics could be observed in the __init__ method of the Metric Composer:
     https://github.com/DataResponsibly/Virny/blob/main/virny/custom_classes/metrics_composer.py

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

        # A dictionary of disparity metrics, where keys are subgroup metrics and values are
        # lists of tuples of the following format: [(<DISPARITY_METRIC_NAME>, <DISPARITY_OPERATION>), ...].
        self.disparity_metric_functions = {
            # Error disparity metrics
            TPR: [(EQUALIZED_ODDS_TPR, self._difference_operation)],
            TNR: [(EQUALIZED_ODDS_TNR, self._difference_operation)],
            FPR: [(EQUALIZED_ODDS_FPR, self._difference_operation)],
            FNR: [(EQUALIZED_ODDS_FNR, self._difference_operation)],
            ACCURACY: [(ACCURACY_DIFFERENCE, self._difference_operation)],
            POSITIVE_RATE: [(DISPARATE_IMPACT, self._ratio_operation)],
            SELECTION_RATE: [(STATISTICAL_PARITY_DIFFERENCE, self._difference_operation)],
            # Stability disparity metrics
            LABEL_STABILITY: [(LABEL_STABILITY_RATIO, self._ratio_operation),
                              (LABEL_STABILITY_DIFFERENCE, self._difference_operation)],
            JITTER: [(JITTER_DIFFERENCE, self._difference_operation)],
            IQR: [(IQR_DIFFERENCE, self._difference_operation)],
            STD: [(STD_DIFFERENCE, self._difference_operation),
                  (STD_RATIO, self._ratio_operation)],
            # Uncertainty disparity metrics
            OVERALL_UNCERTAINTY: [(OVERALL_UNCERTAINTY_DIFFERENCE, self._difference_operation),
                                  (OVERALL_UNCERTAINTY_RATIO, self._ratio_operation)],
            ALEATORIC_UNCERTAINTY: [(ALEATORIC_UNCERTAINTY_DIFFERENCE, self._difference_operation),
                                    (ALEATORIC_UNCERTAINTY_RATIO, self._ratio_operation)],
            EPISTEMIC_UNCERTAINTY: [(EPISTEMIC_UNCERTAINTY_DIFFERENCE, self._difference_operation),
                                    (EPISTEMIC_UNCERTAINTY_RATIO, self._ratio_operation)],
        }

    def _difference_operation(self, cfm, metric_name, dis_group, priv_group):
        return cfm[dis_group][metric_name] - cfm[priv_group][metric_name]

    def _ratio_operation(self, cfm, metric_name, dis_group, priv_group):
        return cfm[dis_group][metric_name] / cfm[priv_group][metric_name]

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
            metric_names = list(cfm['Metric'].unique())
            cfm = cfm.set_index('Metric')

            for sensitive_attr in self.sensitive_attributes_dct.keys():
                dis_group = sensitive_attr + '_dis'
                priv_group = sensitive_attr + '_priv'

                # Compute disparity metrics for each metric in cfm
                groups_metrics_dct[sensitive_attr] = dict()
                for metric_name in metric_names:
                    # Skip a metric that does not have correspondent disparity metrics
                    if metric_name not in self.disparity_metric_functions.keys():
                        continue

                    for disparity_metric_name, disparity_metric_func in self.disparity_metric_functions[metric_name]:
                        groups_metrics_dct[sensitive_attr][disparity_metric_name] = (
                            disparity_metric_func(cfm, metric_name, dis_group, priv_group))

            model_composed_metrics_df = pd.DataFrame(groups_metrics_dct).reset_index()
            model_composed_metrics_df = model_composed_metrics_df.rename(columns={"index": "Metric"})
            model_composed_metrics_df['Model_Name'] = model_name
            models_composed_metrics_df = pd.concat([models_composed_metrics_df, model_composed_metrics_df])

        models_composed_metrics_df = models_composed_metrics_df.reset_index(drop=True)
        return models_composed_metrics_df
