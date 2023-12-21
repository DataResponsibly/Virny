from enum import Enum


class ModelSetting(Enum):
    BATCH = "batch"


class ComputationMode(Enum):
    ERROR_ANALYSIS = "error_analysis"
    POSTPROCESSING_INTERVENTION = "postprocessing_intervention"


INTERSECTION_SIGN = '&'
MODELS_TUNING_SEED = 42
MODELS_TUNING_TEST_SET_FRACTION = 0.2

# Accuracy metrics
MEAN_PREDICTION = 'Mean_Prediction'
STATISTICAL_BIAS = 'Statistical_Bias'
TPR = 'TPR'
TNR = 'TNR'
PPV = 'PPV'
FNR = 'FNR'
FPR = 'FPR'
F1 = 'F1'
ACCURACY = 'Accuracy'
SELECTION_RATE = 'Selection-Rate'
POSITIVE_RATE = 'Positive-Rate'

# Stability metrics
STD = 'Std'
IQR = 'IQR'
JITTER = 'Jitter'
LABEL_STABILITY = 'Label_Stability'

# Uncertainty metrics
ALEATORIC_UNCERTAINTY = 'Aleatoric_Uncertainty'
OVERALL_UNCERTAINTY = 'Overall_Uncertainty'

# Error disparity metrics
EQUALIZED_ODDS_TPR = 'Equalized_Odds_TPR'
EQUALIZED_ODDS_TNR = 'Equalized_Odds_TNR'
EQUALIZED_ODDS_FPR = 'Equalized_Odds_FPR'
EQUALIZED_ODDS_FNR = 'Equalized_Odds_FNR'
DISPARATE_IMPACT = 'Disparate_Impact'
STATISTICAL_PARITY_DIFFERENCE = 'Statistical_Parity_Difference'
ACCURACY_PARITY = 'Accuracy_Parity'

# Stability disparity metrics
LABEL_STABILITY_RATIO = 'Label_Stability_Ratio'
LABEL_STABILITY_DIFFERENCE = 'Label_Stability_Difference'
IQR_PARITY = 'IQR_Parity'
STD_PARITY = 'Std_Parity'
STD_RATIO = 'Std_Ratio'
JITTER_PARITY = 'Jitter_Parity'

# Uncertainty disparity metrics
OVERALL_UNCERTAINTY_PARITY = 'Overall_Uncertainty_Parity'
OVERALL_UNCERTAINTY_RATIO = 'Overall_Uncertainty_Ratio'
ALEATORIC_UNCERTAINTY_PARITY = 'Aleatoric_Uncertainty_Parity'
ALEATORIC_UNCERTAINTY_RATIO = 'Aleatoric_Uncertainty_Ratio'
